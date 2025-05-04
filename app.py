"""
Open Source LLM Ticket Analysis System
========================================================
Enhanced version with detailed logging and timeout handling
"""

import os
import json
import argparse
import numpy as np
import requests
import time
import logging
from typing import List, Dict, Optional, Union, Any
from datetime import datetime
from pathlib import Path
from functools import wraps
from contextlib import contextmanager

# FastAPI for the server
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# For LLM integration
import torch

# For data preprocessing and analysis
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ticket_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global processing status tracker
processing_status = {
    "is_processing": False,
    "total_tickets": 0,
    "processed_tickets": 0,
    "last_update": None,
    "estimated_completion": None
}

# Create a local directory for NLTK data
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)

# Set the NLTK data path
nltk.data.path.append(nltk_data_dir)

# Download necessary NLTK resources with explicit path
try:
    nltk.download('punkt', download_dir=nltk_data_dir)
    nltk.download('stopwords', download_dir=nltk_data_dir)
    nltk.download('vader_lexicon', download_dir=nltk_data_dir)
except Exception as e:
    logger.error(f"Failed to download NLTK resources: {str(e)}")
    raise

# Timeout decorator
def timeout(seconds=30, error_message="Operation timed out"):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            import asyncio
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                logger.error(f"Timeout occurred in {func.__name__}")
                raise HTTPException(status_code=408, detail=error_message)
        return wrapper
    return decorator

# Context manager for timing operations
@contextmanager
def log_time(operation_name):
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        logger.info(f"{operation_name} completed in {duration:.2f} seconds")

# Define ticket data structures
class Ticket(BaseModel):
    id: Optional[str] = None
    subject: str
    description: str
    status: Optional[str] = "new"
    priority: Optional[str] = "medium"
    created_at: Optional[str] = None
    category: Optional[str] = None
    sentiment: Optional[float] = None
    
class TicketAnalysisResult(BaseModel):
    ticket_id: str
    category: str
    priority: str
    sentiment: float
    summary: str
    suggested_response: str
    analysis_details: Dict[str, Any]

class TicketDataset(BaseModel):
    tickets: List[Ticket]
    categories: Optional[List[str]] = None

# Model-based Analyst class
class TicketAnalyst:
    def __init__(self, model_name: str, max_tokens: int = 2048, temperature: float = 0.1, ollama_base_url: str = "http://localhost:11434"):
        """
        Initialize the ticket analysis system with an Ollama model
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.ollama_base_url = ollama_base_url
        
        # Initialize the sentiment analyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # TF-IDF vectorizer for traditional ML classification
        self.vectorizer = TfidfVectorizer(max_features=10000)
        
        # Classification model (initialized during training)
        self.classifier = None
        self.categories = []
        
        # Check if Ollama is available
        try:
            logger.info(f"Checking Ollama connection at {ollama_base_url}")
            response = requests.get(f"{ollama_base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_tags = [m.get("name") for m in models]
                if model_name in model_tags:
                    logger.info(f"Found {model_name} in Ollama models")
                else:
                    logger.warning(f"{model_name} not found in Ollama models. Available models: {', '.join(model_tags)}")
            else:
                logger.warning(f"Could not connect to Ollama at {ollama_base_url}")
        except Exception as e:
            logger.error(f"Error connecting to Ollama: {e}")
            logger.warning("Make sure Ollama is running and accessible")
        
        logger.info(f"Ticket Analyst initialized with Ollama model: {model_name}")

    def train_classifier(self, tickets: List[Ticket], categories: List[str] = None):
        """Train the ticket category classifier using provided data"""
        logger.info(f"Starting classifier training with {len(tickets)} tickets")
        
        if len(tickets) < 10:
            logger.warning("Very small training dataset. Classification may not be accurate")
            
        # Extract categories if not provided
        if categories is None:
            self.categories = list(set([t.category for t in tickets if t.category]))
            logger.info(f"Extracted {len(self.categories)} categories from tickets")
        else:
            self.categories = categories
            logger.info(f"Using provided {len(self.categories)} categories")
            
        # Prepare training data
        texts = [f"{t.subject} {t.description}" for t in tickets if t.category]
        labels = [t.category for t in tickets if t.category]
        
        if not texts or not labels:
            logger.error("No valid training data found (missing categories)")
            raise ValueError("No valid training data found (missing categories)")
        
        logger.info(f"Prepared {len(texts)} training samples")
        
        # Create multi-label targets
        y = pd.get_dummies(labels, columns=self.categories).values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(texts, y, test_size=0.2, random_state=42)
        
        # Vectorize
        logger.info("Vectorizing training data...")
        with log_time("TF-IDF Vectorization"):
            X_train_vec = self.vectorizer.fit_transform(X_train)
            X_test_vec = self.vectorizer.transform(X_test)
        
        # Train classifier
        logger.info("Training classifier...")
        with log_time("Classifier Training"):
            self.classifier = OneVsRestClassifier(LogisticRegression(max_iter=1000))
            self.classifier.fit(X_train_vec, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test_vec)
        report = classification_report(y_test, y_pred, target_names=self.categories)
        logger.info("Classification Report:\n" + report)
        
        logger.info(f"Classifier trained on {len(texts)} tickets with {len(self.categories)} categories")
        return self.categories
    
    def analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text, returning a value from -1 (negative) to 1 (positive)"""
        try:
            scores = self.sentiment_analyzer.polarity_scores(text)
            logger.debug(f"Sentiment analysis for text (length: {len(text)}): {scores}")
            return scores['compound']
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return 0.0  # Neutral as fallback
    
    def call_ollama(self, prompt: str, max_tokens: int = None, temperature: float = None, stop: List[str] = None) -> dict:
        """Make a call to the Ollama API with timeout"""
        if max_tokens is None:
            max_tokens = self.max_tokens
        if temperature is None:
            temperature = self.temperature
            
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        if stop:
            payload["options"]["stop"] = stop
            
        try:
            logger.debug(f"Sending request to Ollama with {len(prompt)} chars prompt")
            response = requests.post(
                f"{self.ollama_base_url}/api/generate", 
                json=payload,
                timeout=30000  # 30 second timeout
            )
            response.raise_for_status()
            logger.debug("Received response from Ollama")
            return response.json()
        except requests.exceptions.Timeout:
            logger.warning("Request to Ollama timed out after 30 seconds")
            return {"response": "Response timed out. Using simplified processing."}
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            return {"response": f"Error: {str(e)}"}
    
    def categorize_ticket(self, ticket: Ticket) -> str:
        """Categorize a ticket using the trained classifier or LLM if no classifier"""
        try:
            if self.classifier is None or not self.categories:
                logger.debug("Using LLM for zero-shot classification")
                prompt = f"""
                Classify the following support ticket into the most appropriate category.
                
                Ticket Subject: {ticket.subject}
                Ticket Description: {ticket.description}
                
                Category:
                """
                
                result = self.call_ollama(prompt, max_tokens=50, temperature=0.1, stop=[".", "\n"])
                category = result.get("response", "").strip()
                
                # Clean up category (take only the first line or word)
                if "\n" in category:
                    category = category.split("\n")[0]
                if " " in category and len(category.split()) > 1:
                    category = category.split()[0]
                
                logger.debug(f"LLM classified ticket as: {category}")
                return category
            else:
                logger.debug("Using trained classifier for categorization")
                text = f"{ticket.subject} {ticket.description}"
                vec = self.vectorizer.transform([text])
                prediction = self.classifier.predict(vec)
                category_idx = np.argmax(prediction[0])
                category = self.categories[category_idx]
                logger.debug(f"Classifier predicted category: {category}")
                return category
        except Exception as e:
            logger.error(f"Failed to categorize ticket: {e}")
            return "Uncategorized"

    def suggest_priority(self, ticket: Ticket) -> str:
        """Suggest priority based on content and sentiment analysis"""
        try:
            sentiment = self.analyze_sentiment(f"{ticket.subject} {ticket.description}")
            logger.debug(f"Ticket sentiment score: {sentiment}")
            
            # Use LLM to suggest priority
            prompt = f"""
            Based on the following support ticket, determine the priority (low, medium, high, critical).
            Consider urgency, impact, and sentiment.
            
            Ticket Subject: {ticket.subject}
            Ticket Description: {ticket.description}
            Sentiment score: {sentiment} (-1 is very negative, 1 is very positive)
            
            The priority should be:
            """
            
            result = self.call_ollama(prompt, max_tokens=20, temperature=0.1, stop=[".", "\n"])
            priority = result.get("response", "medium").strip().lower()
            
            # Normalize priority
            if "critical" in priority or "urgent" in priority:
                priority = "critical"
            elif "high" in priority:
                priority = "high"
            elif "low" in priority:
                priority = "low"
            else:
                priority = "medium"
            
            logger.debug(f"Suggested priority: {priority}")
            return priority
        except Exception as e:
            logger.error(f"Failed to suggest priority: {e}")
            return "medium"

    def generate_response(self, ticket: Ticket) -> str:
        """Generate a suggested response for the ticket"""
        try:
            category = ticket.category or self.categorize_ticket(ticket)
            
            prompt = f"""
            Write a helpful, concise customer support response for the following ticket:
            
            Subject: {ticket.subject}
            Description: {ticket.description}
            Category: {category}
            
            Response:
            """
            
            result = self.call_ollama(prompt, max_tokens=self.max_tokens, temperature=self.temperature)
            response = result.get("response", "").strip()
            logger.debug(f"Generated response (length: {len(response)})")
            return response
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return "Thank you for your ticket. We are looking into this issue and will respond shortly."

    def summarize_ticket(self, ticket: Ticket) -> str:
        """Generate a short summary of the ticket"""
        try:
            prompt = f"""
            Summarize the following support ticket in one sentence:
            
            Subject: {ticket.subject}
            Description: {ticket.description}
            
            Summary:
            """
            
            result = self.call_ollama(prompt, max_tokens=100, temperature=0.1)
            summary = result.get("response", "").strip()
            logger.debug(f"Generated summary: {summary}")
            return summary
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return f"Ticket about {ticket.subject}"

    def analyze_ticket(self, ticket: Ticket) -> TicketAnalysisResult:
        """Perform comprehensive analysis on a ticket"""
        try:
            logger.info(f"Starting full analysis for ticket: {ticket.subject[:50]}...")
            
            # If ticket doesn't have an ID, generate one
            if not ticket.id:
                ticket.id = f"TICKET-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                
            # If no created_at timestamp, add one
            if not ticket.created_at:
                ticket.created_at = datetime.now().isoformat()
                
            # Analyze sentiment
            sentiment = self.analyze_sentiment(f"{ticket.subject} {ticket.description}")
            ticket.sentiment = sentiment
            
            # Categorize ticket
            if not ticket.category:
                ticket.category = self.categorize_ticket(ticket)
                
            # Suggest priority if not set
            if ticket.priority == "medium":  # Default priority
                ticket.priority = self.suggest_priority(ticket)
                
            # Generate summary and response
            summary = self.summarize_ticket(ticket)
            suggested_response = self.generate_response(ticket)
            
            # Compile additional analysis details
            details = {
                "word_count": len(ticket.description.split()),
                "sentiment_breakdown": self.sentiment_analyzer.polarity_scores(ticket.description),
                "analysis_timestamp": datetime.now().isoformat(),
                "analysis_method": "full"
            }
            
            # Create and return analysis result
            result = TicketAnalysisResult(
                ticket_id=ticket.id,
                category=ticket.category,
                priority=ticket.priority,
                sentiment=sentiment,
                summary=summary,
                suggested_response=suggested_response,
                analysis_details=details
            )
            
            logger.info(f"Completed full analysis for ticket: {ticket.subject[:50]}...")
            return result
        except Exception as e:
            logger.error(f"Error in full ticket analysis: {e}")
            raise

    def quick_analyze_ticket(self, ticket: Ticket) -> TicketAnalysisResult:
        """Perform a faster analysis on a ticket by minimizing LLM calls"""
        try:
            logger.info(f"Starting quick analysis for ticket: {ticket.subject[:50]}...")
            
            # If ticket doesn't have an ID, generate one
            if not ticket.id:
                ticket.id = f"TICKET-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                
            # If no created_at timestamp, add one
            if not ticket.created_at:
                ticket.created_at = datetime.now().isoformat()
                
            # Analyze sentiment (fast operation, keep it)
            sentiment = self.analyze_sentiment(f"{ticket.subject} {ticket.description}")
            ticket.sentiment = sentiment
            
            # Use existing category if available, otherwise use a simple rule-based approach
            if ticket.category:
                category = ticket.category
            else:
                # Try to use the classifier if trained
                if self.classifier is not None and self.categories:
                    text = f"{ticket.subject} {ticket.description}"
                    vec = self.vectorizer.transform([text])
                    prediction = self.classifier.predict(vec)
                    category_idx = np.argmax(prediction[0])
                    category = self.categories[category_idx]
                else:
                    # Simple keyword-based categorization as fallback
                    text = ticket.subject.lower() + " " + ticket.description.lower()
                    if "login" in text or "password" in text or "account" in text or "sign" in text:
                        category = "Authentication"
                    elif "payment" in text or "charge" in text or "bill" in text or "price" in text:
                        category = "Billing"
                    elif "crash" in text or "error" in text or "doesn't work" in text or "broken" in text:
                        category = "Bug"
                    elif "feature" in text or "suggestion" in text or "add" in text:
                        category = "Feature Request"
                    elif "thank" in text or "great" in text or "good" in text or "excellent" in text:
                        category = "Feedback"
                    else:
                        category = "General Support"
            
            ticket.category = category
            
            # Set priority based partly on sentiment
            if ticket.priority != "medium":
                priority = ticket.priority
            else:
                if sentiment < -0.5:
                    priority = "high"
                elif sentiment < -0.2:
                    priority = "medium"
                else:
                    priority = "low"
                
                if "urgent" in ticket.description.lower() or "critical" in ticket.description.lower():
                    priority = "critical"
            
            ticket.priority = priority
            
            # Generate a simple summary instead of using LLM
            summary = f"A {priority} priority {category} ticket regarding {ticket.subject}"
            
            # Create placeholder response
            suggested_response = f"Thank you for contacting support about your {category.lower()} issue. We're reviewing your ticket and will respond shortly."
            
            # Compile analysis details
            details = {
                "word_count": len(ticket.description.split()),
                "sentiment_breakdown": self.sentiment_analyzer.polarity_scores(ticket.description),
                "analysis_timestamp": datetime.now().isoformat(),
                "analysis_method": "quick"
            }
            
            # Create and return analysis result
            result = TicketAnalysisResult(
                ticket_id=ticket.id,
                category=ticket.category,
                priority=ticket.priority,
                sentiment=sentiment,
                summary=summary,
                suggested_response=suggested_response,
                analysis_details=details
            )
            
            logger.info(f"Completed quick analysis for ticket: {ticket.subject[:50]}...")
            return result
        except Exception as e:
            logger.error(f"Error in quick ticket analysis: {e}")
            raise

# FastAPI Server
app = FastAPI(title="Ticket Analysis LLM API", 
              description="Local LLM-based support ticket analysis system",
              version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global analyst instance
ticket_analyst = None

@app.on_event("startup")
async def startup_event():
    """Initialize the ticket analyst on server startup"""
    global ticket_analyst
    
    logger.info("Starting up Ticket Analysis API")
    
    # Get model name and Ollama URL from environment variables
    model_name = os.environ.get("OLLAMA_MODEL", "llama3.1:latest")
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    
    logger.info(f"Initializing with model: {model_name}, Ollama URL: {ollama_url}")
    
    # Initialize the analyst with Ollama model
    ticket_analyst = TicketAnalyst(
        model_name=model_name,
        ollama_base_url=ollama_url
    )
    
    # Load any saved categories
    categories_path = Path("data/categories.json")
    if categories_path.exists():
        try:
            with open(categories_path, "r") as f:
                categories = json.load(f)
                if categories:
                    ticket_analyst.categories = categories
                    logger.info(f"Loaded {len(categories)} categories: {', '.join(categories)}")
        except Exception as e:
            logger.error(f"Failed to load categories: {e}")

@app.get("/")
async def root():
    """API root endpoint"""
    return {"message": "Ticket Analysis LLM API is running", 
            "model": ticket_analyst.model_name if ticket_analyst else "No model loaded"}

@app.post("/analyze_ticket", response_model=TicketAnalysisResult)
@timeout(300)
async def analyze_ticket(ticket: Ticket, quick: bool = Query(False)):
    """Analyze a single support ticket"""
    logger.info(f"Analyzing ticket (quick={quick}): {ticket.subject[:50]}...")
    
    if not ticket_analyst:
        logger.error("Ticket analyst not initialized")
        raise HTTPException(status_code=500, detail="Ticket analyst not initialized")
    
    try:
        if quick:
            logger.debug("Performing quick analysis")
            analysis = ticket_analyst.quick_analyze_ticket(ticket)
        else:
            logger.debug("Performing full analysis")
            analysis = ticket_analyst.analyze_ticket(ticket)
        
        logger.info(f"Analysis completed for ticket: {ticket.subject[:50]}...")
        return analysis
    except Exception as e:
        logger.error(f"Error analyzing ticket: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing ticket: {e}")

@app.post("/bulk_analyze", response_model=List[TicketAnalysisResult])
@timeout(50000)  # 5 minute timeout for bulk operations
async def bulk_analyze(tickets: List[Ticket], background_tasks: BackgroundTasks, quick: bool = Query(False)):
    """Analyze multiple tickets in bulk with option for quick analysis"""
    global processing_status
    logger.info(f"Received bulk analyze request with {len(tickets)} tickets")
    logger.info(f"First ticket sample: {tickets[0].dict() if tickets else 'No tickets'}")
    logger.info(f"Starting bulk analysis of {len(tickets)} tickets (quick={quick})")
    
    if not ticket_analyst:
        logger.error("Ticket analyst not initialized")
        raise HTTPException(status_code=500, detail="Ticket analyst not initialized")
    
    processing_status["is_processing"] = True
    processing_status["total_tickets"] = len(tickets)
    processing_status["processed_tickets"] = 0
    processing_status["last_update"] = datetime.now().isoformat()
    processing_status["estimated_completion"] = None
    
    results = []
    start_time = time.time()
    
    try:
        # Use quick analysis for CSV uploads (typically comes with "quick=True" parameter)
        if quick:
            logger.debug("Using quick analysis mode")
            for i, ticket in enumerate(tickets):
                try:
                    analysis = ticket_analyst.quick_analyze_ticket(ticket)
                    results.append(analysis)
                    
                    # Update processing status
                    processing_status["processed_tickets"] = i + 1
                    processing_status["last_update"] = datetime.now().isoformat()
                    
                    # Calculate estimated completion time
                    if i > 0:
                        elapsed = time.time() - start_time
                        tickets_per_second = (i + 1) / elapsed
                        remaining_tickets = len(tickets) - (i + 1)
                        estimated_seconds = remaining_tickets / tickets_per_second if tickets_per_second > 0 else 0
                        processing_status["estimated_completion"] = f"{estimated_seconds:.0f} seconds remaining"
                        
                    # Log progress every 10 tickets or 10%
                    if (i + 1) % 10 == 0 or (i + 1) / len(tickets) >= 0.1:
                        logger.info(f"Processed {i + 1}/{len(tickets)} tickets ({(i + 1)/len(tickets)*100:.1f}%)")
                        
                except Exception as e:
                    logger.error(f"Error processing ticket {i}: {e}")
                    continue
        else:
            logger.debug("Using full analysis mode")
            for i, ticket in enumerate(tickets):
                try:
                    analysis = ticket_analyst.analyze_ticket(ticket)
                    results.append(analysis)
                    
                    # Update processing status
                    processing_status["processed_tickets"] = i + 1
                    processing_status["last_update"] = datetime.now().isoformat()
                    
                    # Calculate estimated completion time
                    if i > 0:
                        elapsed = time.time() - start_time
                        tickets_per_second = (i + 1) / elapsed
                        remaining_tickets = len(tickets) - (i + 1)
                        estimated_seconds = remaining_tickets / tickets_per_second if tickets_per_second > 0 else 0
                        processing_status["estimated_completion"] = f"{estimated_seconds:.0f} seconds remaining"
                        
                    # Log progress every 10 tickets or 10%
                    if (i + 1) % 10 == 0 or (i + 1) / len(tickets) >= 0.1:
                        logger.info(f"Processed {i + 1}/{len(tickets)} tickets ({(i + 1)/len(tickets)*100:.1f}%)")
                        
                except Exception as e:
                    logger.error(f"Error processing ticket {i}: {e}")
                    continue
    finally:
        # Mark processing as complete
        processing_status["is_processing"] = False
        processing_status["estimated_completion"] = "Complete"
        logger.info(f"Bulk analysis completed. Processed {len(results)}/{len(tickets)} tickets successfully")
    
    # Save results to file in background
    background_tasks.add_task(save_analysis_results, results)
    
    return results

@app.post("/upload_csv")
@timeout(60)  # 1 minute timeout for uploads
async def upload_csv_tickets(
    file: UploadFile = File(...),
    subject_col: str = Form("subject"),
    description_col: str = Form("description"),
    category_col: Optional[str] = Form(None),
    priority_col: Optional[str] = Form(None)
):
    """Upload a CSV file of tickets for bulk processing"""
    logger.info(f"Starting CSV upload: {file.filename}")
    
    if not file.filename.endswith('.csv'):
        logger.error("Non-CSV file uploaded")
        return JSONResponse(
            status_code=400,
            content={"error": "Only CSV files are supported"}
        )
    
    try:
        # Read the CSV file
        logger.debug("Reading CSV content")
        content = await file.read()
        
        logger.debug("Parsing CSV data")
        with log_time("CSV Parsing"):
            df = pd.read_csv(pd.io.common.BytesIO(content))
        
        # Validate required columns
        if subject_col not in df.columns or description_col not in df.columns:
            logger.error(f"Missing required columns. Need {subject_col} and {description_col}")
            return JSONResponse(
                status_code=400, 
                content={"error": f"Required columns missing. Need {subject_col} and {description_col}"}
            )
        
        logger.info(f"CSV contains {len(df)} rows with columns: {', '.join(df.columns)}")
        
        # Convert to Ticket objects
        tickets = []
        for _, row in df.iterrows():
            try:
                # Handle the possibility of category or priority columns not existing in dataframe
                category = None
                if category_col and category_col in df.columns:
                    category = row.get(category_col)
                    
                priority = "medium"
                if priority_col and priority_col in df.columns:
                    priority = row.get(priority_col, "medium")
                    
                ticket = Ticket(
                    subject=str(row[subject_col]),
                    description=str(row[description_col]),
                    category=category,
                    priority=priority
                )
                tickets.append(ticket)
            except Exception as e:
                logger.warning(f"Failed to process row {_}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(tickets)} tickets from CSV")
        
        # For CSV uploads, immediately perform a quick analysis on a small batch
        # This provides immediate results without waiting for full LLM analysis
        quick_results = []
        if tickets:
            # Limit to first 3 tickets for initial quick analysis
            sample_tickets = tickets[:min(3, len(tickets))]
            logger.debug(f"Performing quick analysis on {len(sample_tickets)} sample tickets")
            
            for ticket in sample_tickets:
                try:
                    quick_results.append(ticket_analyst.quick_analyze_ticket(ticket))
                except Exception as e:
                    logger.error(f"Failed quick analysis for sample ticket: {e}")
                    continue
        
        return {
            "message": f"Successfully loaded {len(tickets)} tickets from CSV",
            "tickets": tickets,
            "sample_analysis": quick_results
        }
    except Exception as e:
        logger.error(f"Error processing CSV upload: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing CSV: {str(e)}"}
        )

@app.get("/processing_status")
async def get_processing_status():
    """Get the current ticket processing status"""
    global processing_status
    return processing_status

@app.post("/train_classifier")
async def train_classifier(dataset: TicketDataset):
    """Train the ticket classifier with provided data"""
    logger.info(f"Received dataset: {dataset.json()}")  # Log the received dataset
    if not ticket_analyst:
        raise HTTPException(status_code=500, detail="Ticket analyst not initialized")

    if len(dataset.tickets) < 10:
        return JSONResponse(
            status_code=400,
            content={"error": "Need at least 10 categorized tickets for training"}
        )

    try:
        categories = ticket_analyst.train_classifier(
            tickets=dataset.tickets,
            categories=dataset.categories
        )

        # Save categories for future use
        os.makedirs("data", exist_ok=True)
        with open("data/categories.json", "w") as f:
            json.dump(categories, f)

        return {"message": f"Classifier trained on {len(dataset.tickets)} tickets",
                "categories": categories}
    except Exception as e:
        logger.error(f"Error training classifier: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model_info")
async def model_info():
    """Get information about the loaded LLM model"""
    if not ticket_analyst:
        raise HTTPException(status_code=500, detail="No model loaded")
    
    # Check if Ollama is accessible
    try:
        response = requests.get(f"{ticket_analyst.ollama_base_url}/api/tags", timeout=10)
        available_models = []
        if response.status_code == 200:
            models = response.json().get("models", [])
            available_models = [m.get("name") for m in models]
    except:
        available_models = ["Could not fetch models from Ollama"]
    
    return {
        "model_name": ticket_analyst.model_name,
        "ollama_url": ticket_analyst.ollama_base_url,
        "available_models": available_models,
        "categories": ticket_analyst.categories,
        "classifier_trained": ticket_analyst.classifier is not None
    }

@app.get("/ui")
@app.get("/ui/")
async def read_ui():
    return FileResponse('static/index.html')


os.makedirs("static", exist_ok=True)

# Mount static files for the web interface
#app.mount("/ui", StaticFiles(directory="static", html=True), name="static")
app.mount("/static", StaticFiles(directory="static"), name="static")

async def save_analysis_results(results: List[TicketAnalysisResult]):
    """Save analysis results to a file"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("results", exist_ok=True)
        
        # Convert to dict for JSON serialization
        results_dict = [result.dict() for result in results]
        
        file_path = f"results/analysis_{timestamp}.json"
        with open(file_path, "w") as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Saved analysis results to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save analysis results: {e}")

if __name__ == "__main__":
    # Command line interface
    def main():
        parser = argparse.ArgumentParser(description="Ticket Analysis LLM Server")
        parser.add_argument("--model", type=str, default="llama3.1:latest", help="Ollama model name (e.g., llama3, mistral)")
        parser.add_argument("--ollama-url", type=str, default="http://localhost:11434", help="Ollama API URL")
        parser.add_argument("--port", type=int, default=8000, help="Port for the API server")
        parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for the API server")
        parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
        args = parser.parse_args()
        
        # Set logging level
        logger.setLevel(args.log_level)
        
        # Set environment variables for the Ollama model
        os.environ["OLLAMA_MODEL"] = args.model
        os.environ["OLLAMA_URL"] = args.ollama_url
        
        import uvicorn
        logger.info(f"Starting Ticket Analysis LLM Server on {args.host}:{args.port}")
        logger.info(f"Using Ollama model: {args.model}")
        logger.info(f"Ollama API URL: {args.ollama_url}")
        uvicorn.run("app:app", host=args.host, port=args.port, reload=True)

    main()