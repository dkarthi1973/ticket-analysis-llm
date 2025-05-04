"""
Test script for the Ticket Analysis LLM System
This script performs several tests to validate the functionality of the system
"""

import requests
import pandas as pd
import json
import time
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
SAMPLE_CSV = "sample_tickets.csv"

def test_server_connection():
    """Test basic connection to the server"""
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Server connection successful: {data['message']}")
            print(f"   Using model: {data['model']}")
            return True
        else:
            print(f"❌ Server connection failed with status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Server connection error: {str(e)}")
        return False

def test_model_info():
    """Test retrieving model information"""
    try:
        response = requests.get(f"{BASE_URL}/model_info")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model info retrieved successfully:")
            print(f"   Model name: {data['model_name']}")
            print(f"   Ollama URL: {data['ollama_url']}")
            print(f"   Available models: {', '.join(data['available_models'])}")
            print(f"   Classifier trained: {data['classifier_trained']}")
            return True
        else:
            print(f"❌ Model info failed with status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model info error: {str(e)}")
        return False

def test_single_ticket_analysis():
    """Test analyzing a single ticket"""
    test_ticket = {
        "subject": "Cannot log into my account",
        "description": "I've been trying to log in for the past hour but keep getting an 'Invalid password' error. I'm sure I'm using the correct password as it works on the mobile app."
    }
    
    try:
        print("Testing single ticket analysis...")
        print(f"Subject: {test_ticket['subject']}")
        print(f"Description: {test_ticket['description'][:100]}...")
        
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/analyze_ticket",
            json=test_ticket
        )
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Single ticket analysis successful ({end_time - start_time:.2f} seconds)")
            print(f"   Category: {result['category']}")
            print(f"   Priority: {result['priority']}")
            print(f"   Sentiment: {result['sentiment']:.2f}")
            print(f"   Summary: {result['summary']}")
            print(f"   Response (first 100 chars): {result['suggested_response'][:100]}...")
            return True
        else:
            print(f"❌ Single ticket analysis failed with status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Single ticket analysis error: {str(e)}")
        return False

def test_csv_upload():
    """Test uploading and processing a CSV file"""
    if not Path(SAMPLE_CSV).exists():
        print(f"❌ Sample CSV file not found: {SAMPLE_CSV}")
        return False
    
    try:
        # Read the first 3 rows for a smaller test
        df = pd.read_csv(SAMPLE_CSV)
        print(f"CSV contains {len(df)} tickets")
        
        # Use a smaller sample for testing
        test_sample = df.head(3)
        test_sample.to_csv("test_sample.csv", index=False)
        
        # Upload the CSV
        with open("test_sample.csv", "rb") as f:
            files = {"file": ("test_sample.csv", f, "text/csv")}
            response = requests.post(
                f"{BASE_URL}/upload_csv",
                files=files,
                data={
                    "subject_col": "subject",
                    "description_col": "description",
                    "category_col": "category",
                    "priority_col": "priority"
                }
            )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ CSV upload successful")
            print(f"   Loaded {len(result['tickets'])} tickets")
            
            # Test bulk analysis with a small subset
            if len(result['tickets']) > 0:
                return test_bulk_analysis(result['tickets'][:2])
            return True
        else:
            print(f"❌ CSV upload failed with status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ CSV upload error: {str(e)}")
        return False

def test_bulk_analysis(tickets):
    """Test bulk ticket analysis"""
    try:
        print(f"Testing bulk analysis with {len(tickets)} tickets...")
        
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/bulk_analyze",
            json=tickets
        )
        end_time = time.time()
        
        if response.status_code == 200:
            results = response.json()
            print(f"✅ Bulk analysis successful ({end_time - start_time:.2f} seconds)")
            print(f"   Analyzed {len(results)} tickets")
            
            # Show sample results from the first ticket
            if results:
                print(f"   Sample result from first ticket:")
                print(f"   Category: {results[0]['category']}")
                print(f"   Priority: {results[0]['priority']}")
                print(f"   Sentiment: {results[0]['sentiment']:.2f}")
            return True
        else:
            print(f"❌ Bulk analysis failed with status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Bulk analysis error: {str(e)}")
        return False

def test_classifier_training():
    """Test training the classifier"""
    if not Path(SAMPLE_CSV).exists():
        print(f"❌ Sample CSV file not found: {SAMPLE_CSV}")
        return False
    
    try:
        # Load the dataset
        df = pd.read_csv(SAMPLE_CSV)
        
        # Convert to the format expected by the API
        tickets = []
        for _, row in df.iterrows():
            tickets.append({
                "subject": row["subject"],
                "description": row["description"],
                "category": row["category"],
                "priority": row["priority"]
            })
        
        dataset = {
            "tickets": tickets,
            "categories": df["category"].unique().tolist()
        }
        
        print(f"Testing classifier training with {len(tickets)} tickets and {len(dataset['categories'])} categories...")
        
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/train_classifier",
            json=dataset
        )
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Classifier training successful ({end_time - start_time:.2f} seconds)")
            print(f"   Message: {result['message']}")
            print(f"   Categories: {', '.join(result['categories'])}")
            return True
        else:
            print(f"❌ Classifier training failed with status code: {response.status_code}")
            if response.text:
                print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Classifier training error: {str(e)}")
        return False

def run_all_tests():
    """Run all tests and report results"""
    print("=" * 50)
    print("TICKET ANALYSIS LLM SYSTEM TEST")
    print("=" * 50)
    
    # Track test results
    results = {
        "Server Connection": test_server_connection(),
        "Model Info": test_model_info(),
        "Single Ticket Analysis": test_single_ticket_analysis(),
        "CSV Upload": test_csv_upload(),
        "Classifier Training": test_classifier_training()
    }
    
    # Print summary
    print("\n" + "=" * 50)
    print("TEST RESULTS SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test_name, result in results.items():
        status = "PASSED" if result else "FAILED"
        if not result:
            all_passed = False
        print(f"{test_name}: {status}")
    
    print("\nOVERALL: " + ("PASSED" if all_passed else "FAILED"))
    print("=" * 50)

if __name__ == "__main__":
    run_all_tests()