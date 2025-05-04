Ticket Analysis System
An open-source customer support ticket analyzer using local LLMs through Ollama. Analyze sentiment, categorize issues, suggest responses, and prioritize tickets - all while keeping data private.
Features

🤖 Local LLM-powered ticket analysis
📊 Sentiment analysis and ticket categorization
💬 Automated response suggestions
📈 Bulk processing with real-time tracking
🔒 Complete data privacy - no API calls

Quick Setup
Backend (Python/FastAPI)
# Clone repository
git clone https://github.com/yourusername/ticket-analysis-system
cd ticket-analysis-system/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start server (ensure Ollama is running)
python app.py --model llama3.1:latest --ollama-url http://localhost:11434

Frontend (React)
# Navigate to frontend folder
cd ../frontend

# Install dependencies
npm install

# Start development server
npm start

Visit http://localhost:3000 to use the application.

Technologies

Backend: Python, FastAPI, NLTK, Scikit-learn
Frontend: React, Material UI
LLM: Ollama (local language models like Llama 3.1)
