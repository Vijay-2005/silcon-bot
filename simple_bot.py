from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import google.generativeai as genai
from datetime import datetime
import re
import os
from dotenv import load_dotenv

# Load environment variables from .env file (if present)
load_dotenv()

# Configure Gemini API with environment variable
gemini_api_key = os.environ.get("GEMINI_API_KEY")
if not gemini_api_key:
    print("WARNING: GEMINI_API_KEY environment variable not set")
    print("The API will not work until you set this variable")
else:
    genai.configure(api_key=gemini_api_key)

app = FastAPI()

# Configure CORS for web integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Contact information
CONTACT_INFO = """
For further assistance, please contact us at:
Email: siliconsynapse8@gmail.com
24/7 Support: +91 7668055685
Business Hours: Mon-Fri, 9am-6pm
"""

# Models for request/response data
class Message(BaseModel):
    role: str  # "user" or "agent"
    content: str
    timestamp: Optional[str] = None

class Conversation(BaseModel):
    messages: List[Message]

class AgentConfig(BaseModel):
    name: str
    instructions: str

# Check if the query is off-topic (like algebra)
def is_off_topic(query):
    off_topic_patterns = [
        r'\balgebra\b', r'\bequation\b', r'\bmath\b', r'\bmathematics\b',
        r'\bcalculus\b', r'\bphysics\b', r'\bchemistry\b', r'\bhistory\b',
        r'\bformula\b', r'\bsolve\s+for\b', r'\bcompute\b', r'\bderivative\b'
    ]
    
    for pattern in off_topic_patterns:
        if re.search(pattern, query.lower()):
            return True
    return False

# Simple endpoint for customer support
@app.post("/api/support")
async def handle_support_query(conversation: Conversation, agent_config: AgentConfig):
    try:
        # Check if API key is configured
        if not gemini_api_key:
            raise HTTPException(status_code=500, 
                                detail="GEMINI_API_KEY environment variable not set")
        
        # Get the user's question
        if not conversation.messages or len(conversation.messages) == 0:
            raise HTTPException(status_code=400, detail="No messages provided")
        
        user_message = conversation.messages[0].content
        
        # Check if the question is off-topic
        if is_off_topic(user_message):
            response_text = "I'm sorry, but I can only assist with questions related to our products, services, and customer support. I cannot help with topics like algebra or other academic subjects. " + CONTACT_INFO
            return {
                "response": {
                    "role": "agent",
                    "content": response_text,
                    "timestamp": datetime.now().isoformat()
                }
            }
        
        # Prepare prompt for Gemini
        prompt = f"""As a customer support agent named {agent_config.name}, your role is to:
{agent_config.instructions}

Only answer questions about products, services, or business inquiries.
Be professional and helpful, and provide contact information if you can't answer.

User's question: {user_message}"""
        
        # Call Gemini API
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        
        # Format response
        response_text = response.text
        
        # Add contact info if the response seems unsure
        if "I don't know" in response_text or "I'm not sure" in response_text:
            if not response_text.endswith(CONTACT_INFO):
                response_text += CONTACT_INFO
        
        # Return formatted response
        return {
            "response": {
                "role": "agent",
                "content": response_text,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/api/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 