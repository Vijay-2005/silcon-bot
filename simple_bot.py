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

# Check if the query is inappropriate (narrower list, only blocks clearly inappropriate topics)
def is_inappropriate(query):
    inappropriate_patterns = [
        r'\bporn\b', r'\bxxx\b', r'\bhack\b', r'\bcrack\b', r'\billegal\b', 
        r'\bdrug dealer\b', r'\bterrorist\b', r'\blaunder money\b'
    ]
    
    for pattern in inappropriate_patterns:
        if re.search(pattern, query.lower()):
            return True
    return False

# Helper function to check if the AI response indicates it doesn't know the answer
def needs_contact_info(response_text):
    uncertainty_patterns = [
        r"I don't know", r"I don't have", r"I'm not sure", r"I am not sure",
        r"I cannot provide", r"I can't provide", r"I do not have",
        r"don't have information", r"don't have the information",
        r"no information", r"insufficient information", 
        r"cannot answer", r"can't answer", r"unable to answer",
        r"don't have enough details", r"would need more information",
        r"not able to access", r"don't have access", r"beyond my capabilities",
        r"limited knowledge", r"can't determine", r"cannot determine",
        r"you should contact", r"reach out to", r"get in touch with support"
    ]
    
    for pattern in uncertainty_patterns:
        if re.search(pattern, response_text, re.IGNORECASE):
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
        
        # Check if the question is inappropriate
        if is_inappropriate(user_message):
            response_text = "I'm sorry, but I cannot assist with inappropriate or illegal topics. " + CONTACT_INFO
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

GUIDELINES:
1. Answer a wide range of questions to the best of your ability, even if they're somewhat outside typical customer support topics.
2. Try to find a helpful answer whenever possible.
3. If you genuinely don't know the answer or can't help with something, explain why and suggest contacting us directly.
4. Always be professional, friendly, and helpful in your tone.

User's question: {user_message}"""
        
        # Call Gemini API
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        
        # Format response
        response_text = response.text
        
        # Add contact info if the response seems unsure
        if needs_contact_info(response_text):
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