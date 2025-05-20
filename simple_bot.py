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

# Handle special request patterns
def handle_special_requests(query):
    # Request for contact/support email
    if re.search(r'(mail|email|contact|support).*(support|contact|help|assist)', query.lower()):
        return f"Here is our support contact information:{CONTACT_INFO}"
    
    # How can you help me
    if re.search(r'how.*(can|could).*help.*me', query.lower()) or re.search(r'what.*can.*you.*do', query.lower()):
        return """I can help you with a variety of tasks related to our products and services:

1. Product information and recommendations
2. Troubleshooting technical issues
3. Order status and tracking
4. Account management
5. Billing questions
6. Return and refund policies
7. Installation and setup guidance
8. Feature explanations and tutorials

Feel free to ask me about any of these topics or anything else you need!"""
    
    # Return None if no special request patterns match
    return None

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
        
        # Check for special requests that don't need AI processing
        special_response = handle_special_requests(user_message)
        if special_response:
            return {
                "response": {
                    "role": "agent",
                    "content": special_response,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        # Prepare prompt for Gemini with specific instructions
        prompt = f"""As a customer support agent named {agent_config.name}, your role is to:
{agent_config.instructions}

IMPORTANT INSTRUCTIONS:
1. Answer directly and specifically to the user's question. Do NOT give generic responses.
2. If asked how you can help, provide specific examples of the services and assistance you can offer.
3. If asked for contact information or email, provide the contact details directly.
4. Provide detailed, helpful information whenever possible.
5. Be conversational but focused on giving useful information.
6. Never respond with just "Thank you for your message. How else can I help?"

User's question: "{user_message}"

Provide a focused, specific answer to this exact question:"""
        
        # Call Gemini API
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        
        # Format response
        response_text = response.text
        
        # Verify the response isn't too generic
        generic_patterns = [
            r"^Thank you for your message\W+How (else )?can I help you",
            r"^Thanks for reaching out\W+How (else )?can I assist you",
            r"^How (else )?can I help you today\W*$"
        ]
        
        is_generic = False
        for pattern in generic_patterns:
            if re.search(pattern, response_text, re.IGNORECASE):
                is_generic = True
                break
                
        # If response is too generic, provide more specific help
        if is_generic:
            response_text = f"""I'd be happy to help you more specifically. 

For questions about our products and services, I can provide details on features, pricing, and compatibility.

For technical support, I can help troubleshoot issues or provide guidance on using our products.

For account or billing questions, I can explain our policies and options.

{CONTACT_INFO}

Please let me know what specific information you're looking for, and I'll assist you right away."""
        
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