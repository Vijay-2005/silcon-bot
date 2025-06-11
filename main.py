from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import json
import google.generativeai as genai
from datetime import datetime
import re

app = FastAPI()

# Configure CORS for web integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models for request/response data
class Message(BaseModel):
    role: str  # "user" or "agent"
    content: str
    timestamp: Optional[str] = None

class Conversation(BaseModel):
    messages: List[Message]
    metadata: Optional[Dict[str, Any]] = None

class AgentConfig(BaseModel):
    name: str
    instructions: str
    knowledge_base: Optional[str] = None

# Contact information to provide when the bot can't answer
CONTACT_INFO = """
For further assistance, please contact us at:
Email: siliconsynapse8@gmail.com
24/7 Support: +91 7668055685
Business Hours: Mon-Fri, 9am-6pm
"""

# Configure Gemini API
@app.on_event("startup")
async def startup_event():
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        print("Warning: GEMINI_API_KEY environment variable not set")
    else:
        genai.configure(api_key=gemini_api_key)

# Helper function to format conversation for Gemini
def format_conversation_for_gemini(conversation: Conversation, agent_config: AgentConfig):
    system_prompt = f"""You are {agent_config.name}, an AI assistant from Silicon Synapse.
Instructions: {agent_config.instructions}

IMPORTANT GUIDELINES:
1. You represent Silicon Synapse, a platform offering various AI agents for different specialized tasks.
2. Only answer questions about our AI services, agent capabilities, subscription plans, and related inquiries.
3. If asked about topics unrelated to our AI platform like mathematics, algebra, science, or personal advice, politely explain you're an assistant for Silicon Synapse AI services.
4. If you don't know the answer or can't help with a question, provide our contact information.
5. Always be professional, concise, and helpful within your scope.
"""
    if agent_config.knowledge_base:
        system_prompt += f"\nReference knowledge: {agent_config.knowledge_base}"
    
    formatted_messages = [{"role": "system", "parts": [system_prompt]}]
    
    for msg in conversation.messages:
        role = "user" if msg.role == "user" else "model"
        formatted_messages.append({"role": role, "parts": [msg.content]})
    
    return formatted_messages

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

# Helper function to check if the AI response indicates it doesn't know the answer
def needs_contact_info(response_text):
    uncertainty_patterns = [
        r"I don't know", r"I don't have", r"I'm not sure", r"I am not sure",
        r"I cannot provide", r"I can't provide", r"I do not have",
        r"don't have information", r"don't have the information",
        r"no information", r"insufficient information", 
        r"cannot answer", r"can't answer", r"unable to answer"
    ]
    
    for pattern in uncertainty_patterns:
        if re.search(pattern, response_text, re.IGNORECASE):
            return True
    return False

# Endpoint to handle customer support queries
@app.post("/api/support")
async def handle_support_query(conversation: Conversation, agent_config: AgentConfig):
    try:
        # Ensure Gemini API key is configured
        if not os.environ.get("GEMINI_API_KEY"):
            raise HTTPException(status_code=500, detail="Gemini API key not configured")
        
        # Get the last user message
        last_user_message = ""
        for msg in reversed(conversation.messages):
            if msg.role == "user":
                last_user_message = msg.content
                break
        
        # Check if the question is off-topic
        if is_off_topic(last_user_message):
            response_text = "I'm sorry, but I can only assist with questions related to our products, services, and customer support. I cannot help with topics like algebra or other academic subjects. " + CONTACT_INFO
            agent_message = Message(
                role="agent",
                content=response_text,
                timestamp=datetime.now().isoformat()
            )
            return {"response": agent_message}
        
        # Format the conversation for Gemini
        formatted_messages = format_conversation_for_gemini(conversation, agent_config)
        
        # Get response from Gemini
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(formatted_messages)
        
        response_text = response.text
        
        # Check if the AI doesn't know the answer and add contact info if needed
        if needs_contact_info(response_text):
            if not response_text.endswith(CONTACT_INFO):
                response_text += CONTACT_INFO
        
        # Create response message
        agent_message = Message(
            role="agent",
            content=response_text,
            timestamp=datetime.now().isoformat()
        )
        
        # Return the agent's response
        return {"response": agent_message}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Get available agent configurations
@app.get("/api/agents")
async def get_available_agents():
    # In a real implementation, this might come from a database
    # For this example, we'll return a static list
    return {
        "agents": [
            {
                "id": "platform_guide",
                "name": "Silicon Synapse Guide",
                "description": "Helps you navigate our AI platform and find the right agent for your needs."
            },
            {
                "id": "technical_support",
                "name": "Technical Support Agent",
                "description": "Assists with technical issues related to our AI agents and platform integration."
            },
            {
                "id": "ai_consultant",
                "name": "AI Solution Consultant",
                "description": "Provides information about our AI capabilities, pricing plans, and custom solutions."
            },
            {
                "id": "developer_support",
                "name": "Developer Support",
                "description": "Helps developers with API integration, documentation, and implementation questions."
            }
        ]
    }

# Health check endpoint
@app.get("/api/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)