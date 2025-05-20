from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import json
import google.generativeai as genai
from datetime import datetime
import re

# Configure Gemini API directly with key (for testing only)
genai.configure(api_key="AIzaSyCGtYwxvjzkl5T8fQH1f4cj26T5T6zM7FU")

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

# Helper function to format conversation for Gemini
def format_conversation_for_gemini(conversation: Conversation, agent_config: AgentConfig):
    # Create the instructions text
    instructions = f"""You are {agent_config.name}, a customer support agent.
Instructions: {agent_config.instructions}

IMPORTANT GUIDELINES:
1. Only answer questions that are relevant to customer support, products, services, and business inquiries.
2. If asked about topics like mathematics, algebra, science, or any other unrelated subjects, politely decline to answer and explain you're a customer support agent.
3. If you don't know the answer or can't help with a question, provide our contact information.
4. Always be professional, concise, and helpful within your scope.
"""
    if agent_config.knowledge_base:
        instructions += f"\nReference knowledge: {agent_config.knowledge_base}"
    
    # Create a list for messages without using system role
    formatted_messages = []
    
    # Add the first user message with instructions
    first_user_message = None
    for msg in conversation.messages:
        if msg.role == "user":
            first_user_message = msg
            break
    
    if first_user_message:
        # Add instructions to the user's first message
        formatted_messages.append({
            "role": "user",
            "parts": [f"{instructions}\n\nUser question: {first_user_message.content}"]
        })
        
        # Add all other messages
        added_first = False
        for msg in conversation.messages:
            if msg.role == "user" and not added_first:
                added_first = True
                continue  # Skip the first user message since we already added it with instructions
            
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
        model = genai.GenerativeModel('gemini-1.5-flash')  # Using the same model as in testapi.py
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
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Get available agent configurations
@app.get("/api/agents")
async def get_available_agents():
    # In a real implementation, this might come from a database
    # For this example, we'll return a static list
    return {
        "agents": [
            {
                "id": "general_support",
                "name": "General Support Agent",
                "description": "A general customer support agent that can handle a wide range of queries."
            },
            {
                "id": "technical_support",
                "name": "Technical Support Agent",
                "description": "Specialized in solving technical issues and product troubleshooting."
            },
            {
                "id": "sales_agent",
                "name": "Sales Agent",
                "description": "Helps with product inquiries, pricing, and purchasing decisions."
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