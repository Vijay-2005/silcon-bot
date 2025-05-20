import requests
import json

# Check if server is running
def test_health():
    try:
        response = requests.get("http://localhost:8000/api/health")
        print(f"Server status: {response.json()}")
        return True
    except Exception as e:
        print(f"Server not running! Error: {e}")
        return False

# Test the chatbot with different queries
def test_chatbot():
    url = "http://localhost:8000/api/support"
    
    test_cases = ["who made you",
        "How do I track my order?",
        "Solve for x: 2x + 5 = 15",
        "I can't find my order confirmation email"
    ]
    
    for query in test_cases:
        print(f"\n--- Testing: '{query}' ---")
        
        payload = {
            "conversation": {
                "messages": [{"role": "user", "content": query}]
            },
            "agent_config": {
                "name": "Support Agent", 
                "instructions": "You help customers with their orders."
            }
        }
        
        try:
            response = requests.post(url, json=payload)
            print(f"Status code: {response.status_code}")
            print(f"Full response: {response.text}")
            
            # Try to parse as JSON if possible
            if response.headers.get('content-type') == 'application/json':
                print(f"JSON response: {json.dumps(response.json(), indent=2)}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    if test_health():
        test_chatbot()