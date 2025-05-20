import requests
import json

def test_support():
    url = "http://localhost:8000/api/support"
    
    queries = [
        "How do I track my order?",
        "Can I return an item after 30 days?",
        "Solve for x: 2x + 5 = 15",  # Should be rejected as algebra
        "I lost my order confirmation email"
    ]
    
    for query in queries:
        print(f"\n--- Testing: '{query}' ---")
        
        payload = {
            "conversation": {
                "messages": [
                    {"role": "user", "content": query}
                ]
            },
            "agent_config": {
                "name": "Support Agent",
                "instructions": "You help customers with their orders and answer questions about products."
            }
        }
        
        try:
            response = requests.post(url, json=payload)
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                # Limit the output to make it readable
                content = data['response']['content']
                if len(content) > 100:
                    content = content[:100] + "..."
                print(f"Response: {content}")
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Exception: {str(e)}")

if __name__ == "__main__":
    # First check if the server is running
    try:
        health = requests.get("http://localhost:8000/api/health")
        print(f"Server health: {health.json()}")
        
        # If health check passes, run tests
        test_support()
    except Exception as e:
        print(f"Error connecting to server: {str(e)}")
        print("Make sure the server is running with 'python main_direct.py'") 