# Silicon-Bot: AI Customer Support Backend

A powerful AI-powered customer support backend using Gemini API, designed for easy integration with websites and deployment on Vercel.

## Features

- Multiple AI support agents for different business needs
- Seamless integration with your website
- Ready for Vercel deployment
- Configurable agent personalities and knowledge bases
- Simple REST API for easy integration
- Automatic fallback to contact information when the bot can't answer
- Stays on-topic by politely declining to answer irrelevant questions
- Provides business contact information for further assistance

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set your Gemini API key:
   - Create a copy of `env.example` named `.env`
   - Add your Gemini API key to the `.env` file

## Local Development

Run the server locally:

```
python main.py
```

The server will start at http://localhost:8000

## API Endpoints

- `GET /api/health` - Health check endpoint
- `GET /api/agents` - Get available support agent configurations
- `POST /api/support` - Submit a support query and get an AI response

## Contact Information

The bot will automatically provide the following contact information when it can't answer a query:

- Email: siliconsynapse8@gmail.com
- 24/7 Support: +91 7668055685
- Business Hours: Mon-Fri, 9am-6pm

## Deploying to Vercel

1. Push this repository to GitHub
2. Connect your Vercel account to your GitHub repository
3. Add your `GEMINI_API_KEY` as an environment variable in Vercel project settings
4. Deploy

## Frontend Integration

```javascript
// Example fetch request
const response = await fetch('https://your-vercel-url.vercel.app/api/support', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    conversation: {
      messages: [
        { role: "user", content: "I need help with my order" }
      ]
    },
    agent_config: {
      name: "Support Agent",
      instructions: "Be helpful and friendly."
    }
  })
});

const data = await response.json();
console.log(data.response.content);
```
