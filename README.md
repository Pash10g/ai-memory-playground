# AI Memory Service - Quick Start Guide

## 🚀 Getting Started with AI Memory Service & Streamlit Playground

![playgorund](./images/ai-memory.png)

This guide will help you quickly set up and start playing with the AI Memory Service using the interactive Streamlit interface.

## Memory Service 

This project utelise a form of the [ai-memory](https://www.mongodb.com/company/blog/technical/build-ai-memory-systems-mongodb-atlas-aws-claude) project.


## 📋 Prerequisites

- Python 3.10+
- MongoDB Atlas account (free tier works)
- OpenAI API key
- AWS account with Bedrock access (optional, for advanced features)

## ⚙️ Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/mongodb-partners/ai-memory.git
cd ai-memory
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root with the following configuration:

```env
# MongoDB Atlas Configuration
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority&appName=YourCluster

# OpenAI Configuration (Required for Streamlit app)
OPENAI_API_KEY=sk-proj-your-openai-api-key-here

# AWS Configuration (Optional - for advanced memory features)
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_REGION=us-east-1
LLM_MODEL_ID=us.anthropic.claude-3-7-sonnet-20250219-v1:0
EMBEDDING_MODEL_ID=amazon.titan-embed-text-v1

# Memory Service Configuration
SERVICE_HOST=0.0.0.0
SERVICE_PORT=8182
DEBUG=false

# Memory System Parameters (Optional - has defaults)
MAX_DEPTH=5
SIMILARITY_THRESHOLD=0.7
DECAY_FACTOR=0.99
REINFORCEMENT_FACTOR=1.1
```

### 4. MongoDB Atlas Setup

1. **Create a MongoDB Atlas Account**: Go to [mongodb.com/atlas](https://mongodb.com/atlas)
2. **Create a Cluster**: Use the free M0 tier
3. **Enable Vector Search**: In your cluster, go to "Search" → "Create Search Index"
4. **Create Database**: Name it `ai_memory`
5. **Get Connection String**: Replace `username`, `password`, and cluster details in `MONGODB_URI`

### 5. OpenAI API Key Setup

1. **Get API Key**: Go to [platform.openai.com](https://platform.openai.com/api-keys)
2. **Create New Key**: Click "Create new secret key"
3. **Copy Key**: Add it to your `.env` file as `OPENAI_API_KEY`

## 🏃‍♂️ Running the Services

### Start the Memory Service (Backend)
```bash
python main.py
```
The service will start on `http://localhost:8182`

### Start the Streamlit App (Frontend)
```bash
streamlit run streamlit_app_fixed.py
```
The app will open in your browser at `http://localhost:8501`

## 🎮 What You Can Play Around With

### 1. **System Prompt Templates** 📝
- **Helpful Assistant**: General-purpose AI
- **Business Advisor**: Strategic business consultant
- **Learning Tutor**: Educational assistant
- **Code Mentor**: Programming expert
- **Custom**: Write your own personality

**Try This**: Switch between templates and see how the AI's responses change!

### 2. **PydanticAI Tools Playground** 🛠️

#### Pre-built Sample Tools:
- **User Preferences**: `"Get my user preferences for user_001"`
- **Weather Lookup**: `"What's the weather in New York?"`
- **Investment Calculator**: `"Calculate returns for $1000 at 5% for 10 years"`

#### Create Custom Tools:
```json
{
  "tool_name": "task_manager",
  "input": {
    "user_id": "string",
    "action": "string"
  },
  "output": {
    "tasks": ["array of strings"],
    "status": "string"
  }
}
```

![tools](./images/custom_tools.png)

**Try This**: Create a tool for your specific use case and watch the AI use it!

### 3. **Memory System Experiments** 🧠

#### Generate Sample Data:
1. Click "🎯 Random Username" → Gets: `amazing_developer_123`
2. Click "💬 Random Conversation ID" → Gets: `project_planning_20250131_abc123`
3. Click "🚀 Generate Sample Conversations" → Creates realistic conversation history

#### Test Memory Retrieval:
After generating sample data, try these queries:
- `"What did we discuss about AI projects?"`
- `"Tell me about the code review conversation"`
- `"What MongoDB advice was given?"`
- `"Remind me about the tool development discussion"`

### 4. **Memory Retrieval Modes** 🔍

- **Automatic**: Always retrieves memories (best for ongoing conversations)
- **Query-based**: Only when you ask for context (keywords: "remember", "what did we")
- **Disabled**: No memory retrieval (for testing current context only)

**Try This**: Switch modes and see how the AI's awareness changes!

### 5. **Configuration Management** 💾

#### Save Your Perfect Setup:
1. Configure your ideal system prompt
2. Set up your favorite tools
3. Click "💾 Save Config"
4. Download the JSON file

#### Load Previous Configurations:
1. Click "📤 Load Config"
2. Upload your saved JSON
3. Everything restores automatically!

### 6. **Context Window Debugging** 🔍

![context_window](./images/context-memory.png)

At the bottom of the app, explore:
- **Memory Context**: See what memories the AI has access to
- **System Prompt**: View the current AI instructions
- **Tool Status**: Check which tools are available
- **Tool Execution History**: See detailed tool calls with inputs/outputs
- **Configuration**: Review all current settings

## 🎯 Fun Experiments to Try

### Experiment 1: Memory Building
1. Generate sample conversations
2. Start a new conversation thread
3. Ask: `"What do you know about my preferences?"`
4. Watch it pull relevant memories!

### Experiment 2: Tool Chaining
1. Enable PydanticAI tools
2. Load sample tools
3. Ask: `"Check my user preferences and then get the weather for my location"`
4. Watch multiple tools work together!

### Experiment 3: Personality Testing
1. Switch to "Business Advisor" template
2. Ask: `"Should I invest in this project?"`
3. Switch to "Learning Tutor" template
4. Ask the same question
5. Compare the different approaches!

### Experiment 4: Memory Persistence
1. Have a conversation about your interests
2. Start a new conversation thread (different ID)
3. Ask: `"What do you remember about me?"`
4. Test cross-conversation memory!

## 🔧 Troubleshooting

### Service Not Starting?
- Check MongoDB URI is correct
- Verify AWS credentials (if using Bedrock)
- Check port 8182 isn't in use

### Streamlit Errors?
- Ensure OpenAI API key is valid
- Check memory service is running on localhost:8182
- Verify all Python dependencies are installed

### Memory Not Working?
- Check MongoDB Atlas connection
- Verify vector search indexes are created
- Try the health check button in the app

## 🌟 Pro Tips

1. **Start Simple**: Begin with OpenAI mode, then add PydanticAI tools
2. **Generate Sample Data**: Use sample conversations to test memory features
3. **Save Configurations**: Save different setups for different use cases
4. **Monitor Context Window**: Use it to understand what the AI "sees"
5. **Experiment with Prompts**: Custom system prompts can create specialized agents

## 📚 Next Steps

Once you're comfortable with the basics:
- Create custom tools for your business logic
- Experiment with different memory retrieval strategies
- Build complex multi-step workflows using tool chaining
- Integrate with your own data sources

Enjoy exploring the AI Memory Service! 🚀