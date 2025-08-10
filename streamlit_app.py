"""
AI Memory Service - Streamlit Interface

A user-friendly interface for interacting with the AI Memory Service.
Allows users to chat with an OpenAI agent and retrieve memories for specific threads.
"""

import streamlit as st
import httpx
import json
import datetime
import os
import asyncio
import logging
import random
import string
import uuid
import io
from typing import Dict, Any, Optional, List
from openai import OpenAI
from pydantic import BaseModel, Field, create_model
from pydantic_ai import Agent, RunContext
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()  # Load environment variables from .env file

# Configure the page
st.set_page_config(
    page_title="AI Memory Service",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
DEFAULT_API_URL = "http://localhost:8182"

# System Prompt Templates
SYSTEM_PROMPT_TEMPLATES = {
    "helpful_assistant": {
        "name": "Helpful Assistant",
        "description": "A general-purpose AI assistant that's helpful, harmless, and honest",
        "prompt": """You are a helpful, harmless, and honest AI assistant. You aim to be helpful by providing accurate, relevant, and useful information while being respectful and considerate. You should:

- Provide clear and accurate answers to questions
- Be helpful and supportive in your responses
- Admit when you don't know something rather than guessing
- Be concise but thorough in your explanations
- Maintain a friendly and professional tone"""
    },
    "business_advisor": {
        "name": "Business Advisor", 
        "description": "An expert business consultant providing strategic advice and insights",
        "prompt": """You are an experienced business advisor and strategic consultant with expertise across various industries. You help clients with:

- Strategic planning and business development
- Market analysis and competitive intelligence
- Financial planning and investment decisions
- Operations optimization and process improvement
- Risk assessment and mitigation strategies
- Leadership and organizational development

You provide practical, actionable advice backed by business best practices and data-driven insights. You ask clarifying questions when needed to provide the most relevant guidance."""
    },
    "learning_tutor": {
        "name": "Learning Tutor",
        "description": "An educational assistant focused on teaching and learning support",
        "prompt": """You are a patient and knowledgeable educational tutor. Your role is to help students learn and understand concepts across various subjects. You should:

- Break down complex topics into understandable steps
- Use examples and analogies to clarify difficult concepts
- Encourage active learning through questions and practice
- Adapt your teaching style to different learning preferences
- Provide constructive feedback and positive reinforcement
- Foster curiosity and critical thinking skills

Always aim to help students discover answers themselves rather than simply providing solutions."""
    },
    "code_mentor": {
        "name": "Code Mentor",
        "description": "A programming expert providing coding guidance and best practices",
        "prompt": """You are an experienced software engineering mentor with expertise in multiple programming languages and technologies. You help developers by:

- Explaining coding concepts and best practices
- Reviewing code and suggesting improvements
- Helping debug issues and solve programming problems
- Recommending appropriate tools, libraries, and frameworks
- Teaching software architecture and design patterns
- Promoting clean, maintainable, and efficient code

You provide clear explanations, code examples, and step-by-step guidance while encouraging good programming habits and continuous learning."""
    },
    "custom": {
        "name": "Custom",
        "description": "Define your own custom system prompt",
        "prompt": ""
    }
}

# ============================================================================
# Generic Tool System for PydanticAI
# ============================================================================

class ToolDefinition(BaseModel):
    """Definition of a generic tool with input/output schemas"""
    name: str = Field(..., description="Name of the tool")
    description: str = Field(..., description="Description of what the tool does")
    input_schema: Dict[str, Any] = Field(..., description="JSON schema for input parameters")
    output_schema: Dict[str, Any] = Field(..., description="JSON schema for expected output")
    mock_response: Optional[Dict[str, Any]] = Field(None, description="Mock response for testing")

class GenericToolSystem:
    """System for creating and managing generic tools for PydanticAI agents"""
    
    def __init__(self, openai_api_key: str, model: str = "gpt-4o"):
        self.openai_api_key = openai_api_key
        self.model = model
        self.tools: Dict[str, ToolDefinition] = {}
        self.agent = None
        
    def add_tool(self, tool_def: ToolDefinition):
        """Add a tool definition to the system"""
        logger.info(f"Adding tool: {tool_def.name}")
        logger.debug(f"Tool definition: {tool_def}")
        self.tools[tool_def.name] = tool_def
        
    def create_pydantic_models(self, tool_def: ToolDefinition):
        """Create Pydantic models from JSON schemas"""
        logger.info(f"Creating Pydantic models for tool: {tool_def.name}")
        
        # Create input model
        input_fields = {}
        for field_name, field_info in tool_def.input_schema.items():
            logger.debug(f"Processing input field: {field_name} = {field_info}")
            field_type = str  # Default to string
            field_default = ...  # Required by default
            
            if isinstance(field_info, dict):
                if field_info.get("type") == "integer":
                    field_type = int
                elif field_info.get("type") == "number":
                    field_type = float
                elif field_info.get("type") == "boolean":
                    field_type = bool
                elif field_info.get("type") == "array":
                    field_type = List[str]  # Simplified
                
                if "default" in field_info:
                    field_default = field_info["default"]
                elif not field_info.get("required", True):
                    field_default = None
                    field_type = Optional[field_type]
                    
                description = field_info.get("description", "")
            else:
                # Simple type definition
                description = f"Parameter {field_name}"
                
            input_fields[field_name] = (field_type, Field(field_default, description=description))
        
        InputModel = create_model(f"{tool_def.name}Input", **input_fields)
        logger.debug(f"Created InputModel for {tool_def.name}: {InputModel}")
        
        # Create output model
        output_fields = {}
        for field_name, field_info in tool_def.output_schema.items():
            logger.debug(f"Processing output field: {field_name} = {field_info}")
            field_type = str  # Default to string
            
            if isinstance(field_info, dict):
                if field_info.get("type") == "integer":
                    field_type = int
                elif field_info.get("type") == "number":
                    field_type = float
                elif field_info.get("type") == "boolean":
                    field_type = bool
                elif field_info.get("type") == "array":
                    field_type = List[str]  # Simplified
                    
                description = field_info.get("description", "")
            else:
                description = f"Output {field_name}"
                
            output_fields[field_name] = (field_type, Field(..., description=description))
        
        OutputModel = create_model(f"{tool_def.name}Output", **output_fields)
        logger.debug(f"Created OutputModel for {tool_def.name}: {OutputModel}")
        
        return InputModel, OutputModel
        
    def create_agent(self, system_prompt: str = None) -> Agent:
        """Create a PydanticAI agent with all registered tools"""
        logger.info(f"Creating PydanticAI agent with {len(self.tools)} tools")
        logger.debug(f"Available tools: {list(self.tools.keys())}")
        
        if not system_prompt:
            system_prompt = (
                "You are a helpful AI assistant with access to various tools. "
                "Use the available tools to help answer questions and perform tasks. "
                "When using tools, generate realistic outputs based on the input parameters."
            )
            
        logger.debug(f"System prompt: {system_prompt[:200]}...")
        
        # Create agent with dynamic output type (we'll use dict for flexibility)
        try:
            self.agent = Agent(
                f'openai:{self.model}',
                deps_type=Dict[str, Any],  # Dependencies will contain tool definitions and context
                system_prompt=system_prompt,
            )
            logger.info("PydanticAI agent created successfully")
        except Exception as e:
            logger.error(f"Failed to create PydanticAI agent: {e}")
            raise
        
        # Register all tools dynamically
        for tool_name, tool_def in self.tools.items():
            logger.info(f"Registering tool: {tool_name}")
            try:
                self._register_tool(tool_def)
                logger.info(f"Successfully registered tool: {tool_name}")
            except Exception as e:
                logger.error(f"Failed to register tool {tool_name}: {e}")
                raise
            
        return self.agent
    
    def _register_tool(self, tool_def: ToolDefinition):
        """Register a single tool with the agent"""
        logger.info(f"Starting tool registration for: {tool_def.name}")
        
        try:
            InputModel, OutputModel = self.create_pydantic_models(tool_def)
            logger.debug(f"Created models for {tool_def.name}")
        except Exception as e:
            logger.error(f"Failed to create models for {tool_def.name}: {e}")
            raise
        
        # Create the tool function dynamically
        async def generic_tool_func(ctx: RunContext[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
            """Generic tool function that uses OpenAI to generate responses"""
            execution_start = datetime.datetime.now()
            logger.info(f"Executing tool: {tool_def.name} with args: {kwargs}")
            
            # If mock response is provided, use it
            if tool_def.mock_response:
                logger.info(f"Using mock response for {tool_def.name}")
                execution_end = datetime.datetime.now()
                execution_time = (execution_end - execution_start).total_seconds()
                
                # Track execution in session state
                if 'tool_execution_history' in st.session_state:
                    execution_record = {
                    'tool_name': tool_def.name,
                    'timestamp': execution_start.isoformat(),
                    'execution_time_ms': execution_time * 1000,
                    'input_args': kwargs,
                    'output': tool_def.mock_response,
                    'status': 'success',
                    'method': 'mock_response',
                    'error': None
                    }
                    st.session_state.tool_execution_history.append(execution_record)
                
                return tool_def.mock_response
            
            # Otherwise, use OpenAI to generate a realistic response
            try:
                logger.info(f"Generating response for {tool_def.name} using OpenAI")
                
                # Get the user's original query from context or session state
                user_query = ""
                if hasattr(ctx, 'deps') and ctx.deps:
                    # Try to get from context dependencies
                    user_query = ctx.deps.get('user_query', '')
                
                # Fallback: get the last user message from session state
                if not user_query and 'chat_messages' in st.session_state:
                    recent_user_messages = [msg for msg in st.session_state.chat_messages if msg['role'] == 'user']
                    if recent_user_messages:
                        user_query = recent_user_messages[-1]['content']
                
                # Create a prompt for generating the output
                input_description = json.dumps(kwargs, indent=2)
                output_schema_description = json.dumps(tool_def.output_schema, indent=2)
                
                prompt = f"""
        You are a tool called "{tool_def.name}" that {tool_def.description}.

        User's original question/request: "{user_query}"

        Tool input received:
        {input_description}

        Expected output schema:
        {output_schema_description}

        Generate a realistic JSON response that matches the output schema exactly. 
        The response should be relevant to both the user's original question and the tool input parameters.
        Return only valid JSON without any additional text or markdown formatting.
        Make the response realistic, helpful, and contextually appropriate to what the user was asking about.
        """
                
                from openai import OpenAI
                client = OpenAI(api_key=self.openai_api_key)
                
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a tool that generates structured JSON responses."},
                        {"role": "user", "content": prompt}
                    ]
                )
                
                # Parse the JSON response
                response_text = response.choices[0].message.content.strip()
                logger.debug(f"OpenAI response for {tool_def.name}: {response_text}")
                
                # Try to extract JSON if it's wrapped in markdown
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0].strip()
                
                try:
                    result = json.loads(response_text)
                    logger.info(f"Successfully parsed JSON response for {tool_def.name}")
                    execution_end = datetime.datetime.now()
                    execution_time = (execution_end - execution_start).total_seconds()
                    
                    # Track successful execution
                    if 'tool_execution_history' in st.session_state:
                        execution_record = {
                            'tool_name': tool_def.name,
                            'timestamp': execution_start.isoformat(),
                            'execution_time_ms': execution_time * 1000,
                            'input_args': kwargs,
                            'output': result,
                            'status': 'success',
                            'method': 'openai_generated',
                            'error': None
                        }
                        st.session_state.tool_execution_history.append(execution_record)
                    
                    return result
                except json.JSONDecodeError as json_error:
                    logger.warning(f"Failed to parse JSON for {tool_def.name}: {json_error}")
                    # Fallback to a simple response
                    fallback_response = {"result": response_text}
                    execution_end = datetime.datetime.now()
                    execution_time = (execution_end - execution_start).total_seconds()
                    
                    # Track fallback execution
                    if 'tool_execution_history' in st.session_state:
                        execution_record = {
                            'tool_name': tool_def.name,
                            'timestamp': execution_start.isoformat(),
                            'execution_time_ms': execution_time * 1000,
                            'input_args': kwargs,
                            'output': fallback_response,
                            'status': 'fallback',
                            'method': 'json_parse_fallback',
                            'error': str(json_error)
                        }
                        st.session_state.tool_execution_history.append(execution_record)
                    
                    return fallback_response
                    
            except Exception as e:
                logger.error(f"Tool execution failed for {tool_def.name}: {e}")
                execution_end = datetime.datetime.now()
                execution_time = (execution_end - execution_start).total_seconds()
                
                # Return an error response that matches the schema if possible
                error_response = {"error": f"Tool execution failed: {str(e)}"}
                
                # Track failed execution
                if 'tool_execution_history' in st.session_state:
                    execution_record = {
                        'tool_name': tool_def.name,
                        'timestamp': execution_start.isoformat(),
                        'execution_time_ms': execution_time * 1000,
                        'input_args': kwargs,
                        'output': error_response,
                        'status': 'error',
                        'method': 'error_fallback',
                        'error': str(e)
                    }
                    st.session_state.tool_execution_history.append(execution_record)
                
                return error_response
        
        # Set the function name and description
        generic_tool_func.__name__ = tool_def.name
        generic_tool_func.__doc__ = tool_def.description
        
        logger.debug(f"Created tool function {tool_def.name}, registering with agent")
        
        # Use the decorator pattern to register the tool
        try:
            decorated_func = self.agent.tool(generic_tool_func)
            logger.info(f"Successfully registered tool {tool_def.name} with agent")
            return decorated_func
        except Exception as e:
            logger.error(f"Failed to register tool {tool_def.name} with agent: {e}")
            logger.error(f"Agent type: {type(self.agent)}")
            logger.error(f"Agent methods: {dir(self.agent)}")
            raise

# ============================================================================
# End Generic Tool System
# ============================================================================

def display_tool_calls(tool_calls, message_container):
    """Display tool calls in an expandable section under the AI response"""
    if tool_calls and len(tool_calls) > 0:
        with message_container:
            with st.expander(f"🔧 Tool Calls ({len(tool_calls)})", expanded=False):
                for i, call in enumerate(tool_calls, 1):
                    st.write(f"**Tool {i}: {call.tool_name}**")
                    if hasattr(call, 'args') and call.args:
                        st.json(call.args)
                    if hasattr(call, 'result') and call.result:
                        st.write("**Result:**")
                        if isinstance(call.result, dict):
                            st.json(call.result)
                        else:
                            st.write(call.result)
                    if i < len(tool_calls):
                        st.divider()

def display_debug_logs():
    """Display debug logs in an expandable section"""
    if st.sidebar.checkbox("Show Debug Logs", value=False):
        with st.sidebar.expander("🐛 Debug Logs", expanded=False):
            # Capture recent log entries
            import logging
            import io
            import sys
            
            # Create a string buffer to capture logs
            log_capture = io.StringIO()
            handler = logging.StreamHandler(log_capture)
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            
            # Add handler to root logger
            logging.getLogger().addHandler(handler)
            
            # Display captured logs
            log_contents = log_capture.getvalue()
            if log_contents:
                st.text_area("Recent Logs:", log_contents, height=200)
            else:
                st.write("No logs captured yet.")
                
            # Clean up
            logging.getLogger().removeHandler(handler)

class OpenAIClient:
    """Client for interacting with OpenAI API"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    async def chat_completion(self, messages: List[Dict[str, str]], memory_context: str = "") -> str:
        """Generate a chat completion with optional memory context"""
        try:
            # Add memory context to the system message if available
            enhanced_messages = messages.copy()
            if memory_context and enhanced_messages:
                if enhanced_messages[0]["role"] == "system":
                    enhanced_messages[0]["content"] += f"\n\nRelevant memory context:\n{memory_context}"
                else:
                    enhanced_messages.insert(0, {
                        "role": "system", 
                        "content": f"You are a helpful AI assistant. Use this memory context to provide better responses:\n{memory_context}"
                    })

            print("Enhanced messages for OpenAI:", json.dumps(enhanced_messages, indent=2))
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=enhanced_messages,
            )
            
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}") from e

class MemoryServiceClient:
    """Client for interacting with the AI Memory Service API"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        
    async def add_message(self, user_id: str, conversation_id: str, message_type: str, text: str) -> Dict[str, Any]:
        """Add a message to the conversation history"""
        url = f"{self.base_url}/conversation/"
        
        message_data = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "type": message_type,
            "text": text,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=message_data)
                response.raise_for_status()
                return response.json()
        except httpx.ConnectError as e:
            raise Exception(f"Could not connect to AI Memory Service at {url}. Is the service running?") from e
        except httpx.HTTPStatusError as e:
            raise Exception(f"HTTP error {e.response.status_code}: {e.response.text}") from e
        except Exception as e:
            raise Exception(f"Unexpected error when adding message: {str(e)}") from e
    
    async def retrieve_memory(self, user_id: str, text: str) -> Dict[str, Any]:
        """Retrieve memory items, context, and similar memories"""
        url = f"{self.base_url}/retrieve_memory/"
        params = {"user_id": user_id, "text": text}
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                return response.json()
        except httpx.ConnectError as e:
            raise Exception(f"Could not connect to AI Memory Service at {url}. Is the service running?") from e
        except httpx.HTTPStatusError as e:
            raise Exception(f"HTTP error {e.response.status_code}: {e.response.text}") from e
        except Exception as e:
            raise Exception(f"Unexpected error when retrieving memory: {str(e)}") from e
    
    async def health_check(self) -> Dict[str, Any]:
        """Check if the service is healthy"""
        url = f"{self.base_url}/health"
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                return response.json()
        except httpx.ConnectError as e:
            raise Exception(f"Could not connect to AI Memory Service at {url}. Is the service running?") from e
        except httpx.HTTPStatusError as e:
            raise Exception(f"HTTP error {e.response.status_code}: {e.response.text}") from e
        except Exception as e:
            raise Exception(f"Unexpected error during health check: {str(e)}") from e

def init_session_state():
    """Initialize session state variables"""
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    if 'current_user_id' not in st.session_state:
        st.session_state.current_user_id = "user_001"
    if 'current_conversation_id' not in st.session_state:
        st.session_state.current_conversation_id = "conv_001"
    if 'api_url' not in st.session_state:
        st.session_state.api_url = DEFAULT_API_URL
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY", "")
    if 'openai_model' not in st.session_state:
        st.session_state.openai_model = "gpt-4o-mini"
    if 'use_memory_context' not in st.session_state:
        st.session_state.use_memory_context = True
    if 'memory_retrieval_mode' not in st.session_state:
        st.session_state.memory_retrieval_mode = "automatic"
    if 'max_memories' not in st.session_state:
        st.session_state.max_memories = 3
    if 'include_conversation_summary' not in st.session_state:
        st.session_state.include_conversation_summary = True
    if 'memory_relevance_threshold' not in st.session_state:
        st.session_state.memory_relevance_threshold = 0.5
    
    # Tool system session state
    if 'use_pydantic_agent' not in st.session_state:
        st.session_state.use_pydantic_agent = False
    if 'custom_tools' not in st.session_state:
        st.session_state.custom_tools = []
    if 'tool_system' not in st.session_state:
        st.session_state.tool_system = None
    
    # System prompt configuration
    if 'system_prompt_template' not in st.session_state:
        st.session_state.system_prompt_template = "helpful_assistant"
    if 'custom_system_prompt' not in st.session_state:
        st.session_state.custom_system_prompt = ""
    
    # Context tracking
    if 'current_memory_context' not in st.session_state:
        st.session_state.current_memory_context = ""
    if 'current_system_prompt' not in st.session_state:
        st.session_state.current_system_prompt = ""
    if 'last_tool_calls' not in st.session_state:
        st.session_state.last_tool_calls = []
    if 'tool_execution_history' not in st.session_state:
        st.session_state.tool_execution_history = []

def get_serializable_state() -> Dict[str, Any]:
    """Get a serializable version of the session state for saving configurations"""
    # Define which keys to include in the configuration dump
    config_keys = [
        'openai_model', 'use_memory_context', 'memory_retrieval_mode',
        'max_memories', 'include_conversation_summary', 'memory_relevance_threshold',
        'use_pydantic_agent', 'system_prompt_template', 'custom_system_prompt',
        'api_url', 'current_user_id', 'current_conversation_id'
    ]
    
    # Also include custom tools (but serialize them as dictionaries)
    serializable_state = {}
    
    for key in config_keys:
        if key in st.session_state:
            serializable_state[key] = st.session_state[key]
    
    # Handle custom tools separately since they're ToolDefinition objects
    if 'custom_tools' in st.session_state and st.session_state.custom_tools:
        serializable_state['custom_tools'] = []
        for tool in st.session_state.custom_tools:
            if hasattr(tool, 'dict'):
                # Pydantic model
                serializable_state['custom_tools'].append(tool.dict())
            else:
                # Convert to dict manually
                serializable_state['custom_tools'].append({
                    'name': tool.name,
                    'description': tool.description,
                    'input_schema': tool.input_schema,
                    'output_schema': tool.output_schema,
                    'mock_response': tool.mock_response
                })
    
    # Add metadata
    serializable_state['_metadata'] = {
        'export_timestamp': datetime.datetime.now().isoformat(),
        'version': '1.0',
        'tool_count': len(serializable_state.get('custom_tools', [])),
        'app_name': 'AI Memory Service'
    }
    
    return serializable_state

def load_state_from_dict(state_dict: Dict[str, Any]) -> bool:
    """Load session state from a dictionary. Returns True if successful."""
    try:
        # Load basic configuration values
        config_keys = [
            'openai_model', 'use_memory_context', 'memory_retrieval_mode',
            'max_memories', 'include_conversation_summary', 'memory_relevance_threshold',
            'use_pydantic_agent', 'system_prompt_template', 'custom_system_prompt',
            'api_url', 'current_user_id', 'current_conversation_id'
        ]
        
        for key in config_keys:
            if key in state_dict:
                st.session_state[key] = state_dict[key]
        
        # Handle custom tools
        if 'custom_tools' in state_dict:
            st.session_state.custom_tools = []
            for tool_dict in state_dict['custom_tools']:
                try:
                    tool = ToolDefinition(**tool_dict)
                    st.session_state.custom_tools.append(tool)
                except Exception as e:
                    logger.warning(f"Failed to load tool {tool_dict.get('name', 'unknown')}: {e}")
                    continue
        
        # Reset tool system to force recreation with new tools
        st.session_state.tool_system = None
        
        # Clear runtime state that shouldn't be restored
        st.session_state.chat_messages = []
        st.session_state.conversation_history = []
        st.session_state.current_memory_context = ""
        st.session_state.last_tool_calls = []
        st.session_state.tool_execution_history = []
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to load state: {e}")
        return False

def format_timestamp(timestamp_str: str) -> str:
    """Format timestamp for display"""
    try:
        dt = datetime.datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return timestamp_str

def display_memory_results(results: Dict[str, Any]):
    """Display memory retrieval results in a structured format"""
    
    # Related Conversation
    st.subheader("📝 Related Conversation")
    if results.get("related_conversation") == "No conversation found":
        st.info("No related conversation found.")
    else:
        conversation = results.get("related_conversation", [])
        if isinstance(conversation, list):
            for msg in conversation:
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        msg_type = msg.get("type", "unknown")
                        icon = "🤖" if msg_type == "ai" else "👤"
                        st.write(f"{icon} **{msg_type.title()}**: {msg.get('text', '')}")
                    with col2:
                        if msg.get("timestamp"):
                            st.caption(format_timestamp(msg["timestamp"]))
                    st.divider()
    
    # Conversation Summary
    st.subheader("📋 Conversation Summary")
    summary = results.get("conversation_summary", "No summary found")
    if summary == "No summary found":
        st.info("No conversation summary available.")
    else:
        st.info(summary)
    
    # Similar Memories
    st.subheader("🧠 Similar Memories")
    similar_memories = results.get("similar_memories", "No similar memories found")
    if similar_memories == "No similar memories found":
        st.info("No similar memories found.")
    else:
        if isinstance(similar_memories, list):
            for i, memory in enumerate(similar_memories, 1):
                with st.expander(f"Memory {i} (Similarity: {memory.get('similarity', 'N/A'):.3f})"):
                    st.write(f"**Content**: {memory.get('content', 'N/A')}")
                    st.write(f"**Summary**: {memory.get('summary', 'N/A')}")
                    st.write(f"**Importance**: {memory.get('importance', 'N/A'):.3f}")

def should_retrieve_memories(query: str, retrieval_mode: str) -> bool:
    """Determine if memories should be retrieved based on the query and mode"""
    if retrieval_mode == "disabled":
        return False
    elif retrieval_mode == "automatic":
        return True
    elif retrieval_mode == "query_based":
        # Keywords that suggest the user wants contextual information
        context_keywords = [
            "remember", "recall", "what did", "mentioned", "said", "discussed",
            "talked about", "previous", "earlier", "before", "last time",
            "history", "past", "context", "background", "summary", "recap"
        ]
        
        # Questions that often need context
        question_indicators = ["what", "when", "where", "who", "how", "why", "did"]
        
        query_lower = query.lower()
        
        # Check for direct context keywords
        if any(keyword in query_lower for keyword in context_keywords):
            return True
        
        # Check for questions that might need context
        if any(indicator in query_lower for indicator in question_indicators):
            return True
        
        # Check for references to specific topics that might need background
        if "?" in query or query.endswith(".") and len(query.split()) > 5:
            return True
            
        return False
    
    return False

def format_memory_context(memory_results: Dict[str, Any], max_memories: int, 
                         include_summary: bool, relevance_threshold: float) -> str:
    """Format memory results into context string with filtering"""
    memory_context = ""
    
    similar_memories = memory_results.get("similar_memories", [])
    if isinstance(similar_memories, list) and similar_memories:
        # Filter memories by relevance threshold
        relevant_memories = [
            memory for memory in similar_memories 
            if memory.get('similarity', 0) >= relevance_threshold
        ]
        
        # Limit to max_memories
        relevant_memories = relevant_memories[:max_memories]
        
        if relevant_memories:
            memory_context += "## Relevant Previous Conversations\n"
            for i, memory in enumerate(relevant_memories, 1):
                similarity = memory.get('similarity', 0)
                content = memory.get('summary', memory.get('content', ''))
                importance = memory.get('importance', 0)
                
                memory_context += f"{i}. [{similarity:.2f} similarity, {importance:.2f} importance] {content}\n"
            memory_context += "\n"
    
    # Add conversation summary if enabled and available
    if include_summary and memory_results.get("conversation_summary") != "No summary found":
        memory_context += f"## Previous Conversation Summary\n{memory_results['conversation_summary']}\n\n"
    
    # Add related conversation if available and different from summary
    related_conv = memory_results.get("related_conversation", [])
    if isinstance(related_conv, list) and related_conv and not include_summary:
        memory_context += "## Related Conversation Context\n"
        for msg in related_conv[-3:]:  # Last 3 messages for context
            msg_type = msg.get("type", "unknown")
            text = msg.get("text", "")
            memory_context += f"[{msg_type}]: {text}\n"
        memory_context += "\n"
    
    return memory_context.strip()

def run_async(coro):
    """Helper function to run async functions in Streamlit"""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)
    except RuntimeError:
        # If no event loop exists, create a new one
        return asyncio.run(coro)

def create_sample_tools() -> List[ToolDefinition]:
    """Create sample tool definitions for demonstration"""
    logger.info("Creating sample tools")
    
    try:
        tools = [
            ToolDefinition(
                name="user_preference",
                description="Get user preferences and risk score",
                input_schema={
                    "user_id": {"type": "string", "description": "User identifier", "required": True}
                },
                output_schema={
                    "user_risk_score": {"type": "array", "items": {"type": "object", "properties": {"score": {"type": "integer"}}}}
                },
                mock_response={"user_risk_score": [{"score": 92}]}
            ),
            ToolDefinition(
                name="weather_lookup",
                description="Get current weather for a location",
                input_schema={
                    "location": {"type": "string", "description": "City name or coordinates", "required": True},
                    "units": {"type": "string", "description": "Temperature units", "default": "celsius"}
                },
                output_schema={
                    "temperature": {"type": "number", "description": "Current temperature"},
                    "condition": {"type": "string", "description": "Weather condition"},
                    "humidity": {"type": "integer", "description": "Humidity percentage"}
                }
            ),
            ToolDefinition(
                name="calculate_investment",
                description="Calculate investment returns",
                input_schema={
                    "principal": {"type": "number", "description": "Initial investment amount", "required": True},
                    "rate": {"type": "number", "description": "Annual interest rate (as decimal)", "required": True},
                    "years": {"type": "integer", "description": "Investment period in years", "required": True}
                },
                output_schema={
                    "final_amount": {"type": "number", "description": "Final investment value"},
                    "total_return": {"type": "number", "description": "Total return amount"},
                    "return_percentage": {"type": "number", "description": "Return as percentage"}
                }
            )
        ]
        
        logger.info(f"Successfully created {len(tools)} sample tools")
        for tool in tools:
            logger.debug(f"Created tool: {tool.name}")
        
        return tools
        
    except Exception as e:
        logger.error(f"Failed to create sample tools: {e}")
        raise

def parse_tool_schema(schema_text: str) -> Dict[str, Any]:
    """Parse tool schema from JSON text"""
    try:
        return json.loads(schema_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON schema: {str(e)}")

def generate_random_username() -> str:
    """Generate a random username"""
    adjectives = [
        "amazing", "brilliant", "creative", "dynamic", "energetic", "fantastic", 
        "genius", "happy", "innovative", "joyful", "kind", "lovely", "magical",
        "nice", "optimistic", "perfect", "quiet", "radiant", "stellar", "thoughtful"
    ]
    
    nouns = [
        "developer", "coder", "designer", "artist", "engineer", "creator", 
        "builder", "maker", "thinker", "explorer", "innovator", "dreamer",
        "writer", "planner", "solver", "helper", "learner", "teacher"
    ]
    
    adjective = random.choice(adjectives)
    noun = random.choice(nouns)
    number = random.randint(1, 999)
    
    return f"{adjective}_{noun}_{number}"

def generate_random_conversation_id() -> str:
    """Generate a random conversation ID"""
    topics = [
        "project_planning", "code_review", "design_feedback", "team_meeting",
        "brainstorming", "product_demo", "bug_fixing", "feature_discussion",
        "user_research", "market_analysis", "technical_support", "training_session"
    ]
    
    topic = random.choice(topics)
    timestamp = datetime.datetime.now().strftime("%Y%m%d")
    random_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    
    return f"{topic}_{timestamp}_{random_id}"

def generate_sample_conversations() -> List[Dict[str, Any]]:
    """Generate sample conversation messages"""
    conversation_templates = [
        {
            "topic": "AI Project Planning",
            "messages": [
                {"role": "human", "text": "I'm thinking about starting an AI project using MongoDB and vector search. What should I consider?"},
                {"role": "ai", "text": "Great idea! For an AI project with MongoDB, consider: 1) Define your use case clearly, 2) Choose appropriate embedding models, 3) Design your schema for vector storage, 4) Plan for data preprocessing and indexing. What specific AI application are you building?"},
                {"role": "human", "text": "I want to build a document similarity search system for legal documents."},
                {"role": "ai", "text": "Perfect! For legal document similarity: Use sentence-transformers for embeddings, consider GDPR compliance, implement proper document chunking for large files, and use MongoDB Atlas Vector Search for efficient similarity queries. Start with a small dataset to test your pipeline."}
            ]
        },
        {
            "topic": "Code Review Discussion",
            "messages": [
                {"role": "human", "text": "Can you review this Python function for database connections?"},
                {"role": "ai", "text": "I'd be happy to review your code! Please share the function and I'll check for best practices, error handling, security considerations, and performance optimizations."},
                {"role": "human", "text": "Here's the function: def connect_db(): return pymongo.MongoClient('mongodb://localhost:27017')"},
                {"role": "ai", "text": "I see a few improvements: 1) Add error handling with try-catch, 2) Use connection pooling, 3) Don't hardcode connection strings - use environment variables, 4) Add connection timeout parameters, 5) Consider using context managers for proper resource cleanup."}
            ]
        },
        {
            "topic": "Memory System Design",
            "messages": [
                {"role": "human", "text": "How should I design a memory system for an AI assistant?"},
                {"role": "ai", "text": "For an AI memory system, consider: 1) Short-term vs long-term memory storage, 2) Relevance scoring for memory retrieval, 3) Memory consolidation strategies, 4) Privacy and data retention policies, 5) Efficient similarity search using vector databases."},
                {"role": "human", "text": "What about memory retrieval strategies?"},
                {"role": "ai", "text": "Key retrieval strategies: 1) Semantic similarity using embeddings, 2) Temporal relevance (recent vs old memories), 3) Importance scoring based on user feedback, 4) Context-aware retrieval based on current conversation, 5) Hybrid search combining multiple signals."}
            ]
        },
        {
            "topic": "Tool Development",
            "messages": [
                {"role": "human", "text": "I'm building a tool system for PydanticAI. Any advice on dynamic tool registration?"},
                {"role": "ai", "text": "For dynamic tool registration in PydanticAI: 1) Use decorator patterns for tool functions, 2) Implement proper schema validation, 3) Handle tool execution errors gracefully, 4) Consider tool versioning for updates, 5) Add comprehensive logging for debugging."},
                {"role": "human", "text": "How do I handle tool execution failures?"},
                {"role": "ai", "text": "Handle tool failures with: 1) Try-catch blocks around tool execution, 2) Fallback responses that match expected schemas, 3) Clear error messages for users, 4) Logging for debugging, 5) Circuit breaker patterns for external API calls."}
            ]
        }
    ]
    
    return conversation_templates

async def populate_sample_conversations(client: 'MemoryServiceClient', user_id: str, conversation_id: str):
    """Populate the memory service with sample conversations"""
    logger.info(f"Populating sample conversations for user {user_id}, conversation {conversation_id}")
    
    conversations = generate_sample_conversations()
    selected_conversation = random.choice(conversations)
    
    messages_added = 0
    for message in selected_conversation["messages"]:
        try:
            await client.add_message(
                user_id=user_id,
                conversation_id=conversation_id,
                message_type=message["role"],
                text=message["text"]
            )
            messages_added += 1
            logger.debug(f"Added message {messages_added}: {message['text'][:50]}...")
        except Exception as e:
            logger.error(f"Failed to add message: {e}")
            raise
    
    logger.info(f"Successfully added {messages_added} messages from topic: {selected_conversation['topic']}")
    return selected_conversation["topic"], messages_added

def display_context_window():
    """Display current context information at the bottom of the page"""
    st.header("🔍 Current Context Window")
    
    # Create tabs for different context types
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["💭 Memory Context", "📝 System Prompt", "🛠️ Tool Status", "📊 Tool History", "⚙️ Configuration"])
    
    with tab1:
        st.subheader("Memory Context")
        if st.session_state.get('current_memory_context'):
            st.text_area(
                "Current Memory Context:",
                value=st.session_state.current_memory_context,
                height=200,
                disabled=True,
                help="This is the memory context that was retrieved and provided to the AI"
            )
        else:
            st.info("No memory context available. Memory context will appear here when retrieved.")
    
    with tab2:
        st.subheader("Active System Prompt")
        
        # Get current system prompt
        if st.session_state.system_prompt_template == "custom":
            current_prompt = st.session_state.custom_system_prompt
        else:
            current_prompt = SYSTEM_PROMPT_TEMPLATES[st.session_state.system_prompt_template]["prompt"]
        
        if current_prompt:
            st.text_area(
                f"Current System Prompt ({SYSTEM_PROMPT_TEMPLATES[st.session_state.system_prompt_template]['name']}):",
                value=current_prompt,
                height=200,
                disabled=True,
                help="This is the system prompt that defines the AI's behavior and personality"
            )
        else:
            st.warning("No system prompt configured")
    
    with tab3:
        st.subheader("Tool System Status")
        
        if st.session_state.use_pydantic_agent and st.session_state.custom_tools:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Tools Available", len(st.session_state.custom_tools))
                st.metric("Tool System Status", "Active" if st.session_state.tool_system else "Inactive")
            
            with col2:
                if st.session_state.tool_system:
                    st.success("✅ Tool system initialized")
                    st.info(f"Agent Model: {st.session_state.tool_system.model}")
                else:
                    st.warning("⚠️ Tool system not initialized")
            
            # Show available tools
            st.write("**Available Tools:**")
            for tool in st.session_state.custom_tools:
                with st.expander(f"🔧 {tool.name}", expanded=False):
                    st.write(f"**Description:** {tool.description}")
                    st.write("**Input Schema:**")
                    st.json(tool.input_schema)
                    st.write("**Output Schema:**")
                    st.json(tool.output_schema)
                    if tool.mock_response:
                        st.write("**Mock Response:**")
                        st.json(tool.mock_response)
            
            # Show last tool calls
            if st.session_state.get('last_tool_calls'):
                st.write("**Recent Tool Calls:**")
                for i, call in enumerate(st.session_state.last_tool_calls[-3:], 1):  # Show last 3
                    st.write(f"**{i}. {call.get('tool_name', 'Unknown Tool')}**")
                    if call.get('args'):
                        st.json(call['args'])
        else:
            st.info("PydanticAI tools are not enabled or no tools are loaded")
    
    with tab4:
        st.subheader("Tool Execution History")
        
        # Display tool execution history
        if st.session_state.get('tool_execution_history'):
            history = st.session_state.tool_execution_history
            
            # Header with summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Executions", len(history))
            with col2:
                successful = len([h for h in history if h.get('status') == 'success'])
                st.metric("Successful", successful)
            with col3:
                failed = len([h for h in history if h.get('status') == 'error'])
                st.metric("Failed", failed)
            with col4:
                if history:
                    avg_time = sum(h.get('execution_time_ms', 0) for h in history) / len(history)
                    st.metric("Avg Time (ms)", f"{avg_time:.1f}")
                else:
                    st.metric("Avg Time (ms)", "0.0")
            
            st.divider()
            
            # History management controls
            col1, col2, col3 = st.columns(3)
            with col1:
                # Filter by tool name
                tool_names = ['All'] + list(set(h.get('tool_name', 'Unknown') for h in history))
                selected_tool = st.selectbox("Filter by Tool", tool_names, key="tool_filter")
            
            with col2:
                # Filter by status
                statuses = ['All', 'success', 'error', 'fallback']
                selected_status = st.selectbox("Filter by Status", statuses, key="status_filter")
            
            with col3:
                # Clear history button
                if st.button("🗑️ Clear History", help="Clear all tool execution history"):
                    st.session_state.tool_execution_history = []
                    st.rerun()
            
            # Apply filters
            filtered_history = history
            if selected_tool != 'All':
                filtered_history = [h for h in filtered_history if h.get('tool_name') == selected_tool]
            if selected_status != 'All':
                filtered_history = [h for h in filtered_history if h.get('status') == selected_status]
            
            # Display execution history (most recent first)
            if filtered_history:
                st.write(f"**Showing {len(filtered_history)} executions (most recent first):**")
                
                for i, execution in enumerate(reversed(filtered_history[-20:]), 1):  # Show last 20
                    with st.expander(
                        f"#{len(filtered_history) - i + 1}: {execution.get('tool_name', 'Unknown')} "
                        f"({execution.get('status', 'unknown')}) - "
                        f"{execution.get('execution_time_ms', 0):.1f}ms",
                        expanded=False
                    ):
                        # Execution details
                        detail_col1, detail_col2 = st.columns(2)
                        
                        with detail_col1:
                            st.write("**Execution Details:**")
                            st.write(f"• **Tool Name**: {execution.get('tool_name', 'Unknown')}")
                            st.write(f"• **Status**: {execution.get('status', 'unknown')}")
                            st.write(f"• **Method**: {execution.get('method', 'unknown')}")
                            st.write(f"• **Execution Time**: {execution.get('execution_time_ms', 0):.1f} ms")
                            
                            # Format timestamp for display
                            timestamp = execution.get('timestamp', '')
                            if timestamp:
                                try:
                                    from datetime import datetime
                                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                                    formatted_time = dt.strftime("%H:%M:%S")
                                    st.write(f"• **Time**: {formatted_time}")
                                except:
                                    st.write(f"• **Time**: {timestamp}")
                            
                            # Show error if present
                            if execution.get('error'):
                                st.write("**Error:**")
                                st.error(execution['error'])
                        
                        with detail_col2:
                            # Input arguments
                            st.write("**Input Arguments:**")
                            input_args = execution.get('input_args', {})
                            if input_args:
                                st.json(input_args)
                            else:
                                st.info("No input arguments")
                            
                            # Output
                            st.write("**Output:**")
                            output = execution.get('output', {})
                            if output:
                                st.json(output)
                            else:
                                st.info("No output available")
                
                # Show pagination info if there are more than 20 executions
                if len(filtered_history) > 20:
                    st.info(f"Showing latest 20 of {len(filtered_history)} executions. Use filters to narrow down results.")
            else:
                st.info("No executions match the current filters.")
        else:
            st.info("No tool executions recorded yet. Tool executions will appear here when you use PydanticAI tools.")
            
            # Show helpful information about tool execution tracking
            st.markdown("""
            **Tool Execution History tracks:**
            - Tool name and execution timestamp
            - Input parameters and output results  
            - Execution time in milliseconds
            - Success/failure status and method used
            - Error details when tools fail
            
            To see tool executions:
            1. Enable PydanticAI Agent in the sidebar
            2. Load some sample tools or create custom tools
            3. Ask the AI to use tools in your conversation
            """)
    
    with tab5:
        st.subheader("Current Configuration")
        
        # Add configuration management info
        st.info("💡 **Configuration Management**: Use the Save/Load buttons at the top of the sidebar to backup and restore your settings.")
        
        config_col1, config_col2 = st.columns(2)
        
        with config_col1:
            st.write("**OpenAI Settings:**")
            st.write(f"• Model: {st.session_state.openai_model}")
            st.write(f"• API Key: {'✅ Set' if st.session_state.openai_api_key else '❌ Not Set'}")
            
            st.write("**Memory Settings:**")
            st.write(f"• Memory Context: {'✅ Enabled' if st.session_state.use_memory_context else '❌ Disabled'}")
            st.write(f"• Retrieval Mode: {st.session_state.memory_retrieval_mode}")
            st.write(f"• Max Memories: {st.session_state.max_memories}")
            st.write(f"• Relevance Threshold: {st.session_state.memory_relevance_threshold}")
        
        with config_col2:
            st.write("**PydanticAI Settings:**")
            st.write(f"• Agent: {'✅ Enabled' if st.session_state.use_pydantic_agent else '❌ Disabled'}")
            st.write(f"• System Prompt: {SYSTEM_PROMPT_TEMPLATES[st.session_state.system_prompt_template]['name']}")
            st.write(f"• Custom Tools: {len(st.session_state.custom_tools)} loaded")
            
            st.write("**Memory Service:**")
            st.write(f"• API URL: {st.session_state.api_url}")
            st.write(f"• User ID: {st.session_state.current_user_id}")
            st.write(f"• Conversation ID: {st.session_state.current_conversation_id}")
            
            st.write("**Chat Status:**")
            st.write(f"• Messages in Session: {len(st.session_state.chat_messages)}")
            st.write(f"• Conversation History: {len(st.session_state.conversation_history)}")
        
        # Configuration export info
        st.divider()
        st.subheader("📋 Configuration Details")
        
        # Show which settings are saved
        st.markdown("""
        **Settings included in configuration export:**
        - OpenAI model selection (API key not saved for security)
        - Memory context settings and thresholds
        - PydanticAI agent settings and system prompt templates
        - Custom tool definitions and configurations
        - Memory service connection settings
        - User and conversation IDs
        
        **Settings NOT saved:**
        - OpenAI API key (for security reasons)
        - Chat message history
        - Current memory context
        - Tool execution history
        """)
    
    # Add a refresh button
    if st.button("🔄 Refresh Context", help="Refresh the context window display"):
        st.rerun()

def main():
    """Main Streamlit application"""
    
    # Initialize session state
    init_session_state()
    
    # Header
    st.title("🧠 AI Memory Service with Chat")
    st.markdown("Chat with an AI agent that can store and retrieve memories from conversation threads.")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Configuration Load/Save Section
        st.subheader("💾 Configuration Management")
        config_col1, config_col2 = st.columns(2)
        
        with config_col1:
            # Save Configuration Button
            if st.button("💾 Save Config", use_container_width=True, help="Save current configuration as JSON file"):
                try:
                    config_state = get_serializable_state()
                    config_json = json.dumps(config_state, indent=2)
                    
                    # Create download button for the configuration
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"ai_memory_config_{timestamp}.json"
                    
                    st.download_button(
                        label="📥 Download Config",
                        data=config_json,
                        file_name=filename,
                        mime="application/json",
                        use_container_width=True,
                        help="Click to download the configuration file"
                    )
                    st.success("✅ Configuration ready for download!")
                    
                except Exception as e:
                    st.error(f"❌ Error saving configuration: {str(e)}")
        
        with config_col2:
            # Load Configuration File Upload
            uploaded_config = st.file_uploader(
                "📤 Load Config", 
                type=['json'],
                help="Upload a previously saved configuration JSON file",
                label_visibility="collapsed"
            )
            
            if uploaded_config is not None:
                try:
                    # Read and parse the uploaded JSON file
                    config_data = json.loads(uploaded_config.read())
                    
                    # Load the configuration
                    if load_state_from_dict(config_data):
                        st.success("✅ Configuration loaded successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to load configuration")
                        
                except json.JSONDecodeError:
                    st.error("❌ Invalid JSON file format")
                except Exception as e:
                    st.error(f"❌ Error loading configuration: {str(e)}")
        
        st.divider()  # Separator between config management and regular settings
        
        # OpenAI Configuration
        st.subheader("🤖 OpenAI Settings")
        openai_api_key = st.text_input(
            "OpenAI API Key",
            value=st.session_state.openai_api_key,
            type="password",
            help="Your OpenAI API key"
        )
        st.session_state.openai_api_key = openai_api_key
        
        openai_model = st.selectbox(
            "OpenAI Model",
            options=["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-5", "gpt-5-mini"],
            index=["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-5", "gpt-5-mini"].index(st.session_state.openai_model),
            help="Choose the OpenAI model to use"
        )
        st.session_state.openai_model = openai_model
        
        # System Prompt Configuration
        st.subheader("📝 System Prompt")
        system_prompt_template = st.selectbox(
            "System Prompt Template",
            options=list(SYSTEM_PROMPT_TEMPLATES.keys()),
            index=list(SYSTEM_PROMPT_TEMPLATES.keys()).index(st.session_state.system_prompt_template),
            format_func=lambda x: SYSTEM_PROMPT_TEMPLATES[x]["name"],
            help="Choose a pre-defined system prompt template or create a custom one"
        )
        st.session_state.system_prompt_template = system_prompt_template
        
        # Show template description
        template_info = SYSTEM_PROMPT_TEMPLATES[system_prompt_template]
        st.caption(f"📄 {template_info['description']}")
        
        # Custom system prompt input for custom template
        if system_prompt_template == "custom":
            custom_system_prompt = st.text_area(
                "Custom System Prompt",
                value=st.session_state.custom_system_prompt,
                height=150,
                placeholder="Enter your custom system prompt here...",
                help="Define your own system prompt to customize the AI's behavior"
            )
            st.session_state.custom_system_prompt = custom_system_prompt
        else:
            # Show preview of selected template
            with st.expander("🔍 Preview Template", expanded=False):
                st.text_area(
                    "Template Content:",
                    value=template_info["prompt"],
                    height=150,
                    disabled=True
                )
        
        # PydanticAI Agent Configuration
        st.subheader("🛠️ PydanticAI Agent")
        use_pydantic_agent = st.checkbox(
            "Use PydanticAI Agent with Tools",
            value=st.session_state.use_pydantic_agent,
            help="Enable PydanticAI agent with custom tools (experimental)"
        )
        st.session_state.use_pydantic_agent = use_pydantic_agent
        
        if use_pydantic_agent:
            # Tool management section
            with st.expander("🔧 Manage Tools", expanded=False):
                st.write("**Sample Tools:**")
                if st.button("Load Sample Tools"):
                    logger.info("Loading sample tools")
                    try:
                        st.session_state.custom_tools = create_sample_tools()
                        st.session_state.tool_system = None  # Reset tool system
                        logger.info(f"Loaded {len(st.session_state.custom_tools)} sample tools")
                        st.success("✅ Sample tools loaded!")
                    except Exception as e:
                        logger.error(f"Failed to load sample tools: {e}")
                        st.error(f"❌ Failed to load sample tools: {e}")
                    st.rerun()
                
                st.write("**Current Tools:**")
                if st.session_state.custom_tools:
                    for i, tool in enumerate(st.session_state.custom_tools):
                        with st.container():
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"**{tool.name}**: {tool.description}")
                            with col2:
                                if st.button("🗑️", key=f"delete_tool_{i}"):
                                    st.session_state.custom_tools.pop(i)
                                    st.session_state.tool_system = None  # Reset tool system
                                    st.rerun()
                else:
                    st.info("No tools loaded. Click 'Load Sample Tools' to get started.")
                
                # Add custom tool
                st.write("**Add Custom Tool:**")
                with st.form("add_tool_form"):
                    tool_name = st.text_input("Tool Name", placeholder="e.g., user_preference")
                    tool_description = st.text_input("Description", placeholder="e.g., Get user preferences and risk score")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        input_schema_text = st.text_area(
                            "Input Schema (JSON)",
                            height=100,
                            placeholder='{"user_id": {"type": "string", "required": true}}'
                        )
                    with col2:
                        output_schema_text = st.text_area(
                            "Output Schema (JSON)", 
                            height=100,
                            placeholder='{"user_risk_score": [{"score": "int"}]}'
                        )
                    
                    mock_response_text = st.text_area(
                        "Mock Response (JSON, optional)",
                        height=60,
                        placeholder='{"user_risk_score": [{"score": 92}]}'
                    )
                    
                    if st.form_submit_button("Add Tool"):
                        try:
                            input_schema = parse_tool_schema(input_schema_text)
                            output_schema = parse_tool_schema(output_schema_text)
                            mock_response = None
                            if mock_response_text.strip():
                                mock_response = parse_tool_schema(mock_response_text)
                            
                            new_tool = ToolDefinition(
                                name=tool_name,
                                description=tool_description,
                                input_schema=input_schema,
                                output_schema=output_schema,
                                mock_response=mock_response
                            )
                            
                            st.session_state.custom_tools.append(new_tool)
                            st.session_state.tool_system = None  # Reset tool system
                            st.success(f"✅ Tool '{tool_name}' added!")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Error adding tool: {str(e)}")
        
        use_memory_context = st.checkbox(
            "Use Memory Context",
            value=st.session_state.use_memory_context,
            help="Include relevant memory context in AI responses"
        )
        st.session_state.use_memory_context = use_memory_context
        
        # Memory retrieval options (only show if memory context is enabled)
        if use_memory_context:
            st.write("**Memory Retrieval Options:**")
            
            memory_retrieval_mode = st.selectbox(
                "Memory Retrieval Mode",
                options=["automatic", "query_based", "disabled"],
                index=["automatic", "query_based", "disabled"].index(st.session_state.get("memory_retrieval_mode", "automatic")),
                help="How to retrieve memory context:\n"
                     "• Automatic: Always retrieve memories for each message\n"
                     "• Query-based: Only retrieve when user query seems to need context\n"
                     "• Disabled: Never retrieve (but still store messages)"
            )
            st.session_state.memory_retrieval_mode = memory_retrieval_mode
            
            max_memories = st.slider(
                "Max Similar Memories",
                min_value=1,
                max_value=10,
                value=st.session_state.get("max_memories", 3),
                help="Maximum number of similar memories to include in context"
            )
            st.session_state.max_memories = max_memories
            
            include_conversation_summary = st.checkbox(
                "Include Conversation Summary",
                value=st.session_state.get("include_conversation_summary", True),
                help="Include AI-generated summary of related conversations"
            )
            st.session_state.include_conversation_summary = include_conversation_summary
            
            memory_relevance_threshold = st.slider(
                "Memory Relevance Threshold",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.get("memory_relevance_threshold", 0.5),
                step=0.1,
                help="Minimum similarity score to include a memory (0.0 = include all, 1.0 = only exact matches)"
            )
            st.session_state.memory_relevance_threshold = memory_relevance_threshold
        
        st.divider()
        
        # Memory Service Configuration
        st.subheader("🧠 Memory Service")
        api_url = st.text_input(
            "API Base URL",
            value=st.session_state.api_url,
            help="Base URL of the AI Memory Service API"
        )
        st.session_state.api_url = api_url
        
        # User and conversation configuration
        user_id = st.text_input(
            "User ID",
            value=st.session_state.current_user_id,
            help="Unique identifier for the user"
        )
        st.session_state.current_user_id = user_id
        
        conversation_id = st.text_input(
            "Conversation Thread ID",
            value=st.session_state.current_conversation_id,
            help="Unique identifier for the conversation thread"
        )
        st.session_state.current_conversation_id = conversation_id
        
        # Health check
        st.subheader("🏥 Service Status")
        if st.button("Check Service Health"):
            if not api_url.strip():
                st.error("Please enter API Base URL")
            else:
                client = MemoryServiceClient(api_url)
                try:
                    health = run_async(client.health_check())
                    st.success(f"✅ Service is healthy: {health}")
                except Exception as e:
                    st.error(f"❌ Service unavailable: {str(e)}")
        
        # Sample Data Generation
        st.subheader("🎲 Generate Sample Data")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎯 Random Username"):
                new_username = generate_random_username()
                st.session_state.current_user_id = new_username
                st.success(f"Generated: {new_username}")
                st.rerun()
        
        with col2:
            if st.button("💬 Random Conversation ID"):
                new_conv_id = generate_random_conversation_id()
                st.session_state.current_conversation_id = new_conv_id
                st.success(f"Generated: {new_conv_id}")
                st.rerun()
        
        if st.button("🚀 Generate Sample Conversations", type="primary"):
            if not api_url.strip():
                st.error("Please enter API Base URL")
            elif not st.session_state.current_user_id.strip():
                st.error("Please enter a User ID")
            elif not st.session_state.current_conversation_id.strip():
                st.error("Please enter a Conversation ID")
            else:
                try:
                    client = MemoryServiceClient(api_url)
                    with st.spinner("Generating sample conversations..."):
                        topic, message_count = run_async(populate_sample_conversations(
                            client, 
                            st.session_state.current_user_id, 
                            st.session_state.current_conversation_id
                        ))
                    st.success(f"✅ Generated {message_count} messages about '{topic}'")
                    st.info("💡 Try asking the AI about what was discussed or search for memories!")
                except Exception as e:
                    st.error(f"❌ Failed to generate conversations: {str(e)}")
                    logger.error(f"Failed to generate sample conversations: {e}")
        
        st.caption("💡 Use sample data to test memory functionality with realistic conversations")
        
        # Clear conversation history
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_messages = []
            st.session_state.conversation_history = []
            st.rerun()
        
        # Debug logs section
        display_debug_logs()
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    # Left column: Chat Interface
    with col1:
        st.header("💬 Chat with AI Agent")
        
        # Memory status indicator
        if st.session_state.use_memory_context:
            mode = st.session_state.memory_retrieval_mode
            if mode == "automatic":
                st.info("🧠 Memory: Auto-retrieval enabled")
            elif mode == "query_based":
                st.info("🧠 Memory: Query-based retrieval")
            else:
                st.warning("🧠 Memory: Retrieval disabled")
        else:
            st.warning("🧠 Memory: Context disabled")
        
        # Tool system status
        if st.session_state.use_pydantic_agent and st.session_state.custom_tools:
            tool_count = len(st.session_state.custom_tools)
            tool_names = [tool.name for tool in st.session_state.custom_tools]
            st.info(f"🛠️ PydanticAI Agent: {tool_count} tools available")
            with st.expander("🔍 Tool Details", expanded=False):
                for tool in st.session_state.custom_tools:
                    st.write(f"• **{tool.name}**: {tool.description}")
                    if tool.mock_response:
                        st.caption("  (Uses mock response)")
            
            # Tool system debug info
            if st.session_state.tool_system:
                st.success("✅ Tool system initialized")
            else:
                st.warning("⚠️ Tool system not initialized")
        
        # Validate API key
        if not openai_api_key.strip():
            st.warning("⚠️ Please enter your OpenAI API key in the sidebar to start chatting.")
        else:
            # Display chat messages
            chat_container = st.container(height=400)
            with chat_container:
                for message in st.session_state.chat_messages:
                    with st.chat_message(message["role"]):
                        st.write(message["content"])
                        if "timestamp" in message:
                            st.caption(format_timestamp(message["timestamp"]))
                        # Display tool calls if present
                        if "tool_calls" in message and message["tool_calls"]:
                            display_tool_calls(message["tool_calls"], st)
            
            # Chat input
            if prompt := st.chat_input("Type your message here..."):
                if not user_id.strip():
                    st.error("Please enter a User ID in the sidebar.")
                elif not conversation_id.strip():
                    st.error("Please enter a Conversation ID in the sidebar.")
                else:
                    # Add user message to chat
                    user_message = {
                        "role": "user",
                        "content": prompt,
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                    st.session_state.chat_messages.append(user_message)
                    
                    # Display user message
                    with chat_container:
                        with st.chat_message("user"):
                            st.write(prompt)
                            st.caption(format_timestamp(user_message["timestamp"]))
                    
                    try:
                        # Store user message in memory service
                        memory_client = MemoryServiceClient(api_url)
                        run_async(memory_client.add_message(
                            user_id=user_id,
                            conversation_id=conversation_id,
                            message_type="human",
                            text=prompt
                        ))
                        
                        # Get memory context based on retrieval mode and settings
                        memory_context = ""
                        if use_memory_context and should_retrieve_memories(prompt, st.session_state.memory_retrieval_mode):
                            try:
                                with st.spinner("Retrieving relevant memories..."):
                                    memory_results = run_async(memory_client.retrieve_memory(
                                        user_id=user_id,
                                        text=prompt
                                    ))
                                
                                # Format memory context using new function
                                memory_context = format_memory_context(
                                    memory_results=memory_results,
                                    max_memories=st.session_state.max_memories,
                                    include_summary=st.session_state.include_conversation_summary,
                                    relevance_threshold=st.session_state.memory_relevance_threshold
                                )
                                
                                # Store memory context in session state for context window
                                st.session_state.current_memory_context = memory_context
                                
                                # Show memory retrieval status in sidebar
                                if memory_context:
                                    st.sidebar.success(f"✅ Retrieved {len(memory_results.get('similar_memories', []))} memories")
                                else:
                                    st.sidebar.info("ℹ️ No relevant memories found")
                                
                            except Exception as e:
                                st.sidebar.warning(f"Could not retrieve memory context: {str(e)}")
                        elif use_memory_context:
                            st.sidebar.info("ℹ️ Memory retrieval skipped (query-based mode)")
                            st.session_state.current_memory_context = ""  # Clear context
                        else:
                            st.sidebar.info("ℹ️ Memory context disabled")
                            st.session_state.current_memory_context = ""  # Clear context
                        
                        # Generate AI response
                        ai_response = ""
                        tool_calls = []
                        
                        if st.session_state.use_pydantic_agent and st.session_state.custom_tools:
                            # Use PydanticAI agent with tools
                            logger.info("Using PydanticAI agent with tools")
                            logger.info(f"Number of custom tools: {len(st.session_state.custom_tools)}")
                            
                            try:
                                # Initialize or recreate tool system if needed
                                if (st.session_state.tool_system is None or 
                                    st.session_state.tool_system.openai_api_key != openai_api_key or
                                    st.session_state.tool_system.model != openai_model):
                                    
                                    logger.info("Creating new tool system")
                                    st.session_state.tool_system = GenericToolSystem(openai_api_key, openai_model)
                                    
                                    # Add all custom tools
                                    for tool_def in st.session_state.custom_tools:
                                        logger.info(f"Adding tool to system: {tool_def.name}")
                                        st.session_state.tool_system.add_tool(tool_def)
                                    
                                    # Get the appropriate system prompt
                                    if st.session_state.system_prompt_template == "custom":
                                        base_system_prompt = st.session_state.custom_system_prompt
                                        logger.info("Using custom system prompt")
                                    else:
                                        base_system_prompt = SYSTEM_PROMPT_TEMPLATES[st.session_state.system_prompt_template]["prompt"]
                                        logger.info(f"Using template: {st.session_state.system_prompt_template}")
                                    
                                    # Create the agent with memory-aware system prompt
                                    system_prompt = f"""{base_system_prompt}

You have access to various tools that can help you provide better responses. Use them when appropriate.

{f"Relevant memory context: {memory_context}" if memory_context else ""}

Recent conversation context:
{chr(10).join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_messages[-5:]])}
"""
                                    logger.info("Creating agent with system prompt")
                                    st.session_state.tool_system.create_agent(system_prompt)
                                    logger.info("Agent created successfully")
                                
                                # Get AI response using PydanticAI
                                with st.spinner("AI agent is thinking..."):
                                    logger.info("Running PydanticAI agent")
                                    logger.debug(f"Prompt: {prompt}")
                                    logger.debug(f"Deps: {{'user_id': '{user_id}', 'conversation_id': '{conversation_id}'}}")
                                    
                                    result = st.session_state.tool_system.agent.run_sync(
                                        prompt, 
                                        deps={"user_id": user_id, "conversation_id": conversation_id}
                                    )
                                    logger.info("PydanticAI agent run completed")
                                    logger.debug(f"Result type: {type(result)}")
                                    logger.debug(f"Result data: {result.data}")
                                    
                                    ai_response = str(result.data)
                                    
                                    # Capture tool calls for display
                                    if hasattr(result, 'all_messages') and callable(result.all_messages):
                                        all_messages = result.all_messages()
                                        logger.info(f"Found {len(all_messages)} messages in result")
                                        for msg in all_messages:
                                            logger.debug(f"Message type: {type(msg)}, attributes: {dir(msg)}")
                                            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                                logger.info(f"Found tool calls: {len(msg.tool_calls)}")
                                                for tool_call in msg.tool_calls:
                                                    tool_calls.append(tool_call)
                                                    # Store tool call info for context window
                                                    call_info = {
                                                        'tool_name': getattr(tool_call, 'tool_name', 'Unknown'),
                                                        'args': getattr(tool_call, 'args', {}),
                                                        'result': getattr(tool_call, 'result', None)
                                                    }
                                                    st.session_state.last_tool_calls.append(call_info)
                                    else:
                                        logger.info("No all_messages method found or it's not callable")
                                        logger.info("No messages or tool calls found in result")
                                        
                            except Exception as e:
                                logger.error(f"PydanticAI error: {e}")
                                logger.exception("Full traceback:")
                                st.error(f"❌ PydanticAI error: {str(e)}")
                                
                                # Show detailed error information
                                with st.expander("🔍 Error Details", expanded=False):
                                    st.write(f"**Error Type**: {type(e).__name__}")
                                    st.write(f"**Error Message**: {str(e)}")
                                    st.write(f"**Tool System State**: {st.session_state.tool_system is not None}")
                                    if st.session_state.tool_system:
                                        st.write(f"**Number of Tools**: {len(st.session_state.tool_system.tools)}")
                                        st.write(f"**Tool Names**: {list(st.session_state.tool_system.tools.keys())}")
                                        st.write(f"**Agent State**: {st.session_state.tool_system.agent is not None}")
                                    
                                    import traceback
                                    st.code(traceback.format_exc())
                                
                                # Fallback to regular OpenAI
                                logger.info("Falling back to regular OpenAI")
                                openai_client = OpenAIClient(openai_api_key, openai_model)
                                
                                # Get the appropriate system prompt
                                if st.session_state.system_prompt_template == "custom":
                                    base_system_prompt = st.session_state.custom_system_prompt
                                else:
                                    base_system_prompt = SYSTEM_PROMPT_TEMPLATES[st.session_state.system_prompt_template]["prompt"]
                                
                                openai_messages = [
                                    {"role": "system", "content": base_system_prompt}
                                ]
                                
                                # Add recent chat history (last 10 messages)
                                recent_messages = st.session_state.chat_messages[-10:]
                                for msg in recent_messages:
                                    if msg["role"] in ["user", "assistant"]:
                                        openai_messages.append({
                                            "role": msg["role"], 
                                            "content": msg["content"]
                                        })
                                
                                with st.spinner("AI is thinking..."):
                                    ai_response = run_async(openai_client.chat_completion(
                                        messages=openai_messages,
                                        memory_context=memory_context if use_memory_context else ""
                                    ))
                        else:
                            # Use regular OpenAI client
                            openai_client = OpenAIClient(openai_api_key, openai_model)
                            
                            # Get the appropriate system prompt
                            if st.session_state.system_prompt_template == "custom":
                                                               base_system_prompt = st.session_state.custom_system_prompt
                            else:
                                base_system_prompt = SYSTEM_PROMPT_TEMPLATES[st.session_state.system_prompt_template]["prompt"]
                            
                            # Prepare messages for OpenAI
                            openai_messages = [
                                {"role": "system", "content": base_system_prompt}
                            ]
                            
                            # Add recent chat history (last 10 messages)
                            recent_messages = st.session_state.chat_messages[-10:]
                            for msg in recent_messages:
                                if msg["role"] in ["user", "assistant"]:
                                    openai_messages.append({
                                        "role": msg["role"], 
                                        "content": msg["content"]
                                    })
                            
                            # Get AI response
                            with st.spinner("AI is thinking..."):
                                ai_response = run_async(openai_client.chat_completion(
                                    messages=openai_messages,
                                    memory_context=memory_context if use_memory_context else ""
                                ))
                        
                        # Add AI response to chat
                        ai_message = {
                            "role": "assistant",
                            "content": ai_response,
                            "timestamp": datetime.datetime.now().isoformat(),
                            "tool_calls": tool_calls
                        }
                        st.session_state.chat_messages.append(ai_message)
                        
                        # Display AI response
                        with chat_container:
                            with st.chat_message("assistant"):
                                st.write(ai_response)
                                st.caption(format_timestamp(ai_message["timestamp"]))
                                # Display tool calls if any
                                if tool_calls:
                                    display_tool_calls(tool_calls, st)
                        
                        # Store AI response in memory service
                        run_async(memory_client.add_message(
                            user_id=user_id,
                            conversation_id=conversation_id,
                            message_type="ai",
                            text=ai_response
                        ))
                        
                        # Update conversation history
                        st.session_state.conversation_history.extend([
                            {
                                "type": "human",
                                "text": prompt,
                                "timestamp": user_message["timestamp"],
                                "user_id": user_id,
                                "conversation_id": conversation_id
                            },
                            {
                                "type": "ai",
                                "text": ai_response,
                                "timestamp": ai_message["timestamp"],
                                "user_id": user_id,
                                "conversation_id": conversation_id
                            }
                        ])
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error during chat: {str(e)}")
                        st.exception(e)
    
    # Right column: Retrieve Memory
    with col2:
        st.header("🔍 Retrieve Memory")
        
        with st.form("retrieve_memory_form"):
            query_text = st.text_area(
                "Search Query",
                height=150,
                placeholder="Enter your search query here...",
                help="Query to search for relevant memories and conversations"
            )
            
            retrieve_submitted = st.form_submit_button("🔍 Search Memories")
            
            if retrieve_submitted:
                if not query_text.strip():
                    st.error("Please enter a search query.")
                elif not user_id.strip():
                    st.error("Please enter a User ID.")
                else:
                    try:
                        client = MemoryServiceClient(api_url)
                        
                        with st.spinner("Searching memories..."):
                            results = run_async(client.retrieve_memory(
                                user_id=user_id,
                                text=query_text
                            ))
                        
                        st.success("✅ Memory search completed!")
                        
                        # Display results in a structured format
                        display_memory_results(results)
                        
                    except Exception as e:
                        st.error(f"❌ Error retrieving memories: {str(e)}")
                        st.exception(e)  # Show detailed error information
    
    # Conversation History Section
    st.header("💬 Current Session History")
    
    if st.session_state.conversation_history:
        for i, msg in enumerate(reversed(st.session_state.conversation_history)):
            with st.container():
                col1, col2, col3 = st.columns([2, 3, 1])
                
                with col1:
                    icon = "🤖" if msg["type"] == "ai" else "👤"
                    st.write(f"{icon} **{msg['type'].title()}**")
                
                with col2:
                    st.write(msg["text"][:100] + "..." if len(msg["text"]) > 100 else msg["text"])
                
                with col3:
                    st.caption(format_timestamp(msg["timestamp"]))
                
                if i < len(st.session_state.conversation_history) - 1:
                    st.divider()
    else:
        st.info("No messages in current session. Send a message to get started!")
    
    # Context Window
    st.divider()
    display_context_window()
    
    # Instructions
    with st.expander("ℹ️ How to Use"):
        st.markdown("""
        ### Getting Started
        
        1. **Configure OpenAI**: Enter your OpenAI API key in the sidebar to enable chat functionality
        2. **Configure System Prompt**: Choose from templates (Helpful Assistant, Business Advisor, Learning Tutor, Code Mentor) or create custom prompts
        3. **Configure PydanticAI Agent**: Optionally enable tools for enhanced AI capabilities
        4. **Configure Memory Service**: Set your AI Memory Service API URL, User ID, and Conversation Thread ID
        5. **Configure Memory Retrieval**: Choose how and when to retrieve memory context
        6. **Check Service**: Verify the AI Memory Service is running using the health check
        7. **Start Chatting**: Use the chat interface to talk with the AI agent
        8. **Search Memories**: Use the memory retrieval panel to search past conversations
        
        ### Sample Data Generation 🎲
        
        **Quick Start with Sample Data:**
        - **Random Username**: Generate creative usernames like "amazing_developer_123"
        - **Random Conversation ID**: Create topic-based IDs like "project_planning_20250730_abc123"
        - **Sample Conversations**: Populate realistic conversation history for testing
        
        **Sample Topics Available:**
        - AI Project Planning discussions
        - Code Review conversations  
        - Memory System Design talks
        - Tool Development guidance
        
        💡 **Pro Tip**: Generate sample conversations first, then ask questions like "What did we discuss about AI projects?" to test memory retrieval!
        
        ### System Prompt Templates 📝
        
        - **Helpful Assistant**: General-purpose, friendly AI assistant
        - **Business Advisor**: Strategic consultant with business expertise
        - **Learning Tutor**: Educational assistant focused on teaching
        - **Code Mentor**: Programming expert with best practices guidance
        - **Custom**: Define your own system prompt for specialized behavior
        
        ### PydanticAI Tools System 🛠️
        
        - **Tool Playground**: Create custom tools with JSON schemas for input/output
        - **Sample Tools**: Use pre-built tools like user preferences, weather lookup, and investment calculator
        - **Dynamic Tool Creation**: Add tools on-the-fly with custom schemas
        - **Tool Call Visualization**: See which tools were called and their results under each AI response
        
        ### Memory Retrieval Modes 🧠
        
        - **Automatic**: Always retrieve relevant memories for every message (recommended for ongoing conversations)
        - **Query-based**: Only retrieve memories when the user query suggests they need context (keywords like "remember", "what did we discuss", etc.)
        - **Disabled**: Never retrieve memories but still store all messages
        
        ### Tips & Best Practices
        
        - Use descriptive User IDs to organize memories by different users or personas
        - Use meaningful Conversation Thread IDs to group related conversations (e.g., "project_alpha", "support_ticket_123")
        - Enable "Use Memory Context" to make the AI aware of past conversations
        - Try asking the AI to use tools: "What's the weather in New York?" or "Get my user preferences"
        - Search queries can be questions, keywords, or concepts
        - The system uses AI embeddings to find semantically similar content
        - Generate sample conversations to test memory functionality before real usage
        - Experiment with different system prompts for different use cases
        - Clear chat history when starting a new conversation topic
        """)

if __name__ == "__main__":
    main()