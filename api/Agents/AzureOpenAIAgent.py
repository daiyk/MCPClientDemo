from agents import Agent as OpenAIAgent, Runner, set_tracing_disabled, function_tool, OpenAIChatCompletionsModel
from openai import AsyncAzureOpenAI
import dotenv
import os
from Agent import Agent
from typing import Dict, Any, Optional
dotenv.load_dotenv()
BASE_URL = os.getenv("AZURE_OPENAI_API_BASE")
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
MODEL_NAME = os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-4o")

if not BASE_URL or not API_KEY:
    raise ValueError("Azure OpenAI API base URL and API key must be set in environment variables.")
client = AsyncAzureOpenAI(
            azure_endpoint=BASE_URL,
            api_key= API_KEY,
            api_version=API_VERSION,
        )
# disable tracing for this agent as we don't have OpenAI tracing key
set_tracing_disabled(disabled=True)

class AzureOpenAIAgent(Agent):
    def __init__(self,
                name,
                instructions,
                mcp_servers: Optional[Dict[str, Dict[str, Any]]] = None,
                client: AsyncAzureOpenAI = client,
                **kwargs):
            """
            Initialize the Azure OpenAI agent.
            This agent uses the Azure OpenAI API to interact with the specified model.
            It supports multiple MCP servers for configuration flexibility.
            
            Args:
                name: The name of the agent
                mcp_servers: Dictionary of MCP server configurations {server_name: config_dict}
                **kwargs: Additional model-specific configuration parameters
            """
            super().__init__(name=name, instructions=instructions,mcp_servers=mcp_servers, **kwargs)
            self.client = client
            self.agent = OpenAIAgent(
                 name=name,
                 instructions=instructions,
                 model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client)
                 )
            
                 
            
