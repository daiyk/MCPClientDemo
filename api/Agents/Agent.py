from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union

class Agent(ABC):
    """
    Base class for AI model agents with support for multiple MCP servers.
    This class serves as a swappable LLM instance for client classes.
    """
    
    def __init__(self, 
                 name,
                 instructions,
                 mcp_servers: Optional[Dict[str, Dict[str, Any]]] = None,
                 **kwargs):
        """
        Initialize the agent.
        This agent can be configured with multiple MCP servers for flexibility.
        It supports querying the model, counting tokens, and managing tools.
        It also provides methods to add, remove, and set active MCP servers.
        It is designed to be subclassed for specific agent implementations.
        
        Args:
            name: The name of the agent
            mcp_servers: Dictionary of MCP server configurations {server_name: config_dict}
            **kwargs: Additional model-specific configuration parameters
        """
        self.name = name
        self.instructions = instructions
        # Initialize MCP server configurations
        self.mcp_servers = mcp_servers or {}
        self.active_mcp_server = next(iter(self.mcp_servers)) if self.mcp_servers else None
        self.config = kwargs
    
    @abstractmethod
    def query(self, 
              messages: List[Dict[str, str]], 
              temperature: float = 0.7,
              max_tokens: Optional[int] = None,
              stream: bool = False,
              tools: Optional[List[Dict[str, Any]]] = None,
              **kwargs) -> Union[str, Dict[str, Any], Any]:
        """
        Query the model with messages and return the response.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Controls randomness (0 = deterministic, 1 = creative)
            max_tokens: Maximum tokens in the response
            stream: Whether to stream the response
            tools: List of tools available for this query
            **kwargs: Additional model-specific parameters
            
        Returns:
            The model's response
        """
        pass
    
    @abstractmethod
    def get_token_count(self, text: str) -> int:
        """
        Count the number of tokens in the given text.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            The number of tokens
        """
        pass
    
    # MCP Server Management
    def add_mcp_server(self, server_name: str, config: Dict[str, Any]) -> None:
        """
        Add or update an MCP server configuration.
        
        Args:
            server_name: Unique identifier for the MCP server
            config: Configuration dictionary for the server
        """
        self.mcp_servers[server_name] = config
        if self.active_mcp_server is None:
            self.active_mcp_server = server_name
    
    def remove_mcp_server(self, server_name: str) -> bool:
        """
        Remove an MCP server configuration.
        
        Args:
            server_name: The name of the server to remove
            
        Returns:
            True if server was removed, False if not found
        """
        if server_name in self.mcp_servers:
            self.mcp_servers.pop(server_name)
            # Update active server if the removed one was active
            if self.active_mcp_server == server_name:
                self.active_mcp_server = next(iter(self.mcp_servers)) if self.mcp_servers else None
            return True
        return False
    
    def set_active_mcp_server(self, server_name: str) -> bool:
        """
        Set the active MCP server.
        
        Args:
            server_name: The name of the server to activate
            
        Returns:
            True if successful, False if server not found
        """
        if server_name in self.mcp_servers:
            self.active_mcp_server = server_name
            return True
        return False
    
    def get_mcp_servers(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all configured MCP servers.
        
        Returns:
            Dictionary of server configurations
        """
        return self.mcp_servers
    
    # MCP Tool Execution
    def execute_tool(self, 
                     tool_name: str, 
                     tool_input: Dict[str, Any],
                     server_name: Optional[str] = None) -> Any:
        """
        Execute a tool with the given input on the specified or active MCP server.
        
        Args:
            tool_name: The name of the tool to execute
            tool_input: The input parameters for the tool
            server_name: The specific MCP server to use (uses active server if None)
            
        Returns:
            The tool's response
            
        Raises:
            ValueError: If server_name is invalid or no active server exists
            NotImplementedError: If tool execution is not implemented by subclass
        """
        target_server = server_name or self.active_mcp_server
        
        if not target_server:
            raise ValueError("No MCP server specified and no active server configured")
        
        if target_server not in self.mcp_servers:
            raise ValueError(f"MCP server '{target_server}' not found")
        
        # Implementation to be provided by subclasses
        raise NotImplementedError("Tool execution must be implemented by subclasses")
    
    def get_available_tools(self, server_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get available tools from the specified or active MCP server.
        
        Args:
            server_name: The server to query (uses active server if None)
            
        Returns:
            List of available tools with their schemas
            
        Raises:
            ValueError: If server_name is invalid or no active server exists
        """
        target_server = server_name or self.active_mcp_server
        
        if not target_server:
            raise ValueError("No MCP server specified and no active server configured")
        
        if target_server not in self.mcp_servers:
            raise ValueError(f"MCP server '{target_server}' not found")
        
        # Implementation to be provided by subclasses
        raise NotImplementedError("Getting available tools must be implemented by subclasses")
    
    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "name": self.name,
            "provider": self.__class__.__name__.replace("Agent", ""),
            "active_mcp_server": self.active_mcp_server,
            "mcp_servers": list(self.mcp_servers.keys())
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.name}, mcp_servers={list(self.mcp_servers.keys())})"