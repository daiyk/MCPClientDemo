from typing import Optional
from contextlib import AsyncExitStack
from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.llm = None
        self.tools = []
        self.message = []

    # connect to MCP server
    async def connect_to_local_mcp(self, server_script_path: str):
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
        command  = "python" if is_python else "node"
        server_parameters = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()
    # Call tools from the MCP server
    # get MCP tool
    # process a query
    # call LLM

    # logging
