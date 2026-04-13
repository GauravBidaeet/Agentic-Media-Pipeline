import asyncio
import os
from crewai.tools import tool
from crewai import LLM
from mcp_use import MCPAgent, MCPClient
import openpyxl
from dotenv import load_dotenv
from langchain_groq import ChatGroq
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
# ==========================================
# CORE MCP EXECUTION ENGINE
# ==========================================
async def _run_mcp_task(query: str) -> str:
    """Core logic to initialize and run the MCP Agent using the unified config."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "Error: OPENAI_API_KEY environment variable is missing."

    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Pointing to your SINGLE unified config file
    config_path = os.path.join(current_dir, "mcp.json") 
    
    if not os.path.exists(config_path):
        return f"Error: MCP config file not found at absolute path: {config_path}"

    try:
        client = MCPClient.from_config_file(config_path)
        llm = LLM(model="groq/llama-3.3-70b-versatile",api_key=api_key,temperature=0,max_tokens=None,timeout=None,max_retries=2)
        
        # When you load the unified config, the agent gets access to ALL servers defined in it
        agent = MCPAgent(llm=llm, client=client, max_steps=30)
        
        result = await agent.run(query, max_steps=30)
        return f"Success: MCP Agent completed the task.\nFinal Result:\n{result}"
    except Exception as e:
        return f"Error executing MCP Agent: {str(e)}"

def execute_mcp_sync(query: str) -> str:
    """Synchronous wrapper to safely execute the async MCP engine."""
    try:
        return asyncio.run(_run_mcp_task(query))
    except Exception as e:
        return f"Error in asyncio event loop: {str(e)}"

# ==========================================
# CREWAI TOOLS
# ==========================================

@tool
def execute_mcp_browser_query(query: str) -> str:
    """
    Executes a complex web search or browser interaction using the Model Context Protocol (MCP).
    Pass a clear, descriptive instruction to this tool.
    
    Args:
        query (str): The search query or task instruction for the MCP agent.
                     Example: "Find the best restaurant in San Francisco USING GOOGLE SEARCH"
    """
    return execute_mcp_sync(query)

@tool
def execute_mcp_excel_operation(query: str) -> str:
    """
    Executes Excel spreadsheet operations (create, write, format, analyze) using the Model Context Protocol (MCP).
    Pass a clear, descriptive instruction to this tool to manipulate or create Excel files.
    
    Args:
        query (str): The task instruction for the MCP Excel agent.
                     Example: "Create an Excel file named 'report.xlsx' in the current directory with a pivot table."
    """
    return execute_mcp_sync(query)

@tool
def execute_mcp_power_bi_operation(query: str) -> str:
    """
    Executes Power BI operations (data modeling, scheduled refreshes, querying) using the Model Context Protocol (MCP).
    Pass a clear, descriptive instruction to this tool to interact with Power BI.
    
    Args:
        query (str): The task instruction for the MCP Power BI agent.
                     Example: "Schedule a data set refresh in Power BI service by accessing data set settings."
    """
    return execute_mcp_sync(query)

@tool
def create_blank_excel_file(filename: str) -> str:
    """
    Creates a physical, blank Excel (.xlsx) workbook.
    Pass ONLY the desired filename (e.g., 'report.xlsx'). Do not pass a full path.
    """
    try:
        # 1. Get your actual Windows absolute directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 2. Strip out any hallucinated Linux paths the AI tries to sneak in
        clean_filename = os.path.basename(filename)
        
        # 3. Ensure it has the right extension
        if not clean_filename.endswith('.xlsx'):
            clean_filename += '.xlsx'
            
        # 4. Force the correct absolute path
        absolute_path = os.path.join(current_dir, clean_filename)
        
        # 5. Create the physical file
        wb = openpyxl.Workbook()
        wb.save(absolute_path)
        
        # 6. Feed the exact, correct absolute path BACK to the AI so it can't fail Step 2
        return f"Success: Blank Excel file physically created. The EXACT ABSOLUTE PATH you MUST use for the MCP tool is: {absolute_path}"
        
    except Exception as e:
        return f"Error creating file: {str(e)}"