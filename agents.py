import os
from crewai import LLM
from convert_to_audio import convert_to_audio
from crewai import Agent
from whisperx_transcript import transcribe_and_generate_mom
from extract_tasks import extract_tasks_from_transcript
from schedule_to_calendar import schedule_tasks_to_calendar
from mcp_tool import execute_mcp_browser_query, execute_mcp_excel_operation, create_blank_excel_file
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
import os

api_key = os.getenv("GROQ_API_KEY")

## LLM
llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=api_key,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

## agent_1
video_to_audio_converter = Agent(
    role='Local Media Conversion Specialist',
    goal='Extract audio from local video files precisely and efficiently.',
    verbose=True,
    memory=False,
    backstory=(
        "You are an expert file system and media conversion assistant. "
        "Your sole responsibility is to take local video file paths and convert them into audio formats."
    ),
    tools=[convert_to_audio],
    llm=llm,
    allow_delegation=False
)

## agent_2
meeting_transcriber = Agent(
    role='Transcription and MoM Specialist',
    goal='Transcribe local audio files and generate Minutes of Meeting (MoM).',
    verbose=True,
    memory=False,
    backstory=(
        "You are an expert transcriber. You take absolute paths to audio file, "
        "transcribe them, and generate detailed MoM using your specialized tools."
    ),
    tools=[transcribe_and_generate_mom], # Assuming this is imported properly
    llm=llm, 
    allow_delegation=False
)

## agent_3
task_analyst = Agent(
    role='Business Analyst',
    goal="""
        You are an intelligent agent designed to extract actionable tasks from transcripts or meeting documents.

        You have access to the following tool:
        - extract_tasks_from_transcript

        YOUR JOB:
        - If the user provides file paths or asks to extract tasks → you MUST call the tool
        - DO NOT attempt to manually extract tasks yourself
        - ALWAYS prefer the tool for accuracy
        - Output exactly what the tool returns, with no extra conversational filler.
    """,
    verbose=True,
    memory=False,
    backstory=(
        "You are a meticulous project manager. You take absolute file paths "
        "to transcripts and meeting minutes, and use your extraction tool to pull out tasks."
    ),
    tools=[extract_tasks_from_transcript], 
    llm=llm, 
    allow_delegation=False
)

## agent_4
scheduling_agent = Agent(
    role='Executive Calendar Assistant',
    goal='Take an assigned list of tasks and schedule them effectively on Google Calendar.',
    verbose=True,
    memory=False,
    backstory='You are a highly organized executive assistant who manages schedules flawlessly.',
    tools=[schedule_tasks_to_calendar], 
    llm=llm, 
    allow_delegation=False
)

## agent_5
research_agent = Agent(
    role='Data Researcher and Processor',
    goal="""
        Execute any specialized task using available tools:
        - For Excel tasks (create, modify, add pivot tables): Use the MCP Excel tool
        - For web research or browser tasks: Use the MCP browser tool
        
        When given a task to create Excel files with pivot tables and data models:
        1. Use your execute_mcp_excel_operation tool
        2. Provide clear instructions to create the Excel file in the current working directory
        3. Example instruction: "Create an Excel file called 'output_data_model.xlsx' in the current directory with a data table and pivot table summary"
    """,
    verbose=True,
    memory=False,
    backstory='You are a specialist who can access both web research tools and Excel/data model creation tools through MCP protocols. You excel at creating professional Excel documents with pivot tables and data models.',
    tools=[execute_mcp_browser_query, execute_mcp_excel_operation, create_blank_excel_file], 
    llm=llm,
    allow_delegation=False
)

