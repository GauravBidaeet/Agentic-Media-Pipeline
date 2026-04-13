import os
import json
from datetime import datetime, timedelta

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from crewai.tools import tool
from crewai import LLM

# --- Google Calendar API imports ---
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from dotenv import load_dotenv
from langchain_groq import ChatGroq
load_dotenv()

# ============================================================
# Configuration & Security
# ============================================================
SCOPES = ["https://www.googleapis.com/auth/calendar"]
CREDENTIALS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "credentials.json"))
TOKEN_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "token.json"))

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("FATAL: GROQ_API_KEY environment variable is missing.")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=api_key,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

# ============================================================
# Pydantic schemas for structured LLM output
# ============================================================
class ScheduledTask(BaseModel):
    summary: str = Field(description="Short title for the calendar event, e.g. 'Review candidate profile'")
    description: str = Field(description="A more detailed description of the task")
    start_datetime: str = Field(description="ISO 8601 datetime string for when the task should start, e.g. '2026-04-08T10:00:00'")
    duration_minutes: int = Field(description="How long the task should take in minutes, e.g. 30, 60, 120")

class ScheduledTaskList(BaseModel):
    tasks: list[ScheduledTask] = Field(description="A list of tasks with assigned dates, times, and durations")

# ============================================================
# Helpers
# ============================================================
def _get_calendar_service():
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDENTIALS_FILE):
                raise FileNotFoundError(f"Missing '{CREDENTIALS_FILE}'.")
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(TOKEN_FILE, "w") as token_file:
            token_file.write(creds.to_json())

    return build("calendar", "v3", credentials=creds)

def _schedule_tasks_with_llm(tasks: list[str]) -> list[ScheduledTask]:
    parser = PydanticOutputParser(pydantic_object=ScheduledTaskList)
    now = datetime.now()
    base_date = (now + timedelta(days=1)).strftime("%Y-%m-%d")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a smart scheduling assistant. Given a list of tasks, assign each task a reasonable start date/time and duration.

Rules:
- Use {base_date} as the earliest date (tomorrow).
- Schedule tasks during business hours (9:00 AM to 6:00 PM).
- Space tasks out — don't put everything at the same time.
- Give each task a realistic duration (minimum 30 minutes).
- Create a short, clear summary and a detailed description for each.
- Output strictly according to the provided schema.

{format_instructions}"""),
        ("human", "Schedule the following tasks:\n\n{tasks}")
    ])

    prompt = prompt.partial(
        format_instructions=parser.get_format_instructions(),
        base_date=base_date,
    )

    chain = prompt | llm | parser
    result = chain.invoke({"tasks": "\n".join(f"- {t}" for t in tasks)})
    return result.tasks

def _create_calendar_events(service, scheduled_tasks: list[ScheduledTask]) -> list[dict]:
    created_events = []
    for task in scheduled_tasks:
        start_dt = datetime.fromisoformat(task.start_datetime)
        end_dt = start_dt + timedelta(minutes=task.duration_minutes)

        event_body = {
            "summary": task.summary,
            "description": task.description,
            "start": {"dateTime": start_dt.isoformat(), "timeZone": "Asia/Kolkata"},
            "end": {"dateTime": end_dt.isoformat(), "timeZone": "Asia/Kolkata"},
            "reminders": {"useDefault": False, "overrides": [{"method": "popup", "minutes": 30}]},
        }

        event = service.events().insert(calendarId="primary", body=event_body).execute()
        created_events.append({
            "summary": task.summary,
            "start": task.start_datetime,
            "duration_minutes": task.duration_minutes,
            "calendar_link": event.get("htmlLink"),
        })
    return created_events

# ============================================================
# LangChain Tool
# ============================================================
@tool
def schedule_tasks_to_calendar(tasks_string: str) -> str:
    """
    Takes a string of tasks separated by the '|' character and schedules them into Google Calendar.
    The LLM assigns appropriate dates, times, and durations automatically.
    
    Args:
        tasks_string (str): A single string containing all tasks separated by a pipe '|'.
                            Example: "Review candidate profile | Send feedback email | Schedule follow-up"

    Returns:
        A formatted string summarizing all created calendar events with their links.
    """
    if not tasks_string or not tasks_string.strip():
        return "Error: No tasks provided to schedule."

    # Parse the string input from the agent back into a proper list
    tasks = [t.strip() for t in tasks_string.split('|') if t.strip()]

    print(f"\n📅 Scheduling {len(tasks)} tasks to Google Calendar...\n")

    try:
        service = _get_calendar_service()
        scheduled_tasks = _schedule_tasks_with_llm(tasks)
        created_events = _create_calendar_events(service, scheduled_tasks)

        output_lines = [f"Success: Scheduled {len(created_events)} tasks to Google Calendar:\n"]
        for i, evt in enumerate(created_events, 1):
            output_lines.append(
                f"{i}. {evt['summary']}\n"
                f"   📅 {evt['start']} ({evt['duration_minutes']} min)\n"
                f"   🔗 {evt['calendar_link']}"
            )
        return "\n".join(output_lines)
    except Exception as e:
        return f"Error scheduling tasks: {str(e)}"