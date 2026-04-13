🧠 Agentic AI Meeting Automation System

An end-to-end **agentic AI pipeline** that processes meeting recordings (video/audio), generates transcripts and Minutes of Meeting (MoM), extracts actionable tasks, and autonomously executes or schedules them.

## 🚀 Overview

This project automates the entire lifecycle of a meeting:
🎥 Video / Audio Input
↓
🔊 Audio Extraction
↓
📝 Transcription + MoM Generation
↓
📌 Task Extraction
↓
⚙️ Task Routing (MCP Execution / Scheduling)
↓
📅 Calendar Integration

---

## 🏗️ System Architecture

### Agents

| Agent | Responsibility |
|------|----------------|
| 🎬 Media Converter | Converts video → audio |
| 🧾 Transcriber | Generates transcript + MoM |
| 📊 Task Analyst | Extracts actionable tasks |
| 📅 Scheduler | Adds tasks to Google Calendar |
| 🔎 Research/MCP Agent | Executes automation tasks |

---

## ⚙️ Tech Stack

- **LLM**: Groq (LLaMA 3.3)
- **Framework**: CrewAI
- **Speech**: WhisperX (ASR + Diarization)
- **Automation**: MCP (Model Context Protocol)
- **Calendar**: Google Calendar API
- **Media Processing**: MoviePy
- **Parsing**: LangChain + Pydantic

---

## 📂 Project Structure
├── agents.py # Agent definitions
├── crew.py # Crew orchestration
├── convert_to_audio.py # Video → audio tool
├── whisperx_transcript.py # Transcription + MoM
├── extract_tasks.py # Task extraction logic
├── mcp_tool.py # MCP execution engine
├── schedule_to_calendar.py # Google Calendar integration
├── mcp.json # MCP configuration
├── credentials.json # Google API credentials
└── README.md


---

## 🔧 Setup Instructions

### 1. Clone Repository
git clone <repo-url>
cd <repo-name>

### 2. Install Dependencies
pip install -r requirements.txt

### 3. Environment Variables
create .env file 
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
HF_TOKEN=your_huggingface_token

### 4. Google Calendar Setup
- Enable Google Calendar API
- Download credentials.json
- Place it in project root

### 5. Run the Project
python crew.py
