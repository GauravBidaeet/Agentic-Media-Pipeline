from crewai import Task
from agents import video_to_audio_converter, meeting_transcriber, task_analyst, scheduling_agent, research_agent
import os

# Task 1 uses Agent 1
conversion_task = Task(
    description='Convert the video file located at {video_path} into an audio file.',
    expected_output='The absolute path to the generated audio file.',
    agent=video_to_audio_converter
)

# Task 2 uses Agent 2
transcription_task = Task(
    description=(
        'You need to transcribe an audio file. '
        'Read the output provided by the previous conversion task to find the absolute path of the audio file. '
        'Once you have identified the path from the context, pass that exact path into your transcription tool.'
    ),
    expected_output='A confirmation string showing the absolute paths of the transcript and MoM files.',
    agent=meeting_transcriber,
    context=[conversion_task]  # THIS is the dynamic link.
)

# Task 3 uses Agent 3
# Task 3 uses Agent 3
analysis_task = Task(
    description=(
        'Read the output from the transcription task to find the absolute paths '
        'of the generated transcript and MoM files. '
        'Use your extraction tool to extract actionable tasks from those specific files. '
        'Ensure you pass the file paths to your tool as a comma-separated string.'
    ),
    expected_output='A clean, bulleted list of all identified tasks from the meetings.',
    agent=task_analyst,
    context=[transcription_task] # <--- THIS REPLACES YOUR HARDCODED VARIABLE
)

# Task 4 uses Agent 4
calendar_task = Task(
    description=(
        "You have been given the following tasks to execute: '{tasks_list}'. "
        'Pass them into your scheduling tool. You MUST format the tasks as a single string '
        'separated by the pipe symbol "|" before calling the tool. '
        'Example: "Task 1 | Task 2 | Task 3"'
    ),
    expected_output='A confirmation list of the scheduled events and their calendar links.',
    agent=scheduling_agent
)

#Task 5 uses Agent 5
research_task = Task(
    description=(
        "You have been given the following objective: '{search_topic}'.\n\n"
        "DO NOT just provide a text explanation or tutorial. "
        "You MUST use your provided MCP tools to physically execute this objective. "
        "If the task involves Excel, you must create a physical .xlsx file. "
        "You MUST use your provided tools to physically execute this objective.\n\n"
        "EXECUTION PIPELINE FOR EXCEL and CSV TASKS:\n"
        "1. Determine the absolute path for the new file. The current directory is: {os.getcwd()}\n"
        "2. FIRST, use the 'create_blank_excel_file' tool to physically generate the empty .xlsx or .csv file at that absolute path.\n"
        "3. SECOND, use the 'execute_mcp_excel_operation' tool to populate that file with relevant sample data and perform the requested operation (e.g., creating the dynamic named range).\n"
        "Replace <filename> with an appropriate name. Populate it with relevant sample data, and perform the requested operation."
    ),
    expected_output="A confirmation that the file was created and the operation was executed, including the exact absolute file path.",
    agent=research_agent
)