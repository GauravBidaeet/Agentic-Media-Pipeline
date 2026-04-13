import os
from crewai import Crew, Process, Task
from agents import video_to_audio_converter, meeting_transcriber, task_analyst, research_agent, scheduling_agent
from tasks import conversion_task, transcription_task, analysis_task, research_task, calendar_task
import re

# ==========================================
# CREW DEFINITIONS
# ==========================================
pipeline_crew = Crew(
    agents=[video_to_audio_converter, meeting_transcriber, task_analyst],
    tasks=[conversion_task, transcription_task, analysis_task],
    process=Process.sequential # Task 1 -> Task 2 -> Task 3
)

# NOTE: These crews are defined, but their tasks must be built dynamically 
# or accept inputs from the kickoff method.
scheduling_crew = Crew(
    agents=[scheduling_agent],
    tasks=[calendar_task],
    process=Process.sequential
)

commando_crew = Crew(
    agents=[research_agent],
    tasks=[research_task],
    process=Process.sequential
)

# ==========================================
# EXECUTION PIPELINE
# ==========================================
def main():
    user_provided_path = input("Enter the path to the media file: ")
    
    print("\n Phase 1: Extracting Tasks from Video...")
    # 1. Run the first pipeline
    result = pipeline_crew.kickoff(inputs={'video_path': os.path.abspath(user_provided_path)})
    
    # 2. Parse the result properly using the logic we discussed
    try:
        # result.raw contains the raw string output from the final agent
        clean_tasks = [line.strip().lstrip('-* ').strip() for line in result.raw.split('\n') if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith('-') or line.strip().startswith('*'))]
    except AttributeError:
        # Fallback just in case result is returned as a plain string in older versions
        clean_tasks = [line.strip().lstrip('-* ').strip() for line in str(result).split('\n') if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith('-') or line.strip().startswith('*'))]
    
    #print("\n\n######################")
    #print("Extracted and Cleaned Tasks:")
    for index, task in enumerate(clean_tasks):
        #print(f"[{index}] {task}")
        pass

    if not clean_tasks:
        print("No tasks found. Exiting pipeline.")
        return
    # =========================================================
    # WARNING: This is where your custom routing logic MUST go.
    # You need to decide which tasks from 'clean_tasks' go to 
    # the Scheduler, and which go to the Commando (MCP) agent.
    # =========================================================
    comma_separated_tasks = ", ".join(clean_tasks)

    for i, taske in enumerate(clean_tasks, 1):
        print(f"{i}. {taske}")
    
    tasks_for_commando = input("Enter the task index to automate: ")

    try:
        index = int(tasks_for_commando)
        if 1 <= index <= len(clean_tasks):
            selected_task = clean_tasks[index - 1]
            print(f"Selected Task: {selected_task}")
        else:
            print("Invalid index. Out of range.")

    except ValueError:
        print("Invalid input. Enter a number.")

    remaining_tasks = clean_tasks[:index-1] + clean_tasks[index:]

    scheduling_crew.kickoff(inputs={"tasks_list": remaining_tasks})
    commando_crew.kickoff(inputs={"search_topic": selected_task})


# Protect the execution
if __name__ == "__main__":
    main()