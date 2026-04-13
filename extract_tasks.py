import os
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from crewai.tools import tool
from crewai import LLM
from dotenv import load_dotenv
from langchain_groq import ChatGroq
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=api_key,    
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

# 1. Define your strict schema. 
# Do not leave room for the LLM to hallucinate formats.
class TaskExtraction(BaseModel):
    tasks: list[str] = Field(
        description="A clear, actionable list of technical or business tasks extracted from the provided text (documents, minutes of meeting, or transcripts). Example: 'Create an Excel sheet', 'Add data to the database'."
    )

@tool
def extract_tasks_from_transcript(file_paths_str: str) -> str:
    """
Extract tasks that are:
    Explicitly assigned (e.g., "John will send report"),
    Implicit responsibilities (e.g., "We need to fix this bug" → task exists even if no owner),
    Follow-ups (e.g., "Let's revisit next week"),
    Decisions requiring execution,
    Dependencies or blocked actions,
    Suggestions that imply action,
    Questions that require action

    CRITICAL ESCAPE HATCH (ANTI-HALLUCINATION):
    If the provided text is just a general conversation, an interview, or contains NO actual project tasks, you MUST output exactly: "No actionable tasks found."
    DO NOT invent, infer, or hallucinate tasks just to have an output. It is 100% acceptable to find zero tasks.

    DO NOT ignore:
    Partial sentences,
    Casual language,
    Ambiguous ownership

    If ownership is unclear:
    Set owner = "Unassigned"

    If deadline not mentioned:
    Set deadline = "Not specified"

    Normalize vague time references:
    "tomorrow", "next week" → convert if context exists, else keep raw

    DO NOT summarize — extract tasks only,
    DO NOT merge unrelated tasks,
    DO NOT skip duplicates unless identical,
    DO NOT hallucinate tasks not grounded in text,
    Maintain maximum coverage over conciseness

    Before finalizing output, perform these checks:
    Have all action verbs been captured? (send, fix, review, create, update, check, schedule, discuss, follow-up, etc.),
    Are implicit tasks converted into explicit tasks?,
    Are group discussions converted into individual actionable items?,
    Is any sentence containing intent ignored? If yes → reprocess
    
    Args:
        file_paths_str (str): A comma-separated string of ABSOLUTE file paths to read. 
                              Example: '/absolute/path/transcript.txt, /absolute/path/mom.txt'
                              
    Returns:
        str: A highly descriptive string containing the formatted list of extracted tasks, or a confirmation that no tasks were found.
    """
    # 1. Parse the string input from the agent into a proper list of absolute paths
    file_paths = [os.path.abspath(path.strip()) for path in file_paths_str.split(",")]
    
    combined_text = ""
    for abs_path in file_paths:
        if not os.path.exists(abs_path):
            # Return an error string so the agent knows exactly which file failed
            return f"Error: The file at absolute path {abs_path} does not exist."
            
        with open(abs_path, "r", encoding="utf-8") as file:
            combined_text += f"\n--- Content from {os.path.basename(abs_path)} ---\n"
            combined_text += file.read() + "\n"

    if not combined_text.strip():
        return "Error: No valid text found in the provided file(s)."

    parser = PydanticOutputParser(pydantic_object=TaskExtraction)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert technical analyst. Extract every explicit, actionable task mentioned. Output strictly according to the provided schema.\n\n{format_instructions}"),
        ("human", "Extract the tasks from the provided text:\n\n{text}")
    ])
    
    prompt = prompt.partial(format_instructions=parser.get_format_instructions())
    extraction_chain = prompt | llm | parser
    
    try:
        result = extraction_chain.invoke({"text": combined_text})
        
        # 2. FORMAT THE OUTPUT FOR THE AGENT
        if not result.tasks:
            return "Success: Files processed, but no actionable tasks were found in the text."
            
        output_string = "Success: Extracted the following tasks:\n"
        for i, task in enumerate(result.tasks, 1):
            output_string += f"{i}. {task}\n"
            
        return output_string
        
    except Exception as e:
        return f"Error: Failed to extract tasks due to LLM exception: {str(e)}"