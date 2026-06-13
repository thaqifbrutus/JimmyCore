from google import genai
from google.genai import types
import json
from app.config import AI_API_KEY

# Initialize the Gemini client at module level
client = genai.Client(api_key=AI_API_KEY)

# System Prompt that defines the AI's role and behaviour for Jimmy.
SYSTEM_PROMPT = """

You are JimmyCore AI, an expert data analyst and data quality specialist 
embedded inside an AI-powered data quality platform.

Your job is to analyse data profiling results and produce clear, 
actionable, professional reports that help technical and non-technical 
users understand the quality and content of their datasets.

When analysing a dataset profile, you always:
1. Explain what the dataset appears to contain in plain English
2. Summarise the overall data quality in one clear sentence
3. List and explain every quality issue found, ordered by severity
4. Provide specific, actionable recommendations for each issue
5. Give a final verdict on whether this data is ready to use

Your tone is professional but conversational. You write for a mixed 
audience — some readers are developers, some are business analysts, 
some are project managers. Avoid jargon where possible. When you must 
use technical terms, briefly explain them.

Always be honest about data quality. Do not sugarcoat critical issues.
If data is not ready for production use, say so clearly and explain why.

"""

# Function to generate dataset summary
def generate_dataset_summary(profile_data: dict, original_filename: str) -> str:
    """ Takes raw profiling results and asks Gemini to produce a plain-English summary with actionable recommendations."""

    user_prompt = f"""

Please analyse the following data profiling results for a dataset called "{original_filename}" and produce a comprehensive data quality report.

--- PROFILING RESULTS ---
{json.dumps(profile_data, indent=2)}
--- END OF PROFILING RESULTS ---

Your report should include:

1. DATASET OVERVIEW
   What does this dataset appear to contain? 
   How large is it? What are the key columns?

2. OVERALL QUALITY ASSESSMENT
   One clear sentence summarising the quality of this data.

3. ISSUES FOUND
   For each issue detected, explain:
   - What the issue is in plain English
   - Why it matters (what could go wrong if ignored)
   - What should be done to fix it

4. COLUMN HIGHLIGHTS
   Call out any columns that are particularly interesting,
   problematic, or worth noting.

5. RECOMMENDATION
   Is this data ready to use? If yes, with what caveats?
   If no, what needs to happen before it can be used?

Be specific. Reference actual column names and numbers from the profile.
"""
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=1500,
            temperature=0.3
        )
    )

    return response.text

#Function to generate technical context
def generate_technical_context(profile_data: dict, original_filename: str) -> str:
    """Generates technical context for developers and database engineers"""

    user_prompt = f"""

You are reviewing profiling results for "{original_filename}" to prepare technical context for a development team about to work with this data.

--- PROFILING RESULTS ---
{json.dumps(profile_data, indent=2)}
--- END OF PROFILING RESULTS ---

Produce a technical brief that includes:

1. SUGGESTED DATABASE SCHEMA
   Based on the columns and their detected types, suggest appropriate 
   SQL column types (VARCHAR, INTEGER, DATE, BOOLEAN etc.)
   Flag any columns where the detected type differs from what it should be.

2. DATA TRANSFORMATION STEPS
   List the specific transformations needed before this data can be 
   safely inserted into a production database.
   Be specific — name the columns, describe the transformation.

3. VALIDATION RULES
   Suggest validation rules that should be enforced on each column
   (NOT NULL constraints, length limits, format checks, foreign key candidates)

4. RISKS AND WARNINGS
   What could go wrong during import or integration?
   What assumptions are being made about this data?

5. ESTIMATED EFFORT
   Based on the issues found, estimate the cleanup effort:
   Low (minor formatting fixes), Medium (significant transformations needed),
   or High (fundamental data quality problems to resolve first)

Be specific and technical. This output will be used to create development tickets and database migration scripts.
"""
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=1500,
            temperature=0.3
        )
    )

    return response.text

# Function to answer user inputs regarding dataset
def answer_dataset_question(
        profile_data: dict,
        original_filename: str,
        question: str,
        conversation_history: list = None
) -> str:
    """ 
    Allows a user to ask any free form question about their dataset.
    Supports multi-turn conversation via conversation_history.

    conversation_history format:
    [{"role": "user", "content": "..."}, {"role": "model", "content": "..."}]

    Note: Gemini uses "model" instead of "assistant" for the AI role.
    
    """

    # Context anchor
    context_message = f"""

The user is asking questions about a dataset called "{original_filename}".
Here are the profiling results for full context:

{json.dumps(profile_data, indent=2)}

Answwer the user's questions based on this profiling data.
If the answer cannot be determined from the profiling data alone, say so clearly.
"""
    
    # Contents structure for LLM (Gemini)
    contents = [
        {

            "role": "user",
            "parts": [{"text": context_message}]

        },
        {
            "role": "model",
            "parts": [{"text": "Understood. I have reviewed the profiling results and I am ready to answer questions about this dataset."}]
        }
    
    ]

    # Append prior conversation turns if they exist

    if conversation_history:
        for turn in conversation_history:
            contents.append({
                "role": "model" if turn["role"] == "assistant" else turn["role"],
                "parts": [{"text": turn["content"]}]
            })

    # Append the current question
    contents.append({
        "role": "user",
        "parts": [{"text": question}]
    })

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=1000,
            temperature=0.3
        )
    )

    return response.text
