import os
import json
from openai import OpenAI

def get_openai_client():
    """Initializes and returns the OpenAI client configured for local Ollama."""
    return OpenAI(
        base_url='http://localhost:11434/v1',
        api_key='ollama'  # required but ignored by Ollama
    )

def create_plan(user_request: str) -> list[str]:
    """
    Calls the LLM to produce a JSON plan for the given user request.
    """
    client = get_openai_client()
    
    system_prompt = '''You are an agent planner. Given a user request and available tools (search_api, file_reader, file_writer, email_sender, retrieve_rag), output a JSON:
{"plan": ["step description including tool to use", ...]}'''

    user_prompt = f"User Request: {user_request}"
    
    try:
        response = client.chat.completions.create(
            model="llama3",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        plan_data = json.loads(content)
        
        if "plan" in plan_data and isinstance(plan_data["plan"], list):
            return plan_data["plan"]
        else:
            print("[Planner] Error: Expected 'plan' list in JSON.")
            return []
    except Exception as e:
        print(f"[Planner] API call failed: {e}")
        return []
