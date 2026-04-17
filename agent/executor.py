import json
from openai import OpenAI
from tools.search import search_api
from tools.file_ops import file_reader, file_writer
from tools.email import email_sender
from rag.retrieve import retrieve
from agent.planner import get_openai_client

def execute_step(step_description: str, context: str) -> str:
    """
    Executes a specific step by determining the required tool or action using an LLM.
    Returns the output of the step.
    """
    client = get_openai_client()
    
    system_prompt = '''You are an executor agent. Read the step description and the current context.
Determine which tool to execute. Output your choice in JSON format.
Tools available:
- search_api: requires "query" (string)
- file_reader: requires "path" (string)
- file_writer: requires "path" (string) and "content" (string)
- email_sender: requires "to" (string), "subject" (string), "body" (string)
- retrieve_rag: requires "query" (string) for searching local documents
- llm_action: use this when no specific external tool is needed (e.g. summarizing or reasoning). Requires "prompt" (string).

Output JSON format exactly:
{
  "tool": "tool_name",
  "kwargs": {"arg1": "val1", ...}
}
'''
    
    user_prompt = f"Step Description: {step_description}\nContext from previous steps:\n{context}"
    
    try:
        response = client.chat.completions.create(
            model="llama3",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        action_data = json.loads(response.choices[0].message.content)
        tool = action_data.get("tool")
        kwargs = action_data.get("kwargs", {})
        
        print(f"    -> [Executor] Calling {tool} with args: {kwargs}")
        
        if tool == "search_api":
            return search_api(**kwargs)
        elif tool == "file_reader":
            return file_reader(**kwargs)
        elif tool == "file_writer":
            return file_writer(**kwargs)
        elif tool == "email_sender":
            return email_sender(**kwargs)
        elif tool == "retrieve_rag":
            return "\\n".join(retrieve(**kwargs))
        elif tool == "llm_action":
            # Just directly answer using LLM
            prompt = kwargs.get("prompt", "")
            return call_llm_action(prompt, context)
        else:
            return f"Error: Unknown tool {tool}"
            
    except Exception as e:
        return f"Error executing step: {e}"

def call_llm_action(prompt: str, context: str) -> str:
    """Executes a pure LLM transformation (like summarizing)."""
    client = get_openai_client()
    
    try:
        response = client.chat.completions.create(
            model="llama3",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Context provided below."},
                {"role": "user", "content": f"Context:\n{context}\n\nTask: {prompt}"}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error in LLM action: {e}"
