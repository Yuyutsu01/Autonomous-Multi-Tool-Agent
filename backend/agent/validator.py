from agent.telemetry import telemetry

def validate_step(step_description: str, output: str) -> tuple[bool, str]:
    """
    Validates if the output is valid for the given step.
    Checks if output is non-empty and logically complete.
    """
    if not output:
        telemetry.record_metric("Validation Rate", 0, success=False)
        return False, "Output is empty."
        
    output_str = str(output).strip()
    
    if len(output_str) == 0:
        telemetry.record_metric("Validation Rate", 0, success=False)
        return False, "Output string is empty."
        
    # Example heuristic: if the step asks to "summarise" or "explain", expect more detail.
    if "summarise" in step_description.lower() or "summarize" in step_description.lower():
        if len(output_str) < 20:
             telemetry.record_metric("Validation Rate", 0, success=False)
             return False, "Summary is too short (under 20 characters)."
             
    # If the tool returned an error string indicating file not found, etc.
    if output_str.startswith("Error:"):
        telemetry.record_metric("Validation Rate", 0, success=False)
        return False, f"Tool execution returned an error: {output_str}"

    telemetry.record_metric("Validation Rate", 0, success=True)
    return True, "Valid"
