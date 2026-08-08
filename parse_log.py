import re
import ast

with open("local_router_cloud.log", "r") as f:
    content = f.read()

matches = list(re.finditer(r"\[DEBUG\] Sending to SGLang: (\{.*\})\n", content))
if matches:
    last_match = matches[-1]
    req_str = last_match.group(1)
    try:
        # replace boolean values to allow ast.literal_eval if needed
        req_str = req_str.replace("<class 'bool'>", "bool")
        req = ast.literal_eval(req_str)
        messages = req.get("messages", [])
        
        # Let's print the last two assistant messages
        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
        if len(assistant_msgs) >= 2:
            print("--- PREVIOUS ASSISTANT MESSAGE (STEP 4) ---")
            print(assistant_msgs[-2]["content"])
            
        if len(assistant_msgs) >= 1:
            print("--- LATEST ASSISTANT MESSAGE ---")
            print(assistant_msgs[-1]["content"])
            
    except Exception as e:
        print("Failed to parse:", e)
        # Fallback: just regex search for assistant content
        import traceback
        traceback.print_exc()
else:
    print("No matches found.")
