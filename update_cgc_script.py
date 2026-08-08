import re
import time
import json

with open("/root/flashkv0516/app/servers/cgc_api_server.py", "r") as f:
    content = f.read()

# Make sure imports are present
if "import re\n" not in content:
    content = content.replace("import json\n", "import json\nimport re\n")

def inject_logic(content):
    if "hallucinated_match3" in content:
        return content
        
    old_convert = """def convert_dsml_to_tool_call(dsml_text):
    import re
    import json
    
    # Standard format match
    invoke_match = re.search(r'<｜DSML｜invoke name="([^"]+)">(.*?)</｜DSML｜invoke>', dsml_text, re.DOTALL)
    
    # Hallucinated format match (missing invoke tag entirely, just parameters inside tool_calls)
    hallucinated_match = re.search(r'<｜DSML｜tool_calls>\s*<｜DSML｜parameter name="([^"]+)" string="true">(.*?)</｜DSML｜parameter>\s*</｜DSML｜tool_calls>', dsml_text, re.DOTALL)
    
    if invoke_match:"""
    
    new_convert = """def convert_dsml_to_tool_call(dsml_text):
    import re
    import json
    
    # Standard format match
    invoke_match = re.search(r'<｜DSML｜invoke name="([^"]+)">(.*?)</｜DSML｜invoke>', dsml_text, re.DOTALL)
    
    # Hallucinated format match 1
    hallucinated_match = re.search(r'<｜DSML｜tool_calls>\s*<｜DSML｜parameter name="([^"]+)" string="true">(.*?)</｜DSML｜parameter>\s*</｜DSML｜tool_calls>', dsml_text, re.DOTALL)
    
    # Hallucinated format match 2
    hallucinated_match2 = re.search(r'<｜DSML｜tool_calls>\s*<｜DSML｜parameter name="([^"]+)" string="true">(.*?)</｜DSML｜parameter>\s*</｜DSML｜invoke>', dsml_text, re.DOTALL)
    
    # Hallucinated format match 3
    hallucinated_match3 = re.search(r'<｜DSML｜parameter name="([^"]+)" string="true">(.*?)</｜DSML｜parameter>', dsml_text, re.DOTALL)
    
    if invoke_match:"""
    
    if old_convert in content:
        content = content.replace(old_convert, new_convert)
        
        old_return = """    elif hallucinated_match:
        command_val = hallucinated_match.group(2).strip()
        return {
            "name": "bash",
            "arguments": json.dumps({"command": command_val})
        }
    return None"""
        
        new_return = """    elif hallucinated_match:
        command_val = hallucinated_match.group(2).strip()
        return {
            "name": "bash",
            "arguments": json.dumps({"command": command_val})
        }
    elif hallucinated_match2:
        command_val = hallucinated_match2.group(2).strip()
        return {
            "name": "bash",
            "arguments": json.dumps({"command": command_val})
        }
    elif hallucinated_match3:
        command_val = hallucinated_match3.group(2).strip()
        return {
            "name": "bash",
            "arguments": json.dumps({"command": command_val})
        }
    return None"""
        
        content = content.replace(old_return, new_return)
        
    return content

new_content = inject_logic(content)
with open("/root/flashkv0516/app/servers/cgc_api_server.py", "w") as f:
    f.write(new_content)

