import sys
from gguf import GGUFReader

gguf_path = r"C:\Users\alexchuang\Desktop\fastprefill\gemma4_gguf\gemma-4-26B-A4B-it-heretic.IQ4_XS.gguf"

reader = GGUFReader(gguf_path)

# Find general.tags
if "general.tags" in reader.fields:
    field = reader.fields["general.tags"]
    print(f"general.tags type: {type(field)}")
    val = field.contents()
    print(f"general.tags value: {val}")
    print(f"general.tags length: {len(val)}")

# Also check general.tags position in the file
print("\nChecking field positions...")
for key in ["general.tags", "gemma4.block_count", "general.base_model.0.repo_url"]:
    if key in reader.fields:
        field = reader.fields[key]
        print(f"{key}: offset={field.offset}")