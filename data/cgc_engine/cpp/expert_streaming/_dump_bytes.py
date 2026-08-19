import struct

gguf_path = r"C:\Users\alexchuang\Desktop\fastprefill\gemma4_gguf\gemma-4-26B-A4B-it-heretic.IQ4_XS.gguf"

with open(gguf_path, 'rb') as f:
    # Dump bytes from 804 to 950
    f.seek(804)
    data = f.read(150)
    
    print("Raw bytes from offset 804:")
    for i in range(0, len(data), 16):
        hex_str = ' '.join(f'{b:02x}' for b in data[i:i+16])
        ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in data[i:i+16])
        print(f"  {804+i:4d}: {hex_str:<48s} {ascii_str}")