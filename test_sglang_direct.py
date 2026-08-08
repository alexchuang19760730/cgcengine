#!/usr/bin/env python3
"""Test SGLang decode speed directly (bypassing proxy)"""
import urllib.request, json, time

url = 'http://127.0.0.1:30000/v1/chat/completions'
payload = json.dumps({
    'model': '/data/models/DeepSeek-V4-Flash-UD-IQ2',
    'messages': [{'role': 'user', 'content': 'Write a Python function to sort a list'}],
    'max_tokens': 50,
    'stream': True
}).encode()
req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'}, method='POST')
t0 = time.time()
first_tok = None
count = 0
tokens = []
with urllib.request.urlopen(req, timeout=120) as resp:
    for line in resp:
        line = line.decode().strip()
        if line.startswith('data:'):
            d = line[5:].strip()
            if d == '[DONE]':
                break
            try:
                obj = json.loads(d)
                ch = obj.get('choices', [{}])[0].get('delta', {}).get('content', '')
                if ch:
                    count += 1
                    tokens.append(ch)
                    if first_tok is None:
                        first_tok = time.time()
                        print(f'SGLang TTFT: {(first_tok-t0)*1000:.0f}ms')
            except:
                pass
total = time.time() - t0
if first_tok and count > 1:
    dec = time.time() - first_tok
    print(f'SGLang Decode: {count} tokens in {dec*1000:.0f}ms = {(count-1)/dec:.1f} tok/s')
print(f'SGLang Total: {total*1000:.0f}ms, {count} tokens')
print(f'Text: {" ".join(tokens)[:200]}')
