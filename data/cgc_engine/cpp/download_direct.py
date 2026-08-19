#!/usr/bin/env python3
"""Download Gemma 4 IQ4_XS using direct HTTP download with retry support."""

import os
import sys
import time
import requests

# Configuration
mirror_url = "https://hf-mirror.com"
repo_id = "mradermacher/gemma-4-26B-A4B-it-heretic-GGUF"
filename = "gemma-4-26B-A4B-it-heretic.IQ4_XS.gguf"
local_dir = r"C:\Users\alexchuang\Desktop\fastprefill\gemma4_gguf"

# Download settings
MAX_RETRIES = 100
RETRY_DELAY = 5  # seconds between retries
CONNECT_TIMEOUT = 30
READ_TIMEOUT = 300  # 5 minutes for read operations

# Construct download URL
download_url = f"{mirror_url}/{repo_id}/resolve/main/{filename}"
local_path = os.path.join(local_dir, filename)

print(f"Downloading Gemma 4 IQ4_XS model...")
print(f"  From: {download_url}")
print(f"  To: {local_path}")
print()

# Create directory if needed
os.makedirs(local_dir, exist_ok=True)

# Get total file size via HEAD request
print("  Fetching file info...")
response = requests.head(download_url, allow_redirects=True, timeout=CONNECT_TIMEOUT)
total_size = int(response.headers.get('content-length', 0))
print(f"  Total file size: {total_size:,} bytes ({total_size / 1024 / 1024 / 1024:.2f} GB)")

# Check if already partially downloaded
existing_size = 0
if os.path.exists(local_path):
    existing_size = os.path.getsize(local_path)
    print(f"  Existing file size: {existing_size:,} bytes ({existing_size / 1024 / 1024 / 1024:.2f} GB)")

# Verify if download is already complete
if total_size > 0 and existing_size >= total_size:
    print(f"\n✅ Download already complete!")
    sys.exit(0)

# Download with retry and resume support
retry_count = 0
start_time = time.time()

while retry_count < MAX_RETRIES:
    try:
        # Get current file size for resume
        current_size = existing_size
        if retry_count > 0:
            if os.path.exists(local_path):
                current_size = os.path.getsize(local_path)
            print(f"\n  [Retry {retry_count}] Resuming from byte {current_size:,}...")
            time.sleep(RETRY_DELAY)
        
        headers = {}
        mode = 'wb'
        if current_size > 0:
            headers["Range"] = f"bytes={current_size}-"
            mode = 'ab'
        
        downloaded = current_size
        
        with requests.get(download_url, headers=headers, stream=True, 
                         timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)) as response:
            response.raise_for_status()
            
            # Check if range request was honored
            if current_size > 0 and response.status_code == 200:
                # Server doesn't support range, start fresh
                print(f"  Server doesn't support range requests, starting fresh...")
                mode = 'wb'
                downloaded = 0
            
            last_update = time.time()
            
            with open(local_path, mode) as f:
                for chunk in response.iter_content(chunk_size=8192 * 1024):  # 8MB chunks
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        existing_size = downloaded  # Update for next retry
                        
                        # Update progress every 3 seconds
                        current_time = time.time()
                        if current_time - last_update >= 3:
                            elapsed = current_time - start_time
                            speed = downloaded / elapsed if elapsed > 0 else 0
                            
                            if total_size > 0:
                                progress = (downloaded / total_size) * 100
                                remaining = (total_size - downloaded) / speed if speed > 0 else 0
                                print(f"\r  Progress: {progress:.1f}% ({downloaded / 1024 / 1024 / 1024:.2f}/{total_size / 1024 / 1024 / 1024:.2f} GB) "
                                      f"Speed: {speed / 1024 / 1024:.1f} MB/s, Remaining: {remaining:.0f}s", end='', flush=True)
                            else:
                                print(f"\r  Downloaded: {downloaded / 1024 / 1024 / 1024:.2f} GB "
                                      f"Speed: {speed / 1024 / 1024:.1f} MB/s, Elapsed: {elapsed:.0f}s", end='', flush=True)
                            
                            last_update = current_time
        
        # Check if download is complete
        final_size = os.path.getsize(local_path)
        if total_size > 0 and final_size >= total_size:
            print(f"\n\n✅ Download complete!")
            print(f"  File: {local_path}")
            print(f"  Size: {final_size:,} bytes ({final_size / 1024 / 1024 / 1024:.2f} GB)")
            print(f"  Time: {time.time() - start_time:.0f} seconds")
            sys.exit(0)
        else:
            print(f"\n\n  Incomplete download (got {final_size}/{total_size}), retrying...")
            retry_count += 1
            
    except (requests.exceptions.RequestException, IOError) as e:
        retry_count += 1
        print(f"\n  ❌ Download error (attempt {retry_count}/{MAX_RETRIES}): {e}")
        
        if retry_count >= MAX_RETRIES:
            print(f"\n  Max retries reached. Partial file saved at: {local_path}")
            print(f"  Current size: {existing_size / 1024 / 1024 / 1024:.2f} GB")
            sys.exit(1)
        
        print(f"  Waiting {RETRY_DELAY} seconds before retry...")
        time.sleep(RETRY_DELAY)
