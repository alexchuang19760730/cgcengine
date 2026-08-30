#!/usr/bin/env python3
"""Deploy CGC engine to HarmonyOS NEXT phone (Mate 70 Pro) via HDC"""
import subprocess
import sys
import os

HDC = r"D:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\toolchains\hdc.exe"
DEV_ID = "3DK0224920000525"
PHONE_DIR = "/data/local/tmp/cgc"

def hdc(*args):
    cmd = [HDC, "-t", DEV_ID] + list(args)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return r.stdout + r.stderr

def send_file(local_path, remote_path):
    print(f"  Sending {os.path.basename(local_path)} -> {remote_path}")
    out = hdc("file", "send", local_path, remote_path)
    if "finish" in out.lower() or "File count" in out:
        print(f"    OK")
    else:
        print(f"    WARN: {out.strip()}")

def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    phone_bin = os.path.join(repo_root, "src", "llama.cpp", "build-harmony-phone", "bin")
    
    print("=== Deploying CGC Engine to Mate 70 Pro ===")
    print(f"Device: {DEV_ID}")
    print()
    
    # Create directory
    hdc("shell", f"mkdir -p {PHONE_DIR}/models")
    
    # Push binaries
    print("Pushing binaries...")
    for name in ["llama-simple", "llama-server", "llama-speculative-simple"]:
        local = os.path.join(phone_bin, name)
        if os.path.exists(local):
            send_file(local, f"{PHONE_DIR}/{name}")
            hdc("shell", f"chmod +x {PHONE_DIR}/{name}")
        else:
            print(f"  {name}: NOT FOUND (skipped)")
    
    # Push model
    model_dir = os.path.join(repo_root, "models", "gguf")
    model_file = "Qwen3-14B-IQ4_XS.gguf"
    model_path = os.path.join(model_dir, model_file)
    
    print()
    print("Pushing model...")
    if os.path.exists(model_path):
        send_file(model_path, f"{PHONE_DIR}/models/{model_file}")
    else:
        print(f"  {model_file}: NOT FOUND")
        print(f"  请先下载: huggingface-cli download bartowski/Qwen_Qwen3-14B-GGUF Qwen3-14B-IQ4_XS.gguf --local-dir {model_dir}")
        return
    
    # Push run script
    print()
    print("Pushing run script...")
    run_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_phone.sh")
    if os.path.exists(run_script):
        send_file(run_script, f"{PHONE_DIR}/run_phone.sh")
        hdc("shell", f"chmod +x {PHONE_DIR}/run_phone.sh")
    
    print()
    print("=== Deployment complete! ===")
    print(f"在手机上运行:")
    print(f"  hdc shell")
    print(f"  cd {PHONE_DIR}")
    print(f"  ./run_phone.sh -m models/{model_file} -n 128 -p 'Hello'")

if __name__ == "__main__":
    main()
