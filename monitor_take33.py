import time
import subprocess
import sys
import os

def clear_screen():
    os.system('clear')

print("Starting monitor for Take 33... (Press Ctrl+C to stop)")
time.sleep(1)

while True:
    try:
        clear_screen()
        print("="*60)
        print(f"🔄 Auto-refreshing Take 33 Progress | Time: {time.strftime('%H:%M:%S')}")
        print("="*60)
        
        result = subprocess.run(['python3', 'get_traj33.py'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        
        # Display only the last 60 lines to avoid terminal clutter
        output_lines = lines[-60:] if len(lines) > 60 else lines
        print('\n'.join(output_lines))
        
        time.sleep(10)
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        time.sleep(10)
