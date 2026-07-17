import subprocess
import time
import threading
import os
import sys

IDLE_TIMEOUT = 1800.0  # 30 minutes in seconds
last_activity = time.time()
state_lock = threading.Lock()

def timeout_checker(proc: subprocess.Popen):
    """Monitors the temporal gap since the last recorded activity."""
    global last_activity
    while True:
        time.sleep(10)
        with state_lock:
            elapsed = time.time() - last_activity
            
        if elapsed > IDLE_TIMEOUT:
            sys.stdout.write(f"[MONITOR] Core server idle for {IDLE_TIMEOUT} seconds. Initiating termination.\n")
            sys.stdout.flush()
            
            # Send SIGTERM to the Rasa subprocess
            proc.terminate()
            time.sleep(5)
            
            # Force container exit, signaling Fly.io to stop the machine
            os._exit(0)

def run_server():
    """Spawns the Rasa server and proxies its log output to track activity."""
    global last_activity
    
    command = [
        "stdbuf", "-oL", "-eL",  
        "rasa", "run", 
        "--enable-api", 
        "--cors", "*", 
        "--port", "5005", 
        "--interface", "0.0.0.0",
        "--debug"  # CRITICAL: Required to expose 'Received user message' in logs
    ]
    
    child_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    
    # Launch Rasa as a child process, capturing stdout and stderr combined
    proc = subprocess.Popen(
        command, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True,
        bufsize=1,
        env=child_env
    )
    
    # Initialize the independent watchdog thread
    monitor_thread = threading.Thread(target=timeout_checker, args=(proc,), daemon=True)
    monitor_thread.start()

    # Define the signifiers that are permitted to bypass the output filter.
    # Notice we removed "Logged UserUtterance" from here so it stays completely invisible.
    manifest_keywords = [
        "Rasa server is up", 
        "ERROR", 
        "WARNING"
    ]

    # Iterate continuously over the child process's output
    for line in iter(proc.stdout.readline, ''):
        
        # 1. Silently reset the kill-switch if a human actually speaks
        if "Logged UserUtterance" in line:
            with state_lock:
                last_activity = time.time()
            
        # 2. Conditionally manifest only critical startup lines or errors to stdout
        if any(keyword in line for keyword in manifest_keywords):
            sys.stdout.write(line)
            sys.stdout.flush()
        
    proc.wait()
    os._exit(0)

if __name__ == "__main__":
    run_server()