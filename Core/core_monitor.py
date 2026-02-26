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
        "rasa", "run", 
        "--enable-api", 
        "--cors", "*", 
        "--port", "5005", 
        "--interface", "0.0.0.0"
    ]
    
    # Launch Rasa as a child process, capturing stdout and stderr combined
    proc = subprocess.Popen(
        command, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True,
        bufsize=1
    )
    
    # Initialize the independent watchdog thread
    monitor_thread = threading.Thread(target=timeout_checker, args=(proc,), daemon=True)
    monitor_thread.start()

    # Iterate continuously over the child process's output
    for line in iter(proc.stdout.readline, ''):
        with state_lock:
            last_activity = time.time()
            
        # Pipe the log line to the standard Docker output for Fly.io logging
        sys.stdout.write(line)
        sys.stdout.flush()
        
    proc.wait()
    os._exit(0)

if __name__ == "__main__":
    run_server()