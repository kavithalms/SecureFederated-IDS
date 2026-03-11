import subprocess
import time

NUM_CLIENTS = 5

processes = []

for i in range(NUM_CLIENTS):
    print(f"Starting client {i}")
    p = subprocess.Popen(["python", "client.py", str(i)])
    processes.append(p)
    time.sleep(1)  # small delay

for p in processes:
    p.wait()
