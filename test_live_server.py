import subprocess
import sys
import time
import urllib.request
import json

import os

env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"

print("Starting Uvicorn server on http://127.0.0.1:8000 ...")
proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"], env=env)

try:
    # Give server 3 seconds to boot
    time.sleep(3)
    
    # 1. Test /chain/invoke via real HTTP POST
    req = urllib.request.Request(
        "http://127.0.0.1:8000/chain/invoke",
        data=json.dumps({"input": {"text": "Local HTTP Test"}}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        print(f"[OK] Live HTTP POST /chain/invoke status: {resp.status}, response: {body}")
        
    # 2. Test /chain/playground/ via real HTTP GET
    with urllib.request.urlopen("http://127.0.0.1:8000/chain/playground/") as resp:
        print(f"[OK] Live HTTP GET /chain/playground/ status: {resp.status}")

finally:
    proc.terminate()
    proc.wait()
    print("Uvicorn server terminated cleanly.")
