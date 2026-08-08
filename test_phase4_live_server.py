import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from tests.test_ingestion import create_pdf_helper

print("=" * 60)
print("PHASE 4: LIVE UVICORN SERVER SMOKE TEST")
print("=" * 60)

env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"

print("\n1. Starting Uvicorn server on http://127.0.0.1:8000 ...")
proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "server:app", "--host", "127.0.0.1", "--port", "8000"],
    env=env
)

try:
    # Give server 3.5 seconds to start up
    time.sleep(3.5)

    # A. Test GET /health
    print("\n--- A. Testing GET http://127.0.0.1:8000/health ---")
    with urllib.request.urlopen("http://127.0.0.1:8000/health") as resp:
        body = json.loads(resp.read().decode("utf-8"))
        print(f"[OK] GET /health status: {resp.status}, response: {body}")

    # B. Test POST /upload
    print("\n--- B. Testing POST http://127.0.0.1:8000/upload ---")
    tmp_pdf = Path("temp_live_test.pdf")
    create_pdf_helper(tmp_pdf, ["Live server PDF ingestion test page with unique content."])
    
    # Construct multipart form-data payload manually
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    with open(tmp_pdf, "rb") as f:
        file_bytes = f.read()

    body_bytes = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="temp_live_test.pdf"\r\n'
        f"Content-Type: application/pdf\r\n\r\n"
    ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req_upload = urllib.request.Request(
        "http://127.0.0.1:8000/upload",
        data=body_bytes,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    with urllib.request.urlopen(req_upload) as resp:
        upload_resp = json.loads(resp.read().decode("utf-8"))
        print(f"[OK] POST /upload status: {resp.status}, response: {upload_resp}")

    # C. Test POST /rag/invoke
    print("\n--- C. Testing POST http://127.0.0.1:8000/rag/invoke ---")
    invoke_payload = json.dumps({"input": {"question": "unique content"}}).encode("utf-8")
    req_invoke = urllib.request.Request(
        "http://127.0.0.1:8000/rag/invoke",
        data=invoke_payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req_invoke) as resp:
        invoke_resp = json.loads(resp.read().decode("utf-8"))
        print(f"[OK] POST /rag/invoke status: {resp.status}, response output: {invoke_resp.get('output')}")

    # D. Test POST /rag/stream
    print("\n--- D. Testing POST http://127.0.0.1:8000/rag/stream ---")
    req_stream = urllib.request.Request(
        "http://127.0.0.1:8000/rag/stream",
        data=invoke_payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req_stream) as resp:
        stream_data = resp.read().decode("utf-8")
        print(f"[OK] POST /rag/stream status: {resp.status}, bytes received: {len(stream_data)}")

    # E. Test GET /rag/playground/
    print("\n--- E. Testing GET http://127.0.0.1:8000/rag/playground/ ---")
    with urllib.request.urlopen("http://127.0.0.1:8000/rag/playground/") as resp:
        print(f"[OK] GET /rag/playground/ status: {resp.status} (LangServe UI Playground loaded)")

    # F. Test GET /docs
    print("\n--- F. Testing GET http://127.0.0.1:8000/docs ---")
    with urllib.request.urlopen("http://127.0.0.1:8000/docs") as resp:
        print(f"[OK] GET /docs status: {resp.status} (FastAPI OpenAPI Docs loaded)")

finally:
    if tmp_pdf.exists():
        tmp_pdf.unlink()
    proc.terminate()
    proc.wait()
    print("\nUvicorn server terminated cleanly.")

print("=" * 60)
print("PHASE 4 LIVE SERVER TEST COMPLETED SUCCESSFULLY")
print("=" * 60)
