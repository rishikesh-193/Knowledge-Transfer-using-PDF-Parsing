import json
import os
import urllib.request
from pathlib import Path

from tests.test_ingestion import create_pdf_helper

RENDER_URL = "https://knowledge-transfer-using-pdf-parsing.onrender.com"
ORIGIN = "http://127.0.0.1:5500"

print("=" * 70)
print("FRONTEND -> LIVE RENDER BACKEND INTEGRATION TEST")
print("=" * 70)

# 1. Health Check & CORS Headers
print(f"\n--- 1. Testing GET {RENDER_URL}/health from Origin: {ORIGIN} ---")
req_health = urllib.request.Request(
    f"{RENDER_URL}/health",
    headers={"Origin": ORIGIN}
)
try:
    with urllib.request.urlopen(req_health, timeout=15) as resp:
        cors_origin = resp.headers.get("Access-Control-Allow-Origin", "None")
        body = json.loads(resp.read().decode("utf-8"))
        print(f"[OK] Health status: {resp.status}, response: {body}")
        print(f"[CORS] Access-Control-Allow-Origin: {cors_origin}")
except Exception as e:
    print(f"[FAIL] Health check error: {e}")

# 2. PDF Upload Integration
print(f"\n--- 2. Testing POST {RENDER_URL}/upload from Origin: {ORIGIN} ---")
pdf_path = Path("temp_render_integration.pdf")
create_pdf_helper(pdf_path, ["Postgres relational database and Redis caching are core components of Knowledge Transfer architecture."])

boundary = "----WebKitFormBoundaryRenderIntegration"
with open(pdf_path, "rb") as f:
    file_bytes = f.read()

body_bytes = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="file"; filename="temp_render_integration.pdf"\r\n'
    f"Content-Type: application/pdf\r\n\r\n"
).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

req_upload = urllib.request.Request(
    f"{RENDER_URL}/upload",
    data=body_bytes,
    headers={
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Origin": ORIGIN
    }
)

try:
    with urllib.request.urlopen(req_upload, timeout=25) as resp:
        upload_data = json.loads(resp.read().decode("utf-8"))
        print(f"[OK] Upload status: {resp.status}")
        print(f"     filename: {upload_data.get('filename')}")
        print(f"     pages: {upload_data.get('pages')}")
        print(f"     chunks_indexed: {upload_data.get('chunks_indexed')}")
except Exception as e:
    print(f"[FAIL] Upload error: {e}")

# 3. Answerable Question RAG Invoke
print(f"\n--- 3. Testing POST {RENDER_URL}/rag/invoke (Answerable Question) ---")
payload_answerable = json.dumps({"input": {"question": "What database is used for caching?"}}).encode("utf-8")
req_rag1 = urllib.request.Request(
    f"{RENDER_URL}/rag/invoke",
    data=payload_answerable,
    headers={"Content-Type": "application/json", "Origin": ORIGIN}
)
try:
    with urllib.request.urlopen(req_rag1, timeout=25) as resp:
        rag1_data = json.loads(resp.read().decode("utf-8"))
        output1 = rag1_data.get("output", {})
        print(f"[OK] RAG Invoke status: {resp.status}")
        print(f"     Answer: {output1.get('answer')}")
        print(f"     Sources: {output1.get('sources')}")
except Exception as e:
    print(f"[FAIL] RAG Invoke error: {e}")

# 4. Irrelevant Question RAG Invoke (Fallback Check)
print(f"\n--- 4. Testing POST {RENDER_URL}/rag/invoke (Irrelevant Question) ---")
payload_irrelevant = json.dumps({"input": {"question": "Unrelated topic quantum mechanics thermodynamics"}}).encode("utf-8")
req_rag2 = urllib.request.Request(
    f"{RENDER_URL}/rag/invoke",
    data=payload_irrelevant,
    headers={"Content-Type": "application/json", "Origin": ORIGIN}
)
try:
    with urllib.request.urlopen(req_rag2, timeout=25) as resp:
        rag2_data = json.loads(resp.read().decode("utf-8"))
        output2 = rag2_data.get("output", {})
        print(f"[OK] Irrelevant Query status: {resp.status}")
        print(f"     Answer: {output2.get('answer')}")
        print(f"     Sources: {output2.get('sources')}")
except Exception as e:
    print(f"[FAIL] Irrelevant Query error: {e}")

# 5. Endpoint Schema & Docs Availability
print(f"\n--- 5. Verifying /docs, /rag/playground/, /rag/input_schema, /rag/output_schema ---")
for ep in ["/docs", "/rag/playground/", "/rag/input_schema", "/rag/output_schema"]:
    try:
        with urllib.request.urlopen(f"{RENDER_URL}{ep}", timeout=15) as resp:
            print(f"[OK] GET {ep} status: {resp.status}")
    except Exception as e:
        print(f"[FAIL] GET {ep} error: {e}")

# Cleanup
if pdf_path.exists():
    pdf_path.unlink()

print("\n" + "=" * 70)
print("FRONTEND -> LIVE RENDER INTEGRATION TEST COMPLETE")
print("=" * 70)
