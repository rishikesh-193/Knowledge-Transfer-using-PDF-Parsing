import sys
import os
import subprocess
from dotenv import load_dotenv

load_dotenv()

failures = []

print("=" * 60)
print("PHASE 0: VERIFICATION & COMPATIBILITY SUITE")
print("=" * 60)

# Model configuration from environment variables with defaults
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")

# Step 1: Verify imports
modules_to_test = [
    ("langchain", "langchain"),
    ("langchain_core", "langchain-core"),
    ("langchain_google_genai", "langchain-google-genai"),
    ("langchain_chroma", "langchain-chroma"),
    ("langserve", "langserve"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("pydantic", "pydantic"),
    ("chromadb", "chromadb"),
    ("pypdf", "pypdf"),
    ("dotenv", "python-dotenv"),
    ("sse_starlette", "sse-starlette")
]

print("\n--- 1. Testing Module Imports ---")
for mod_name, pkg_name in modules_to_test:
    try:
        mod = __import__(mod_name)
        version = getattr(mod, "__version__", "N/A")
        print(f"[OK] {pkg_name} ({mod_name}) imported successfully. Version: {version}")
    except Exception as e:
        err_msg = f"Import failed for package '{pkg_name}' ({mod_name}): {e}"
        print(f"[FAIL] {err_msg}")
        failures.append(err_msg)

# Step 2: Run pip check
print("\n--- 2. Running pip check ---")
pip_check_res = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
if pip_check_res.returncode == 0:
    print("[OK] pip check passed with 0 errors:")
    print(pip_check_res.stdout.strip())
else:
    err_msg = f"pip check failed: {pip_check_res.stdout.strip() or pip_check_res.stderr.strip()}"
    print(f"[FAIL] {err_msg}")
    failures.append(err_msg)

# Step 3: Test Gemini Chat
print(f"\n--- 3. Testing Gemini Chat ({GEMINI_MODEL}) ---")
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    err_msg = "GOOGLE_API_KEY environment variable is missing or empty"
    print(f"[FAIL] {err_msg}")
    failures.append(err_msg)
else:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=api_key)
        res = llm.invoke("Hello, answer with 'OK'")
        print(f"[OK] ChatGoogleGenerativeAI ({GEMINI_MODEL}) response: {res.content.strip()}")
    except Exception as e:
        err_msg = f"Gemini Chat ({GEMINI_MODEL}) failed: {e}"
        print(f"[FAIL] {err_msg}")
        failures.append(err_msg)

# Step 4: Test Gemini Embeddings
print(f"\n--- 4. Testing Gemini Embeddings ({EMBEDDING_MODEL}) ---")
if not api_key:
    # Already recorded missing GOOGLE_API_KEY above
    print(f"[FAIL] Skipping embedding test because GOOGLE_API_KEY is not set.")
else:
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=api_key)
        vector = embeddings.embed_query("Test vector embedding generation")
        print(f"[OK] GoogleGenerativeAIEmbeddings ({EMBEDDING_MODEL}) returned vector of length: {len(vector)}")
    except Exception as e:
        err_msg = f"Gemini Embeddings ({EMBEDDING_MODEL}) failed: {e}"
        print(f"[FAIL] {err_msg}")
        failures.append(err_msg)

# Step 5: Test LangServe Runnable endpoint (/invoke and /playground/)
print("\n--- 5. Testing LangServe Runnable Routes (/chain/invoke and /chain/playground/) ---")
try:
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    
    # Test /chain/invoke
    invoke_resp = client.post("/chain/invoke", json={"input": {"text": "Phase 0 Verification"}})
    if invoke_resp.status_code == 200:
        print(f"[OK] /chain/invoke status: 200, response: {invoke_resp.json()}")
    else:
        err_msg = f"LangServe /chain/invoke returned HTTP {invoke_resp.status_code}: {invoke_resp.text}"
        print(f"[FAIL] {err_msg}")
        failures.append(err_msg)

    # Test /chain/playground/
    playground_resp = client.get("/chain/playground/")
    if playground_resp.status_code == 200:
        print(f"[OK] /chain/playground/ status: 200 (UI Playground loaded successfully)")
    else:
        err_msg = f"LangServe /chain/playground/ returned HTTP {playground_resp.status_code}"
        print(f"[FAIL] {err_msg}")
        failures.append(err_msg)

except Exception as e:
    err_msg = f"LangServe TestClient check failed with exception: {e}"
    print(f"[FAIL] {err_msg}")
    failures.append(err_msg)

# Final summary & Exit status
print("\n" + "=" * 60)
if failures:
    print("PHASE 0: BLOCKED")
    print("\nRecorded Failures:")
    for idx, failure in enumerate(failures, start=1):
        print(f"  {idx}. {failure}")
    print("=" * 60)
    sys.exit(1)
else:
    print("PHASE 0: GREEN")
    print("=" * 60)
    sys.exit(0)
