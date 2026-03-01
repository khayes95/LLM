#!/usr/bin/env python3
"""Check if vLLM server is ready and run a quick test completion.

Usage:
    python scripts/smoke_test_qwen35_server.py
    python scripts/smoke_test_qwen35_server.py --wait_timeout 600
    python scripts/smoke_test_qwen35_server.py --base_url http://localhost:8100/v1
"""
import argparse
import json
import sys
import time
import urllib.request
import urllib.error


def wait_for_server(base_url: str, timeout: int) -> str:
    """Poll /v1/models until the server is ready. Returns model name or empty string."""
    models_url = f"{base_url}/models"
    start = time.time()
    attempt = 0
    while time.time() - start < timeout:
        attempt += 1
        try:
            req = urllib.request.Request(models_url)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                models = data.get("data", [])
                if models:
                    model_id = models[0]["id"]
                    print(f"Server ready after {time.time() - start:.0f}s "
                          f"(attempt {attempt}): {[m['id'] for m in models]}")
                    return model_id
        except (urllib.error.URLError, ConnectionError, OSError, TimeoutError):
            pass
        if attempt % 6 == 0:
            elapsed = time.time() - start
            print(f"  Waiting for server... {elapsed:.0f}s / {timeout}s")
        time.sleep(10)
    print(f"Server not ready after {timeout}s")
    return ""


def test_completion(base_url: str, model_name: str) -> bool:
    """Run a single test completion."""
    url = f"{base_url}/chat/completions"
    payload = json.dumps({
        "model": model_name,
        "messages": [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
        "max_tokens": 32,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(url, data=payload,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            text = data["choices"][0]["message"]["content"]
            print(f"Test completion (model={model_name}): '{text.strip()}'")
            return True
    except Exception as e:
        print(f"Test completion failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url", default="http://localhost:8100/v1")
    parser.add_argument("--wait_timeout", type=int, default=600)
    args = parser.parse_args()

    model_name = wait_for_server(args.base_url, args.wait_timeout)
    if not model_name:
        sys.exit(1)

    if not test_completion(args.base_url, model_name):
        sys.exit(1)

    print("Server smoke test passed!")


if __name__ == "__main__":
    main()
