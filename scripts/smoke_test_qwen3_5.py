"""Smoke test for Qwen3.5-397B-A17B-FP8 served via vLLM."""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error


def wait_for_server(base_url: str, timeout: int = 600, interval: int = 10):
    """Poll the server until it's ready or timeout."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(f"{base_url}/v1/models")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                models = [m["id"] for m in data.get("data", [])]
                print(f"Server ready! Models: {models}")
                return True
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            elapsed = int(time.time() - start)
            print(f"  Waiting for server... ({elapsed}s / {timeout}s)")
            time.sleep(interval)
    print(f"ERROR: Server not ready after {timeout}s")
    return False


def test_completion(base_url: str):
    """Send a simple completion request."""
    payload = json.dumps({
        "model": "/scratch/khayes/.cache/huggingface/hub/Qwen3.5-397B-A17B-FP8",
        "messages": [
            {"role": "user", "content": "What is 2 + 2? Answer with just the number."}
        ],
        "max_tokens": 256,
        "temperature": 0.7,
    }).encode()

    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    print("\nSending test prompt: 'What is 2 + 2?'")
    start = time.time()
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
    elapsed = time.time() - start

    content = result["choices"][0]["message"]["content"]
    usage = result.get("usage", {})
    print(f"Response: {content}")
    print(f"Tokens: prompt={usage.get('prompt_tokens')}, "
          f"completion={usage.get('completion_tokens')}")
    print(f"Latency: {elapsed:.2f}s")
    return result


def test_longer_completion(base_url: str):
    """Send a harder prompt to test generation quality."""
    payload = json.dumps({
        "model": "/scratch/khayes/.cache/huggingface/hub/Qwen3.5-397B-A17B-FP8",
        "messages": [
            {"role": "user",
             "content": "Explain the concept of uncertainty quantification "
                        "in machine learning in 2-3 sentences."}
        ],
        "max_tokens": 512,
        "temperature": 0.7,
    }).encode()

    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    print("\nSending longer prompt about UQ...")
    start = time.time()
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
    elapsed = time.time() - start

    content = result["choices"][0]["message"]["content"]
    usage = result.get("usage", {})
    print(f"Response: {content}")
    print(f"Tokens: prompt={usage.get('prompt_tokens')}, "
          f"completion={usage.get('completion_tokens')}")
    print(f"Latency: {elapsed:.2f}s")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url", default="http://localhost:8100")
    parser.add_argument("--wait_timeout", type=int, default=600,
                        help="Seconds to wait for server to be ready")
    args = parser.parse_args()

    print(f"Smoke test for Qwen3.5-397B-A17B-FP8")
    print(f"Server: {args.base_url}")
    print("=" * 50)

    if not wait_for_server(args.base_url, timeout=args.wait_timeout):
        sys.exit(1)

    try:
        test_completion(args.base_url)
        test_longer_completion(args.base_url)
        print("\n" + "=" * 50)
        print("SMOKE TEST PASSED")
        print("=" * 50)
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
