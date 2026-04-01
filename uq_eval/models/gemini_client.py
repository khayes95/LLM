from __future__ import annotations

import fcntl
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

import urllib.request
import urllib.error

from ..types import ModelRequest, ModelResponse
from .base import BaseModelClient


_COUNTER_PATH = "/scratch/khayes/LLM/data/gemini_daily_counter.json"

_DEFAULT_MAX_DAILY = 250

# Vertex AI credentials path
_GCP_CREDENTIALS_PATH = "/scratch/khayes/LLM/data/credentials/gcp_adc.json"

# Vertex AI project/location
_GCP_PROJECT = "finegrainv1"
_GCP_LOCATION = "global"


def _parse_data_url(data_url: str) -> tuple[str, str]:
    """Parse a data URL into (mime_type, base64_data)."""
    match = re.match(r"data:([^;]+);base64,(.+)", data_url, re.DOTALL)
    if match:
        return match.group(1), match.group(2)
    return "", ""


def _convert_content_to_gemini(content: Any) -> list[dict[str, Any]]:
    """Convert OpenAI-style content (string or list) to Gemini parts."""
    if isinstance(content, str):
        return [{"text": content}]

    if not isinstance(content, list):
        return [{"text": str(content)}]

    parts: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            parts.append({"text": str(item)})
            continue

        item_type = item.get("type", "")

        if item_type in ("text", "input_text"):
            parts.append({"text": item.get("text", "")})

        elif item_type == "input_image":
            data_url = item.get("image_url", "")
            mime_type, b64_data = _parse_data_url(data_url)
            if b64_data:
                parts.append({
                    "inline_data": {"mime_type": mime_type, "data": b64_data}
                })

        elif item_type == "image_url":
            url_obj = item.get("image_url", {})
            url = url_obj.get("url", "") if isinstance(url_obj, dict) else str(url_obj)
            mime_type, b64_data = _parse_data_url(url)
            if b64_data:
                parts.append({
                    "inline_data": {"mime_type": mime_type, "data": b64_data}
                })

        else:
            text = item.get("text", "") or str(item)
            parts.append({"text": text})

    return parts


def _extract_system_and_contents(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Separate system/developer messages and convert the rest to Gemini format."""
    system_parts: list[str] = []
    gemini_contents: list[dict[str, Any]] = []

    role_map = {"assistant": "model", "user": "user"}

    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")

        if role in ("system", "developer"):
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") in ("text", "input_text"):
                        system_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        system_parts.append(part)
            continue

        gemini_role = role_map.get(role, "user")
        parts = _convert_content_to_gemini(content)
        gemini_contents.append({"role": gemini_role, "parts": parts})

    # Merge consecutive same-role messages (Gemini requires alternating roles)
    merged: list[dict[str, Any]] = []
    for msg in gemini_contents:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["parts"].extend(msg["parts"])
        else:
            merged.append(msg)

    if merged and merged[0]["role"] == "model":
        merged.insert(0, {"role": "user", "parts": [{"text": "Please continue."}]})

    system_text = "\n\n".join(system_parts) if system_parts else None
    return system_text, merged


class DailyLimitReached(Exception):
    pass


def _read_counter(path: str) -> dict[str, Any]:
    today = date.today().isoformat()
    default = {"date": today, "count": 0, "max_daily": _DEFAULT_MAX_DAILY}

    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return default

    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return default

    if data.get("date") != today:
        data["date"] = today
        data["count"] = 0
        if "max_daily" not in data:
            data["max_daily"] = _DEFAULT_MAX_DAILY

    return data


def _write_counter(path: str, data: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _check_and_increment(path: str) -> tuple[bool, int, int]:
    """Atomically check the daily limit and increment if allowed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    lock_path = path + ".lock"
    lock_fd = open(lock_path, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)

        data = _read_counter(path)
        max_daily = data.get("max_daily", _DEFAULT_MAX_DAILY)
        count = data.get("count", 0)

        if count >= max_daily:
            return False, count, max_daily

        data["count"] = count + 1
        _write_counter(path, data)
        return True, count + 1, max_daily

    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


def _get_vertex_access_token(credentials_path: str) -> str:
    """Get a fresh access token using the refresh token from ADC credentials."""
    with open(credentials_path) as f:
        creds = json.load(f)

    payload = json.dumps({
        "client_id": creds["client_id"],
        "client_secret": creds["client_secret"],
        "refresh_token": creds["refresh_token"],
        "grant_type": "refresh_token",
    }).encode()

    req = urllib.request.Request(
        "https://oauth2.googleapis.com/token",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        token_data = json.loads(resp.read())
    return token_data["access_token"]


@dataclass(slots=True)
class GeminiClient(BaseModelClient):
    """Google Gemini backend supporting both AI Studio and Vertex AI.

    Modes:
      - AI Studio (default): Uses GOOGLE_API_KEY env var
      - Vertex AI: Set use_vertex=True, uses GCP credentials for auth.
        Required for Gemini 3.x Pro models.

    Hard daily limit enforced via a counter file to prevent runaway spending.
    """

    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None  # accepted for CLI compatibility
    timeout_s: float = 600.0
    max_retries: int = 5
    retry_sleep_s: float = 2.0
    rpm: int = 10  # requests per minute
    counter_path: str = _COUNTER_PATH
    max_daily: Optional[int] = None
    use_vertex: bool = False
    gcp_credentials_path: str = _GCP_CREDENTIALS_PATH
    gcp_project: str = _GCP_PROJECT
    gcp_location: str = _GCP_LOCATION
    thinking_level: Optional[str] = None  # "minimal", "low", "medium", "high"

    def __post_init__(self) -> None:
        # Determine mode
        self._use_vertex = self.use_vertex or os.environ.get("GEMINI_USE_VERTEX", "").lower() in ("1", "true", "yes")

        if self._use_vertex:
            # Vertex AI mode — use OAuth
            self._api_key = None
            self._access_token: Optional[str] = None
            self._token_expiry: float = 0.0
            creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", self.gcp_credentials_path)
            self._credentials_path = creds_path
            if not os.path.exists(creds_path):
                raise ValueError(f"GCP credentials not found at {creds_path}")
        else:
            # AI Studio mode — use API key
            self._api_key = self.api_key or os.environ.get("GOOGLE_API_KEY")
            if not self._api_key:
                raise ValueError(
                    "Google API key required. Pass api_key= or set GOOGLE_API_KEY env var."
                )

        self._min_interval = 60.0 / self.rpm if self.rpm > 0 else 0.0
        self._last_request_time = 0.0

        # Set max_daily in counter file if overridden
        if self.max_daily is not None:
            os.makedirs(os.path.dirname(self.counter_path), exist_ok=True)
            lock_path = self.counter_path + ".lock"
            lock_fd = open(lock_path, "w")
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                data = _read_counter(self.counter_path)
                data["max_daily"] = self.max_daily
                _write_counter(self.counter_path, data)
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()

    def _get_access_token(self) -> str:
        """Get a valid access token, refreshing if needed."""
        if self._access_token and time.time() < self._token_expiry:
            return self._access_token
        self._access_token = _get_vertex_access_token(self._credentials_path)
        self._token_expiry = time.time() + 3300  # tokens last ~1 hour, refresh at 55 min
        return self._access_token

    def _build_endpoint_and_headers(self) -> tuple[str, dict[str, str]]:
        """Build the API endpoint URL and headers based on mode."""
        if self._use_vertex:
            url = (
                f"https://aiplatform.googleapis.com/v1/"
                f"projects/{self.gcp_project}/locations/{self.gcp_location}/"
                f"publishers/google/models/{self.model_name}:generateContent"
            )
            token = self._get_access_token()
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            }
        else:
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/"
                f"models/{self.model_name}:generateContent?key={self._api_key}"
            )
            headers = {"Content-Type": "application/json"}

        return url, headers

    def _rate_limit_wait(self) -> None:
        if self._min_interval <= 0:
            return
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

    def generate(self, req: ModelRequest) -> ModelResponse:
        # --- DAILY LIMIT CHECK ---
        allowed, count, max_daily = _check_and_increment(self.counter_path)
        if not allowed:
            msg = (
                f"DAILY LIMIT REACHED: {count}/{max_daily} requests used today. "
                f"Refusing to make API call. Adjust max_daily in {self.counter_path} "
                f"or wait until tomorrow."
            )
            print(f"[gemini] {msg}")
            return ModelResponse(
                text=f"DAILY_LIMIT_REACHED: {count}/{max_daily}",
                raw={"error": msg, "daily_count": count, "max_daily": max_daily},
                usage=None,
            )

        # Build the request body
        system_text, contents = _extract_system_and_contents(req.messages)

        body: dict[str, Any] = {"contents": contents}

        if system_text:
            body["system_instruction"] = {
                "parts": [{"text": system_text}]
            }

        # Generation config
        gen_config: dict[str, Any] = {}
        if req.max_output_tokens:
            gen_config["maxOutputTokens"] = req.max_output_tokens
        if req.temperature is not None:
            gen_config["temperature"] = float(req.temperature)
        if gen_config:
            body["generationConfig"] = gen_config

        # Thinking config
        thinking_level = self.thinking_level or os.environ.get("GEMINI_THINKING_LEVEL")
        if thinking_level:
            body["generationConfig"] = body.get("generationConfig", {})
            body["generationConfig"]["thinkingConfig"] = {
                "thinkingLevel": thinking_level.upper()
            }

        # Build URL and headers
        endpoint, headers = self._build_endpoint_and_headers()
        payload = json.dumps(body).encode("utf-8")

        last_err: str | None = None
        for attempt in range(self.max_retries):
            try:
                self._rate_limit_wait()

                http_req = urllib.request.Request(
                    endpoint, data=payload, headers=headers, method="POST",
                )

                with urllib.request.urlopen(
                    http_req, timeout=self.timeout_s
                ) as http_resp:
                    resp_data = json.loads(http_resp.read().decode("utf-8"))

                self._last_request_time = time.time()

                # Extract response text (skip thought signature parts)
                text = None
                try:
                    parts = resp_data["candidates"][0]["content"]["parts"]
                    for part in parts:
                        if "text" in part and "thoughtSignature" not in part:
                            text = part["text"]
                            break
                    if text is None:
                        # Fall back to first text part
                        for part in parts:
                            if "text" in part:
                                text = part["text"]
                                break
                except (KeyError, IndexError, TypeError):
                    pass

                if text is None:
                    finish_reason = "unknown"
                    try:
                        finish_reason = resp_data["candidates"][0].get(
                            "finishReason", "unknown"
                        )
                    except (KeyError, IndexError, TypeError):
                        pass
                    text = f"GEMINI_NO_TEXT: finishReason={finish_reason}"

                # Extract usage
                usage = None
                usage_meta = resp_data.get("usageMetadata")
                if usage_meta:
                    input_tokens = usage_meta.get("promptTokenCount", 0)
                    output_tokens = usage_meta.get("candidatesTokenCount", 0)
                    thinking_tokens = usage_meta.get("thoughtsTokenCount", 0)
                    usage = {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "thinking_tokens": thinking_tokens,
                        "total_tokens": input_tokens + output_tokens + thinking_tokens,
                    }

                return ModelResponse(text=text, raw=resp_data, usage=usage)

            except urllib.error.HTTPError as e:
                status = e.code
                err_body = ""
                try:
                    err_body = e.read().decode("utf-8", errors="replace")[:500]
                except Exception:
                    pass
                last_err = f"HTTPError({status}): {err_body}"

                if status == 429:
                    wait = self.retry_sleep_s * (2 ** attempt)
                    print(
                        f"[gemini] rate limited (429), waiting {wait:.0f}s "
                        f"(attempt {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(wait)
                elif status == 401:
                    # Auth expired — refresh token and retry
                    if self._use_vertex:
                        print("[gemini] token expired, refreshing...")
                        self._token_expiry = 0
                        endpoint, headers = self._build_endpoint_and_headers()
                        continue
                    break
                elif status >= 500:
                    wait = self.retry_sleep_s * (2 ** attempt)
                    print(f"[gemini] server error {status}, retrying in {wait:.0f}s")
                    time.sleep(wait)
                elif status == 400:
                    print(f"[gemini] bad request (400), not retrying: {err_body[:200]}")
                    break
                else:
                    print(f"[gemini] HTTP {status}, not retrying: {err_body[:200]}")
                    break

            except urllib.error.URLError as e:
                last_err = f"URLError: {e.reason}"
                wait = self.retry_sleep_s * (2 ** attempt)
                print(f"[gemini] connection error, retrying in {wait:.0f}s")
                time.sleep(wait)

            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                break

        return ModelResponse(
            text=f"API_FAILED after {self.max_retries} tries: {last_err}",
            raw={"error": last_err},
            usage=None,
        )
