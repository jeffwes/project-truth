"""Small helper for calling Gemini v1beta :generateContent endpoint.

Usage:
  from gemini.client import call_gemini
  res = call_gemini("Hello world")

The function returns a dict with keys:
  - `ok`: bool
  - `text`: extracted assistant text when available (or None)
  - `raw`: full parsed JSON response when available
  - `status_code`: HTTP status code when request completed
  - `error`: error message on failure
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests


def _parse_response_for_text(data: Any) -> Optional[str]:
    """Best-effort extract assistant text from various response shapes."""
    text = None
    try:
        if isinstance(data, dict) and "candidates" in data:
            c = data.get("candidates")
            if isinstance(c, list) and c:
                first = c[0]
                if isinstance(first, dict):
                    cont = first.get("content")
                    if isinstance(cont, dict):
                        parts = cont.get("parts") or []
                        if isinstance(parts, list) and parts:
                            p0 = parts[0]
                            if isinstance(p0, dict):
                                text = p0.get("text")
        if not text and isinstance(data, dict) and "output" in data:
            out = data.get("output")
            text = str(out)[:200]
        if not text and isinstance(data, dict) and "choices" in data:
            ch = data.get("choices")
            if isinstance(ch, list) and ch:
                maybe = ch[0]
                if isinstance(maybe, dict):
                    text = maybe.get("text") or maybe.get("message") or str(maybe)
    except Exception:
        text = None
    return text


def call_gemini(
    prompt: str,
    model: str = "gemini-3-pro-preview",
    api_key_env: str = "GEMINI_API_KEY",
    timeout: int = 15,
    return_raw: bool = False,
    enable_search_tool: bool = False,
    response_schema: Optional[dict] = None,
) -> Dict[str, Any]:
    """Call Gemini, preferring the `google.generativeai` SDK when available.

    - If the `google.generativeai` package is installed, use it and attempt to
      enable the Search Tool when `enable_search_tool=True` (best-effort).
    - Otherwise fall back to the REST v1beta `:generateContent` request.

    Returns a dict: {ok, text, raw, status_code, error}
    """
    api_key = os.environ.get(api_key_env)
    if not api_key:
        return {"ok": False, "error": f"env var {api_key_env} not set", "text": None, "raw": None, "status_code": None}

    # Try SDK first
    try:
        import google.generativeai as genai  # type: ignore
    except Exception:
        genai = None

    if genai is not None:
        try:
            # configure SDK if available
            if hasattr(genai, "configure"):
                try:
                    genai.configure(api_key=api_key)
                except Exception:
                    # some SDK versions use a different configure name/signature
                    pass

            # Prefer the GenerativeModel interface exposed by the SDK
            resp = None
            if hasattr(genai, "GenerativeModel"):
                try:
                    model_obj = genai.GenerativeModel(model)
                    contents = [{"parts": [{"text": prompt}]}]
                    tool_config = None
                    if enable_search_tool:
                        try:
                            # construct a ToolConfig to enable Google Search retrieval
                            tool_config = genai.protos.ToolConfig(google_search_retrieval=genai.protos.GoogleSearchRetrieval())
                        except Exception:
                            tool_config = None

                    # If a response_schema is supplied, try to construct a GenerationConfig
                    gen_cfg = None
                    if response_schema is not None and hasattr(genai, 'types'):
                        try:
                            gen_cfg = genai.types.GenerationConfig(response_schema=response_schema)
                        except Exception:
                            gen_cfg = None

                    resp = model_obj.generate_content(contents, tool_config=tool_config, generation_config=gen_cfg)
                except Exception:
                    resp = None

            # Fallback older/simpler SDK entrypoints
            if resp is None:
                # 1) generate_text
                if hasattr(genai, "generate_text"):
                    try:
                        resp = genai.generate_text(model=model, prompt=prompt)
                    except Exception:
                        try:
                            resp = genai.generate_text(model=model, messages=[{"content": prompt}])
                        except Exception:
                            resp = None

            if resp is None and hasattr(genai, "generate"):
                try:
                    resp = genai.generate(model=model, prompt=prompt)
                except Exception:
                    resp = None

            if resp is None and hasattr(genai, "TextGenerationModel"):
                try:
                    model_obj = genai.TextGenerationModel.from_pretrained(model)
                    resp = model_obj.generate(prompt)
                except Exception:
                    resp = None

            if resp is None:
                raise RuntimeError("SDK present but no supported generate API available")

            # Try to normalize SDK response to a dict-like or proto
            data = None
            try:
                if hasattr(resp, "to_dict"):
                    data = resp.to_dict()
                elif hasattr(resp, "as_dict"):
                    data = resp.as_dict()
                else:
                    # proto-like objects: attempt to access candidates/content
                    try:
                        data = resp
                    except Exception:
                        data = resp
            except Exception:
                data = resp

            text = _parse_response_for_text(data)
            if return_raw:
                return {"ok": True, "text": text, "raw": data, "status_code": 200}
            return {"ok": True, "text": text, "raw": None, "status_code": 200}

        except Exception as e:
            # if SDK call fails, fall back to REST
            sdk_err = str(e)
    else:
        sdk_err = None

    # Fallback: REST v1beta generateContent
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    payload: Dict[str, Any] = {
        "contents": [{"parts": [{"text": prompt}]}]
    }

    # Best-effort: include a field that might activate grounding if supported by the API
    if enable_search_tool:
        # Note: enabling search/tooling via REST may depend on the server-side support.
        # We add a `tool` hint in the payload; if the API ignores it, we'll still fall back.
        payload["toolHints"] = {"enableSearch": True}

    headers = {"Content-Type": "application/json"}

    try:
        resp = requests.post(url, params={"key": api_key}, headers=headers, json=payload, timeout=timeout)
    except Exception as e:
        err = f"network error: {e}"
        if sdk_err:
            err = f"sdk_error: {sdk_err}; rest_error: {err}"
        return {"ok": False, "error": err, "text": None, "raw": None, "status_code": None}

    status = resp.status_code
    if status != 200:
        body = None
        try:
            body = resp.text
        except Exception:
            body = "<unreadable body>"
        err = f"status {status}: {body}"
        if sdk_err:
            err = f"sdk_error: {sdk_err}; rest_error: {err}"
        return {"ok": False, "error": err, "text": None, "raw": body, "status_code": status}

    try:
        data = resp.json()
    except Exception:
        return {"ok": True, "text": None, "raw": resp.text, "status_code": status}

    text = _parse_response_for_text(data)
    if return_raw:
        return {"ok": True, "text": text, "raw": data, "status_code": status}
    return {"ok": True, "text": text, "raw": None, "status_code": status}
