"""
Lightweight HTTP server around vLLM native API that supports two-pass
thinking→answer with optional logits processor to cap thinking tokens.

Run:
  python -m hangman.providers.vllm_http_server \
      --model Qwen/Qwen3-14B \
      --port 8001 \
      --trust-remote-code

Requires: fastapi, uvicorn, vllm
"""

import argparse
import logging
import re
import json
from typing import List, Literal, Optional, Dict, Any, Union

from fastapi import FastAPI
from fastapi import Body
from pydantic import BaseModel, Field

from vllm import LLM, SamplingParams


app = FastAPI(title="vLLM Native Two-Pass Server")
LOGGER = logging.getLogger("uvicorn.error")


class GenerateRequest(BaseModel):
    messages: List[Dict[str, str]] = Field(..., description="List of chat messages: [{role, content}]")
    temperature: float = 0.3
    two_pass: bool = True
    think_tag: Literal["think", "thinking"] = "think"
    max_thinking_tokens: int = 256
    max_response_tokens: int = 1024
    # Tool-calling extensions
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[Literal["auto", "none"], Dict[str, str]]] = "auto"
    tool_parser: Optional[Literal["openai", "hermes"]] = "hermes"


class GenerateResponse(BaseModel):
    response: str
    thinking: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    pre_tool_text: Optional[str] = None


class ServerConfig(BaseModel):
    model: str
    trust_remote_code: bool = True
    dtype: Optional[str] = None


# Globals initialized on startup
LLM_ENGINE: Optional[LLM] = None
TOKENIZER: Any = None
SRV_CFG: Optional[ServerConfig] = None


def _apply_chat_template(messages: List[Dict[str, str]]) -> str:
    assert TOKENIZER is not None
    return TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _maybe_build_think_logits_processor(tag_open: str, tag_close: str, max_thinking_tokens: int):
    """
    Disabled for vLLM v1 compatibility: per-request logits processors are not supported.
    We rely on stop sequences and token caps instead.
    """
    return None


def _build_hermes_tools_preamble(tools: Optional[List[Dict[str, Any]]], tool_choice: Optional[Union[str, Dict[str, str]]]) -> str:
    if not tools:
        return ""
    lines: List[str] = []
    lines.append("# Tools")
    lines.append("")
    lines.append("You may call one or more functions to assist with the user query.")
    lines.append("")
    lines.append("You are provided with function signatures within <tools></tools> XML tags:")
    lines.append("<tools>")
    for t in tools:
        try:
            entry = {
                "type": "function",
                "function": {
                    "name": t.get("name"),
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {}),
                },
            }
            lines.append(json.dumps(entry))
        except Exception:
            continue
    lines.append("</tools>")
    lines.append("")
    lines.append("For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:")
    lines.append("<tool_call>")
    lines.append('{"name": <function-name>, "arguments": <args-json-object>}')
    lines.append("</tool_call>")

    if isinstance(tool_choice, dict) and tool_choice.get("name"):
        lines.append(f"You MUST use only the tool named '{tool_choice['name']}' if you choose to call a tool.")
    elif tool_choice == "none":
        lines.append("Do not call any tools; reply normally.")

    return "\n".join(lines)


def _extract_tool_calls(text: str) -> (List[Dict[str, Any]], str):
    """
    Extract zero or more <tool_call>...</tool_call> blocks and return (tool_calls, pre_tool_text).
    Each block may contain a single JSON object or a JSON array of calls.
    """
    calls: List[Dict[str, Any]] = []
    # Find all blocks
    blocks = list(re.finditer(r"<tool_call>([\s\S]*?)</tool_call>", text))
    if not blocks:
        return calls, text.strip()

    for bi, m in enumerate(blocks):
        inner = (m.group(1) or "").strip()
        data: Any = None
        try:
            data = json.loads(inner)
        except Exception:
            salvage = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", inner)
            if salvage:
                try:
                    data = json.loads(salvage.group(1))
                except Exception:
                    data = None

        if data is None:
            continue

        if isinstance(data, dict) and data.get("name"):
            calls.append({
                "id": f"call_{len(calls)+1}",
                "name": data.get("name"),
                "arguments": data.get("arguments", {}),
            })
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item.get("name"):
                    calls.append({
                        "id": f"call_{len(calls)+1}",
                        "name": item.get("name"),
                        "arguments": item.get("arguments", {}),
                    })

    # Remove all blocks to form pre_tool_text
    pre_tool_text = re.sub(r"<tool_call>[\s\S]*?</tool_call>", "", text).strip()
    return calls, pre_tool_text


def _two_pass_generate(req: GenerateRequest) -> GenerateResponse:
    assert LLM_ENGINE is not None
    # Optionally inject Hermes tools preamble
    messages = list(req.messages)
    if (req.tools and (req.tool_parser or "hermes") == "hermes"):
        preamble = _build_hermes_tools_preamble(req.tools, req.tool_choice)
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = (messages[0].get("content", "") + "\n\n" + preamble).strip()
        else:
            messages = [{"role": "system", "content": preamble}] + messages

    prompt_base = _apply_chat_template(messages)
    try:
        LOGGER.info("\n===== vLLM Server: Prompt (Base) =====\n%s\n======================================", prompt_base)
    except Exception:
        pass

    if not req.two_pass:
        # Single pass: return parsed thinking/response if tags are present
        sp = SamplingParams(temperature=req.temperature, max_tokens=req.max_response_tokens)
        outs = LLM_ENGINE.generate([prompt_base], sp)
        text = outs[0].outputs[0].text
        tag = req.think_tag
        start_open = f"<{tag}>"
        end_close = f"</{tag}>"
        if start_open in text:
            s = text.find(start_open)
            e = text.find(end_close)
            if e != -1 and e > s:
                thinking_core = text[s + len(start_open): e].strip()
                response_after = text[e + len(end_close):].strip()
                return GenerateResponse(response=response_after, thinking=thinking_core)
            else:
                # Opened but not closed: synthesize closing and surface error in response
                thinking_partial = text[s + len(start_open):].strip()
                thinking_full = f"{thinking_partial}{end_close}"
                return GenerateResponse(
                    response="Error: model did not close the thinking segment; no public answer generated.",
                    thinking=thinking_full,
                )
        return GenerateResponse(response=text.strip(), thinking="")

    start_open = f"<{req.think_tag}>"
    end_close = f"</{req.think_tag}>"

    # Pass 1: thinking
    prompt_think = prompt_base + start_open
    try:
        LOGGER.info("\n===== vLLM Server: Prompt (Pass 1 - thinking) =====\n%s\n==================================================", prompt_think)
    except Exception:
        pass
    sp1 = SamplingParams(
        temperature=req.temperature,
        max_tokens=req.max_thinking_tokens,
        stop=[end_close],
    )
    outs1 = LLM_ENGINE.generate([prompt_think], sp1)
    text1 = outs1[0].outputs[0].text

    if end_close in text1:
        end_idx = text1.find(end_close)
        thinking_core = text1[:end_idx].strip()
        response_prefix = text1[end_idx + len(end_close):].strip()
    else:
        thinking_core = text1.strip()
        response_prefix = ""

    # Pass 2: answer continuation
    closed_block = f"{start_open}{thinking_core}{end_close}"
    prompt_answer = prompt_base + closed_block
    try:
        LOGGER.info("\n===== vLLM Server: Prompt (Pass 2 - answer) =====\n%s\n================================================", prompt_answer)
    except Exception:
        pass
    # For Hermes tools, we do NOT add a stop on </tool_call>; let generation end naturally
    sp2 = SamplingParams(temperature=req.temperature, max_tokens=req.max_response_tokens)
    outs2 = LLM_ENGINE.generate([prompt_answer], sp2)
    text2 = outs2[0].outputs[0].text.strip()

    combined = (response_prefix + (" " if response_prefix and text2 else "") + text2).strip()

    # If Hermes tools are enabled, try extracting tool calls
    tool_calls: Optional[List[Dict[str, Any]]] = None
    pre_tool_text: Optional[str] = None
    if req.tools and (req.tool_parser or "hermes") == "hermes":
        calls, pre_text = _extract_tool_calls(combined)
        tool_calls = calls or []
        pre_tool_text = pre_text
        response = pre_text
    else:
        response = combined

    return GenerateResponse(response=response, thinking=thinking_core, tool_calls=tool_calls, pre_tool_text=pre_tool_text)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "model": SRV_CFG.model if SRV_CFG else "uninitialized"}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest = Body(...)) -> GenerateResponse:
    return _two_pass_generate(req)


def main():
    parser = argparse.ArgumentParser(description="vLLM Native Two-Pass HTTP Server")
    parser.add_argument("--model", required=True, help="HF model id or local path")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8001, help="Bind port")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to vLLM")
    parser.add_argument("--dtype", default=None, help="Optional dtype (e.g., bfloat16)")
    args = parser.parse_args()

    global LLM_ENGINE, TOKENIZER, SRV_CFG
    SRV_CFG = ServerConfig(model=args.model, trust_remote_code=args.trust_remote_code, dtype=args.dtype)

    logging.info("Loading vLLM model '%s' (dtype=%s) ...", args.model, args.dtype or "auto")
    if args.dtype:
        LLM_ENGINE = LLM(model=args.model, trust_remote_code=args.trust_remote_code, dtype=args.dtype)
    else:
        LLM_ENGINE = LLM(model=args.model, trust_remote_code=args.trust_remote_code)
    TOKENIZER = LLM_ENGINE.get_tokenizer()
    logging.info("Model loaded.")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()


