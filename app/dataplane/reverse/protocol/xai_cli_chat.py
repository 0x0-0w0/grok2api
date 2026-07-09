"""XAI CLI (cli-chat-proxy.grok.com) chat protocol.

端点: POST https://cli-chat-proxy.grok.com/v1/chat/completions
认证: Authorization: Bearer <access_token> + X-XAI-Token-Auth: xai-grok-cli
格式: Standard OpenAI Chat Completions API (request & response)
"""

from typing import Any, AsyncGenerator

import orjson

from app.platform.errors import UpstreamError
from app.platform.logging.logger import logger
from app.dataplane.reverse.runtime.endpoint_table import CLI_CHAT_COMPLETIONS

def build_cli_payload(
    *,
    messages: list[dict[str, Any]],
    model: str,
    temperature: float = 0.7,
    top_p: float = 0.95,
    stream: bool = True,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "stream": stream,
    }


async def stream_cli_chat(
    access_token: str,
    payload: dict[str, Any],
    *,
    timeout_s: float = 120.0,
) -> AsyncGenerator[str, None]:
    """POST to cli-chat-proxy, yield raw SSE lines (data: ...)."""

    from curl_cffi.requests import AsyncSession

    headers = {
        "Authorization": f"Bearer {access_token}",
        "X-XAI-Token-Auth": "xai-grok-cli",
        "Content-Type": "application/json",
        "x-grok-client-version": "0.2.93",
        "Accept": "text/event-stream",
    }
    payload_bytes = orjson.dumps(payload)

    async with AsyncSession(impersonate="chrome136") as session:
        try:
            response = await session.post(
                CLI_CHAT_COMPLETIONS,
                headers=headers,
                data=payload_bytes,
                timeout=timeout_s,
                stream=True,
            )
        except Exception as exc:
            raise UpstreamError(f"CLI transport failed: {exc}", status=502) from exc

        if response.status_code != 200:
            try:
                body = response.content.decode("utf-8", "replace")[:500]
            except Exception:
                body = ""
            raise UpstreamError(
                f"CLI API returned {response.status_code}",
                status=response.status_code,
                body=body,
            )

        try:
            async for raw_line in response.aiter_lines():
                if isinstance(raw_line, bytes):
                    try:
                        raw_line = raw_line.decode("utf-8")
                    except UnicodeDecodeError:
                        raw_line = raw_line.decode("utf-8", errors="replace")
                yield raw_line
        except Exception as exc:
            raise UpstreamError(f"CLI stream read failed: {exc}", status=502) from exc


__all__ = [
    "build_cli_payload",
    "stream_cli_chat",
]
