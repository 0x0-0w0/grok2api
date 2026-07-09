"""CLI chat completion service — routes to cli-chat-proxy.grok.com/v1/chat/completions.

Standard OpenAI Chat Completions API — OAuth access_token auth.
"""

import asyncio
import json
from typing import Any, AsyncGenerator

import orjson

from app.platform.logging.logger import logger
from app.platform.config.snapshot import get_config
from app.platform.errors import RateLimitError, UpstreamError
from app.platform.runtime.clock import now_s
from app.platform.tokens import estimate_prompt_tokens, estimate_tokens
from app.control.account.enums import FeedbackKind
from app.control.account.invalid_credentials import feedback_kind_for_error
from app.control.account.runtime import get_refresh_service
from app.control.model.registry import resolve as resolve_model
from app.dataplane.account.selector import current_strategy
from app.dataplane.reverse.protocol.xai_cli_chat import (
    build_cli_payload,
    stream_cli_chat,
)
from app.products._account_selection import reserve_account, selection_max_retries
from app.products.openai.chat import _configured_retry_codes, _should_retry_upstream
from ._format import (
    make_response_id,
    make_stream_chunk,
    make_thinking_chunk,
    make_chat_response,
    build_usage,
)


def _log_task_exception(task: "asyncio.Task") -> None:
    exc = task.exception() if not task.cancelled() else None
    if exc:
        logger.warning("bg task failed: %s %s", task.get_name(), exc)


async def _quota_sync(token: str, mode_id: int) -> None:
    try:
        if current_strategy() != "quota" and mode_id != 6:
            return
        svc = get_refresh_service()
        if svc:
            await svc.refresh_call_async(token, mode_id)
    except Exception as exc:
        logger.warning("cli quota sync: %s... mode=%s %s", token[:10], mode_id, exc)


async def _fail_sync(token: str, mode_id: int, exc: BaseException | None = None) -> None:
    try:
        svc = get_refresh_service()
        if svc:
            await svc.record_failure_async(token, mode_id, exc)
    except Exception as e:
        logger.warning("cli fail sync: %s... mode=%s %s", token[:10], mode_id, e)


# In-memory cache: ssotoken → access_token (populated before DB commit)
_cli_token_cache: dict[str, str] = {}


async def _get_cli_access_token(ssotoken: str) -> str | None:
    """Look up cli_access_token, auto-refresh if expired, lazy-init if missing."""
    # Check in-memory cache first (hot path)
    cached = _cli_token_cache.get(ssotoken)
    if cached:
        return cached
    try:
        from app.control.account.runtime import get_account_repo
        from app.platform.runtime.clock import now_ms
        from app.dataplane.reverse.protocol.xai_oauth import acquire_cli_token, refresh_cli_token
        from app.control.account.commands import AccountPatch

        repo = get_account_repo()
        if repo is None:
            return None
        records = await repo.get_accounts([ssotoken])
        if not records:
            return None
        rec = records[0]
        access_token = rec.ext.get("cli_access_token")

        if not access_token:
            # --- Lazy init: no token yet, run full OAuth ---
            logger.info("cli token missing, lazy acquiring: %s...", ssotoken[:8])
            try:
                result = await acquire_cli_token(ssotoken)
            except Exception as exc:
                logger.warning("cli lazy acquire failed: %s... error=%s", ssotoken[:8], exc)
                return None
            expires_at = now_ms() + result["expires_in"] * 1000
            try:
                await repo.patch_accounts([AccountPatch(
                    token=ssotoken,
                    ext_merge={
                        "cli_access_token": result["access_token"],
                        "cli_refresh_token": result["refresh_token"],
                        "cli_expires_at": expires_at,
                        "cli_email": result.get("email", ""),
                        "cli_sub": result.get("sub", ""),
                    },
                )])
            except Exception as exc:
                logger.warning("cli lazy persist: %s... error=%s", ssotoken[:8], exc)
            access_token = result["access_token"]
            _cli_token_cache[ssotoken] = access_token
            return access_token

        expires_at = rec.ext.get("cli_expires_at")
        if expires_at and expires_at <= now_ms():
            # Token expired — try refresh
            refresh_tok = rec.ext.get("cli_refresh_token")
            if not refresh_tok:
                logger.warning("cli token expired and no refresh_token: %s...", ssotoken[:8])
                return None
            logger.info("cli token expired, refreshing: %s...", ssotoken[:8])
            try:
                result = await refresh_cli_token(refresh_tok)
            except Exception as exc:
                logger.warning("cli token refresh failed: %s... error=%s", ssotoken[:8], exc)
                return None
            if result is None:
                return None
            new_expires_at = now_ms() + result["expires_in"] * 1000
            try:
                await repo.patch_accounts([AccountPatch(
                    token=ssotoken,
                    ext_merge={
                        "cli_access_token": result["access_token"],
                        "cli_refresh_token": result.get("refresh_token", refresh_tok),
                        "cli_expires_at": new_expires_at,
                    },
                )])
            except Exception as exc:
                logger.warning("cli token persist after refresh: %s... error=%s", ssotoken[:8], exc)
            access_token = result["access_token"]

        return access_token
    except Exception:
        pass
    return None


async def completions(
    *,
    model: str,
    messages: list[dict],
    stream: bool = True,
    emit_think: bool | None = None,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> dict | AsyncGenerator[str, None]:
    cfg = get_config()
    spec = resolve_model(model)
    timeout_s = cfg.get_float("chat.timeout", 120.0)
    max_retries = selection_max_retries()
    retry_codes = _configured_retry_codes(cfg)
    response_id = make_response_id()

    logger.info("cli chat: model=%s stream=%s msgs=%s", model, stream, len(messages))

    from app.dataplane.account import _directory as _acct_dir
    if _acct_dir is None:
        raise RateLimitError("Account directory not initialised")
    directory = _acct_dir

    if stream:
        async def _run_stream() -> AsyncGenerator[str, None]:
            excluded: list[str] = []
            for attempt in range(max_retries + 1):
                acct, selected_mode_id = await reserve_account(
                    directory, spec,
                    now_s_override=now_s(),
                    exclude_tokens=excluded or None,
                )
                if acct is None:
                    raise RateLimitError("No available accounts")

                ssotoken = acct.token
                access_token = await _get_cli_access_token(ssotoken)
                if not access_token:
                    logger.warning("cli no token: %s...", ssotoken[:8])
                    excluded.append(ssotoken)
                    await directory.release(acct)
                    continue

                success = False
                fail_exc: BaseException | None = None
                full_text: list[str] = []
                usage_data: dict = {}

                try:
                    payload = build_cli_payload(
                        messages=messages, model="grok-4.5",
                        temperature=temperature, top_p=top_p, stream=True,
                    )
                    try:
                        _gen = stream_cli_chat(access_token, payload, timeout_s=timeout_s)
                        logger.info("cli stream_chat generator: type=%s", type(_gen))
                        async for line in _gen:
                            line = line.strip()
                            if not line or not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if data == "[DONE]":
                                continue
                            try:
                                obj = orjson.loads(data)
                            except Exception:
                                continue
                            choices = obj.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                reasoning = delta.get("reasoning_content", "")
                                if reasoning:
                                    chunk = make_thinking_chunk(response_id, model, reasoning)
                                    yield f"data: {orjson.dumps(chunk).decode()}\n\n"
                                if content:
                                    full_text.append(content)
                                    chunk = make_stream_chunk(response_id, model, content)
                                    yield f"data: {orjson.dumps(chunk).decode()}\n\n"
                            usg = obj.get("usage")
                            if usg:
                                usage_data = usg

                        usage = build_usage(
                            usage_data.get("prompt_tokens", 0) or estimate_prompt_tokens(messages),
                            usage_data.get("completion_tokens", 0) or estimate_tokens("".join(full_text)),
                        )
                        final = make_stream_chunk(response_id, model, "", is_final=True)
                        final["usage"] = usage
                        yield f"data: {orjson.dumps(final).decode()}\n\n"
                        yield "data: [DONE]\n\n"
                        success = True
                        logger.info("cli stream ok: %s/%s %s", attempt + 1, max_retries + 1, model)
                        return

                    except UpstreamError as exc:
                        fail_exc = exc
                        if _should_retry_upstream(exc, retry_codes) and attempt < max_retries:
                            logger.warning("cli retry: %s/%s status=%s", attempt + 1, max_retries, exc.status)
                            excluded.append(ssotoken)
                        else:
                            raise

                finally:
                    await directory.release(acct)
                    kind = (
                        FeedbackKind.SUCCESS if success
                        else feedback_kind_for_error(fail_exc) if fail_exc
                        else FeedbackKind.SERVER_ERROR
                    )
                    await directory.feedback(ssotoken, kind, selected_mode_id, now_s_val=now_s())
                    if success:
                        asyncio.create_task(_quota_sync(ssotoken, selected_mode_id)).add_done_callback(_log_task_exception)
                    else:
                        asyncio.create_task(_fail_sync(ssotoken, selected_mode_id, fail_exc)).add_done_callback(_log_task_exception)

                    if excluded and not success and ssotoken not in excluded:
                        continue
                    if not success and fail_exc:
                        raise fail_exc

            raise RateLimitError("No CLI accounts with a valid access token — import tokens and wait for OAuth")

        return _run_stream()

        # Non-streaming
        excluded: list[str] = []
        for attempt in range(max_retries + 1):
            acct, selected_mode_id = await reserve_account(
                directory, spec,
                now_s_override=now_s(),
                exclude_tokens=excluded or None,
            )
            if acct is None:
                raise RateLimitError("No available accounts")

            ssotoken = acct.token
            access_token = await _get_cli_access_token(ssotoken)
            if not access_token:
                excluded.append(ssotoken)
                await directory.release(acct)
                continue

            success = False
            fail_exc: BaseException | None = None
            try:
                payload = build_cli_payload(
                    messages=messages, model="grok-4.5",
                    temperature=temperature, top_p=top_p, stream=False,
                )
                from curl_cffi.requests import AsyncSession as _S
                async with _S(impersonate="chrome136") as _s:
                    r = await _s.post(
                        "https://cli-chat-proxy.grok.com/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {access_token}",
                            "X-XAI-Token-Auth": "xai-grok-cli",
                            "Content-Type": "application/json",
                            "x-grok-client-version": "0.2.93",
                        },
                        data=orjson.dumps(payload),
                        timeout=timeout_s,
                    )
                    if r.status_code != 200:
                        raise UpstreamError(f"CLI API returned {r.status_code}", status=r.status_code, body=r.text[:500])
                    obj = r.json()

                msg = obj.get("choices", [{}])[0].get("message", {})
                full_text = msg.get("content", "")
                usage_data = obj.get("usage", {})

                usage = build_usage(
                    usage_data.get("prompt_tokens", 0) or estimate_prompt_tokens(messages),
                    usage_data.get("completion_tokens", 0) or estimate_tokens(full_text),
                )
                resp = make_chat_response(response_id, model, full_text, usage)
                success = True
                logger.info("cli non-stream ok: %s", model)
                return resp

            except UpstreamError as exc:
                fail_exc = exc
                if _should_retry_upstream(exc, retry_codes) and attempt < max_retries:
                    excluded.append(ssotoken)
                    await directory.release(acct)
                    await directory.feedback(ssotoken, FeedbackKind.SUCCESS, selected_mode_id, now_s_val=now_s())
                    continue
                raise
            finally:
                if not success:
                    await directory.release(acct)
                    kind = feedback_kind_for_error(fail_exc) if fail_exc else FeedbackKind.SERVER_ERROR
                    await directory.feedback(ssotoken, kind, selected_mode_id, now_s_val=now_s())
                else:
                    asyncio.create_task(_quota_sync(ssotoken, selected_mode_id)).add_done_callback(_log_task_exception)


__all__ = ["completions"]
