"""串行 + RPM 限流器 for console.x.ai 请求.

console.x.ai 免费账号共享同一 team(60 RPM + 严格并发窗口),所有 SSO token
都映射到它。换 token 无用,只能在网关侧串行化所有 console 请求(Semaphore),
彻底避免并发互相烧配额;并加 RPM 兜底,防止单测活窗口反复重试烧光 60。

设计:
- Semaphore(burst): 限制同时在飞的 console 请求数(burst=2 → 最多 2 并发)
- Token bucket RPM: 限制每分钟总请求数(rpm=50,留 10 余量给真实流量)
- __aenter__ 取 semaphore + 1 个 RPM token;30s 内拿不到 → 抛 RateLimitError
  (快速失败,让 NewAPI 立即看到 429 而不是干等 90s 超时)
- __aexit__ 释放 semaphore(通过 async with 语义,异常/成功都释放)
"""

import asyncio
import time

from app.platform.errors import RateLimitError
from app.platform.logging.logger import logger
from app.platform.config.snapshot import get_config


class ConsoleRateLimiter:
    """Semaphore 串行 + token bucket RPM 兜底。"""

    def __init__(self, burst: int, rpm: float, max_wait: float = 30.0) -> None:
        self._burst = burst
        self._rpm = rpm
        self._max_wait = max_wait
        self._sem = asyncio.Semaphore(burst)
        self._tokens = rpm  # 预算满载
        self._last = time.monotonic()
        self._tok_lock = asyncio.Lock()

    async def _acquire_token(self, max_wait: float) -> bool:
        """取 1 个 RPM token,最多等 max_wait 秒。"""
        async with self._tok_lock:
            now = time.monotonic()
            elapsed = now - self._last
            if elapsed > 0:
                self._tokens = min(self._rpm, self._tokens + elapsed * (self._rpm / 60.0))
                self._last = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            deficit = 1.0 - self._tokens
            wait = min(max_wait, deficit / (self._rpm / 60.0)) if self._rpm > 0 else max_wait
        if wait <= 0:
            return False
        try:
            await asyncio.sleep(wait)
        except asyncio.CancelledError:
            return False
        async with self._tok_lock:
            now = time.monotonic()
            elapsed = now - self._last
            if elapsed > 0:
                self._tokens = min(self._rpm, self._tokens + elapsed * (self._rpm / 60.0))
                self._last = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
        return False

    async def __aenter__(self) -> "ConsoleRateLimiter":
        await self._sem.acquire()
        ok = await self._acquire_token(self._max_wait)
        if not ok:
            self._sem.release()
            raise RateLimitError(
                "console rate limit: capacity exhausted (rpm={} burst={})".format(
                    self._rpm, self._burst
                )
            )
        return self

    async def __aexit__(self, *exc) -> None:
        self._sem.release()


_limiter: ConsoleRateLimiter | None = None


def get_console_rate_limiter() -> ConsoleRateLimiter:
    """单例限流器。burst=2 并发,rpm=50(留 10 余量)。"""
    global _limiter
    if _limiter is None:
        cfg = get_config()
        rpm = cfg.get_float("console.rate_limit_rpm", 50.0)
        burst = int(cfg.get_float("console.rate_limit_burst", 2.0))
        max_wait = cfg.get_float("console.rate_limit_max_wait", 30.0)
        _limiter = ConsoleRateLimiter(burst=burst, rpm=rpm, max_wait=max_wait)
        logger.info(
            "console rate limiter initialized: rpm={} burst={} max_wait={}s",
            rpm, burst, max_wait,
        )
    return _limiter


def reset_console_rate_limiter() -> None:
    global _limiter
    _limiter = None


__all__ = ["ConsoleRateLimiter", "get_console_rate_limiter", "reset_console_rate_limiter"]
