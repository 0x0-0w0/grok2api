"""x.ai Grok CLI OAuth2 PKCE token acquisition.

Pure-protocol flow — no browser needed.  Uses curl_cffi impersonation
so the TLS fingerprint matches a real Chrome.

Public API
----------
.. function:: acquire_cli_token(sso: str, cf: str = "") -> dict

    Returns ``{"access_token", "refresh_token", "expires_in", "id_token",
    "email", "sub", "token_type"}`` on success.
    Raises ``RuntimeError`` when any step fails.
"""

import base64
import hashlib
import json
import re
import secrets
import uuid
import urllib.parse
from urllib.parse import urlencode

from curl_cffi.requests import AsyncSession

from app.platform.errors import UpstreamError
from app.platform.logging.logger import logger

CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
SCOPE = "openid profile email offline_access grok-cli:access api:access"
REDIRECT_URI = "http://127.0.0.1:56121/callback"
CONSENT_ACTION_ID = "4005315a1d7e426de592990bb54bb37471f39dd6d2"
BROWSER = "chrome136"


def _pkce():
    cv = secrets.token_urlsafe(64)
    cc = base64.urlsafe_b64encode(hashlib.sha256(cv.encode()).digest()).rstrip(b"=").decode()
    return cv, cc


def _headers(sso: str, cf: str = "", **kw) -> dict:
    ck = f"sso={sso}; sso-rw={sso}" + (f"; cf_clearance={cf}" if cf else "")
    h = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/149.0.0.0 Safari/537.36"
        ),
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "en-US,en;q=0.9",
        "Cookie": ck,
        "Sec-Ch-Ua": '"Google Chrome";v="149", "Chromium";v="149", "Not)A;Brand";v="24"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
        "x-xai-request-id": str(uuid.uuid4()),
    }
    h.update(kw)
    return h


async def acquire_cli_token(sso: str, cf: str = "", *, proxy_url: str = "") -> dict:
    """Run full OAuth2 PKCE flow and return token dict.

    Returns a dict with keys: ``access_token``, ``refresh_token``,
    ``expires_in``, ``id_token``, ``token_type``, ``email``, ``sub``.
    """

    cv, cc = _pkce()
    state = secrets.token_hex(16)
    nonce = secrets.token_hex(16)

    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPE,
        "code_challenge": cc,
        "code_challenge_method": "S256",
        "state": state,
        "nonce": nonce,
        "plan": "generic",
        "referrer": "cli-proxy-api",
    }
    auth_url = "https://auth.x.ai/oauth2/authorize?" + urlencode(params)

    session_kwargs: dict = {"impersonate": BROWSER}
    if proxy_url:
        session_kwargs["proxy"] = proxy_url

    async with AsyncSession(**session_kwargs) as session:
        # Step 1: GET auth.x.ai → 302/303 redirect
        h1 = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/149.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "en-US,en;q=0.9",
            "Cookie": f"sso={sso}; sso-rw={sso}" + (f"; cf_clearance={cf}" if cf else ""),
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
        }
        r = await session.get(auth_url, headers=h1, allow_redirects=False)
        if r.status_code not in (302, 303):
            raise RuntimeError(f"auth.x.ai returned {r.status_code} (expected 302/303)")

        loc = r.headers.get("location", "")
        if "/sign-in" in loc:
            m = re.search(r"return_to=([^&]+)", loc)
            if not m:
                raise RuntimeError("no return_to in sign-in redirect")
            consent_url = "https://accounts.x.ai" + urllib.parse.unquote(m.group(1))
        elif "/consent" in loc:
            consent_url = loc if loc.startswith("http") else "https://accounts.x.ai" + loc
        else:
            raise RuntimeError(f"unexpected redirect: {loc[:100]}")

        logger.debug("xai oauth consent url resolved: token={}...", sso[:8])

        # Step 2: GET consent page
        h2 = _headers(sso, cf,
            Accept="text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            **{"Sec-Fetch-Dest": "document", "Sec-Fetch-Mode": "navigate",
               "Sec-Fetch-Site": "none", "Referer": "https://auth.x.ai/"},
        )
        r = await session.get(consent_url, headers=h2)
        if r.status_code != 200 or "sign-in" in r.text[:3000].lower():
            raise RuntimeError("consent page not returned (SSO may be expired)")

        # Step 3: POST consent
        h3 = _headers(sso, cf, **{
            "Content-Type": "text/plain;charset=UTF-8",
            "Next-Action": CONSENT_ACTION_ID,
            "Accept": "text/x-component",
            "Referer": consent_url,
            "Origin": "https://accounts.x.ai",
        })
        body = json.dumps([{
            "action": "allow",
            "clientId": CLIENT_ID,
            "redirectUri": REDIRECT_URI,
            "scope": SCOPE,
            "state": state,
            "codeChallenge": cc,
            "codeChallengeMethod": "S256",
            "nonce": nonce,
            "principalType": "User",
            "principalId": "",
            "referrer": "cli-proxy-api",
        }])
        r = await session.post(consent_url, headers=h3, data=body)
        text = r.text
        loc = r.headers.get("location", "")

        # Step 4: Extract auth code
        code = None
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line.split(":", 1)[1])
                if data.get("action") == "allow" and data.get("success"):
                    code = data.get("code")
                    break
            except (json.JSONDecodeError, IndexError, AttributeError):
                continue
        if not code:
            m = re.search(r'"code"\s*:\s*"([A-Za-z0-9_\-]+)"', text)
            if m:
                code = m.group(1)
        if not code and loc:
            m = re.search(r"[?&]code=([^&]+)", loc)
            if m:
                code = m.group(1)
        if not code:
            raise RuntimeError("failed to extract authorization code from consent response")

        # Step 5: Exchange code for tokens
        h6 = _headers(sso, cf, **{
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        })
        form = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": REDIRECT_URI,
            "client_id": CLIENT_ID,
            "code_verifier": cv,
        }
        r = await session.post("https://auth.x.ai/oauth2/token", headers=h6, data=form)
        if r.status_code != 200:
            raise RuntimeError(f"token exchange failed: {r.status_code} {r.text[:200]}")

        data = r.json()

    email = sub = ""
    id_token = data.get("id_token", "")
    if id_token:
        parts = id_token.split(".")
        if len(parts) >= 2:
            payload = parts[1] + "=" * ((4 - len(parts[1]) % 4) % 4)
            try:
                claims = json.loads(base64.urlsafe_b64decode(payload))
                email = claims.get("email", "")
                sub = claims.get("sub", "")
            except Exception:
                pass

    return {
        "access_token": data["access_token"],
        "refresh_token": data.get("refresh_token", ""),
        "expires_in": data.get("expires_in", 3600),
        "id_token": id_token,
        "token_type": data.get("token_type", "Bearer"),
        "email": email,
        "sub": sub,
    }


async def refresh_cli_token(refresh_token: str, *, proxy_url: str = "") -> dict | None:
    """Refresh an expired CLI access_token using the stored refresh_token.

    Returns the same dict shape as ``acquire_cli_token``, or ``None`` on failure.
    """

    from curl_cffi.requests import AsyncSession

    logger.debug("xai oauth refresh: refresh_token={}...", refresh_token[:8])

    session_kwargs: dict = {"impersonate": BROWSER}
    if proxy_url:
        session_kwargs["proxy"] = proxy_url

    async with AsyncSession(**session_kwargs) as session:
        try:
            r = await session.post(
                "https://auth.x.ai/oauth2/token",
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/149.0.0.0 Safari/537.36"
                    ),
                },
                data={
                    "grant_type": "refresh_token",
                    "client_id": CLIENT_ID,
                    "refresh_token": refresh_token,
                },
                timeout=15.0,
            )
            if r.status_code != 200:
                logger.warning("xai oauth refresh failed: status={} body={}", r.status_code, r.text[:200])
                return None

            data = r.json()
        except Exception as exc:
            logger.warning("xai oauth refresh transport error: {}", exc)
            return None

    email = sub = ""
    id_token = data.get("id_token", "")
    if id_token:
        parts = id_token.split(".")
        if len(parts) >= 2:
            payload = parts[1] + "=" * ((4 - len(parts[1]) % 4) % 4)
            try:
                claims = json.loads(base64.urlsafe_b64decode(payload))
                email = claims.get("email", "")
                sub = claims.get("sub", "")
            except Exception:
                pass

    return {
        "access_token": data["access_token"],
        "refresh_token": data.get("refresh_token", refresh_token),
        "expires_in": data.get("expires_in", 3600),
        "id_token": id_token,
        "token_type": data.get("token_type", "Bearer"),
        "email": email,
        "sub": sub,
    }


__all__ = ["acquire_cli_token", "refresh_cli_token"]
