"""Localhost CSRF guard for body-less state-changing endpoints (L5).

Binding 127.0.0.1 and the TrustedHost middleware stop remote hosts and DNS
rebinding, but neither stops a page open in a browser *on this machine* from
submitting a cross-site HTML form to a body-less POST (a form post carries no
JSON body to trip content-type validation, so endpoints that take no body are
reachable). Verifying the Origin/Referer host closes that class for the
handful of body-less mutations.
"""

from __future__ import annotations

from urllib.parse import urlparse

from fastapi import HTTPException, Request

# Hosts that count as "this machine". "testserver" is Starlette's TestClient
# default and app.js runs same-origin against 127.0.0.1/localhost.
_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "testserver"})


def verify_local_origin(request: Request) -> None:
    """Reject a state-changing request whose Origin/Referer names a
    non-localhost host. Absent headers (curl, non-browser agents) pass — the
    surface is localhost-only by bind; the guard exists purely to stop a
    same-machine browser being driven cross-site. The first header present
    decides (Origin preferred; browsers always send it on cross-origin
    POSTs)."""
    for header in ("origin", "referer"):
        value = request.headers.get(header)
        if not value:
            continue
        if urlparse(value).hostname not in _LOCAL_HOSTS:
            raise HTTPException(status_code=403, detail="cross-origin request rejected")
        return
