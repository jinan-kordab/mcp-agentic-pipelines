# Copyright (c) 2026 Jinan Kordab
# SPDX-License-Identifier: MIT

"""
Rate Limiter Middleware
========================
Token-bucket rate limiting per user and global.
Uses Redis for distributed rate limiting.

Limits:
  - 10 claims/minute per user
  - 100 claims/minute global
"""

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from app.services.cache import redis_client
from app.core.config import settings


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware for API endpoints."""

    async def dispatch(self, request: Request, call_next):
        # Only rate-limit POST /api/v1/claims
        if request.url.path == "/api/v1/claims" and request.method == "POST":
            # Extract user identity (placeholder — use JWT in production)
            user_id = request.headers.get("x-user-id", "anonymous")
            client_ip = request.client.host if request.client else "unknown"
            rate_key = f"rate:{user_id}:{client_ip}"

            allowed = await redis_client.check_rate_limit(rate_key)
            if not allowed:
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded. Please wait before submitting another claim.",
                )

        response = await call_next(request)
        return response
