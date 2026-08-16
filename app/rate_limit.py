"""
Simple per-IP daily rate limiting for the /analyze endpoint.
Stored in Postgres so it survives Render's free-tier restarts.
"""
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, Request
from sqlalchemy.orm import Session

from app.config import settings
from app.models import RateLimit


def get_client_ip(request: Request) -> str:
    """
    Extracts client IP, respecting X-Forwarded-For when behind Render's proxy.
    Takes the first IP in the chain (the actual client).
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def check_and_increment(db: Session, ip: str) -> None:
    """
    Raises HTTPException(429) if the IP has hit its daily cap.
    Otherwise increments the counter (creating the row if needed).
    """
    now = datetime.now(timezone.utc)
    tomorrow_start = (now + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    row = db.get(RateLimit, ip)

    if row is None:
        row = RateLimit(ip=ip, analyses_today=1, reset_at=tomorrow_start)
        db.add(row)
        db.commit()
        return

    # Window rolled over — reset the counter
    if row.reset_at <= now:
        row.analyses_today = 1
        row.reset_at = tomorrow_start
        db.commit()
        return

    if row.analyses_today >= settings.max_analyses_per_ip_per_day:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Daily limit of {settings.max_analyses_per_ip_per_day} "
                f"analyses reached. Try again after "
                f"{row.reset_at.isoformat()}."
            ),
        )

    row.analyses_today += 1
    db.commit()


def check_and_increment_chat(db: Session, ip: str, hourly_limit: int = 30) -> None:
    """
    Rate limit for chat messages: N messages per IP per hour.
    Reuses the same rate_limits table by piggybacking on the counter,
    but with a 1-hour window instead of 24h.

    For simplicity, this uses a separate "chat:<ip>" key so chat and
    analyze counters don't collide.
    """
    now = datetime.now(timezone.utc)
    next_hour = (now + timedelta(hours=1)).replace(
        minute=0, second=0, microsecond=0
    )

    chat_key = f"chat:{ip}"
    row = db.get(RateLimit, chat_key)

    if row is None:
        row = RateLimit(ip=chat_key, analyses_today=1, reset_at=next_hour)
        db.add(row)
        db.commit()
        return

    if row.reset_at <= now:
        row.analyses_today = 1
        row.reset_at = next_hour
        db.commit()
        return

    if row.analyses_today >= hourly_limit:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Chat rate limit reached ({hourly_limit} messages/hour). "
                f"Try again after {row.reset_at.isoformat()}."
            ),
        )

    row.analyses_today += 1
    db.commit()
