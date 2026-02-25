"""
Centralized error handling for the frontend and backend.
Maps exceptions to user-friendly messages and provides safe wrappers.
"""
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")


def user_facing_message(exc: BaseException) -> str:
    """
    Return a short, user-friendly message for an exception.
    Use in UI (st.error, st.warning) and logs.
    """
    if isinstance(exc, FileNotFoundError):
        return f"File or path not found: {exc}"
    if isinstance(exc, PermissionError):
        return "Permission denied. Check file/folder access."
    if isinstance(exc, ValueError):
        return str(exc) if exc else "Invalid value or data."
    if isinstance(exc, ImportError):
        return f"Missing dependency or import error: {exc}"
    if isinstance(exc, KeyError):
        return f"Missing expected data: {exc}"
    if isinstance(exc, TypeError):
        return f"Invalid type or usage: {exc}"
    if isinstance(exc, MemoryError):
        return "Not enough memory. Try a smaller file or dataset."
    if isinstance(exc, UnicodeDecodeError):
        return "File encoding error. Try saving the file as UTF-8."
    if isinstance(exc, Exception):
        return str(exc) if str(exc) else type(exc).__name__
    return "An error occurred."


def safe_run(
    fn: Callable[..., T],
    default: T,
    *args: Any,
    **kwargs: Any,
) -> T:
    """
    Run fn(*args, **kwargs); on any exception return default.
    Useful for non-critical code (e.g. loading optional stats).
    """
    try:
        return fn(*args, **kwargs)
    except Exception:
        return default
