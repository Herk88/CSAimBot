"""
Utilities that safely wrap OpenCV HighGUI calls and detect GUI backend support.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

import cv2  # type: ignore[import]
import numpy as np

__HIGHGUI_OK: bool | None = None
__HIGHGUI_REASON: Optional[str] = None
__PROBE_WINDOW_NAME = "__hg_probe__"


def has_highgui() -> bool:
    """
    Detect whether the active OpenCV build provides HighGUI support.

    The probe only runs once and caches its result. HighGUI calls are only allowed
    from the main thread to avoid backend crashes (notably on Windows/Qt builds).
    """
    global __HIGHGUI_OK, __HIGHGUI_REASON

    if __HIGHGUI_OK is not None:
        return __HIGHGUI_OK

    if threading.current_thread() is not threading.main_thread():
        __HIGHGUI_OK = False
        __HIGHGUI_REASON = "probe attempted from non-main thread"
        return __HIGHGUI_OK

    try:
        cv2.namedWindow(__PROBE_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.imshow(__PROBE_WINDOW_NAME, np.zeros((1, 1, 3), np.uint8))
        try:
            cv2.waitKey(1)
        except Exception:
            pass
        cv2.destroyWindow(__PROBE_WINDOW_NAME)
        try:
            cv2.waitKey(1)
        except Exception:
            pass
        __HIGHGUI_OK = True
        __HIGHGUI_REASON = "probe succeeded"
    except Exception as exc:
        __HIGHGUI_OK = False
        __HIGHGUI_REASON = f"cv2 window probe failed: {exc}"
    finally:
        try:
            cv2.destroyWindow(__PROBE_WINDOW_NAME)
        except Exception:
            pass

    return __HIGHGUI_OK


def highgui_probe_reason() -> Optional[str]:
    """
    Return the cached reason for the last HighGUI probe decision.
    """
    return __HIGHGUI_REASON


def imshow_safe(window: str, frame: Any) -> None:
    """
    Display an image using HighGUI when available; otherwise no-op.
    """
    if has_highgui():
        cv2.imshow(window, frame)


def waitkey_safe(delay: int = 1) -> int:
    """
    Read keyboard input via HighGUI when available; otherwise return -1.
    """
    if has_highgui():
        return cv2.waitKey(delay)
    return -1


def destroy_all_windows_safe() -> None:
    """
    Destroy all HighGUI windows when supported; otherwise no-op.
    """
    if has_highgui():
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
