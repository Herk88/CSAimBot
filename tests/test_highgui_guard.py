import importlib
import threading

import numpy as np


def _reload_guard(monkeypatch):
    module = importlib.import_module("highgui_guard")
    module = importlib.reload(module)
    monkeypatch.setattr(module, "_highgui_guard__HIGHGUI_OK", None, raising=False)
    return module


def test_highgui_guard_headless(monkeypatch):
    hg = _reload_guard(monkeypatch)

    calls = {"named": 0}

    def named(*args, **kwargs):
        calls["named"] += 1
        raise RuntimeError("no gui")

    def boom(*args, **kwargs):
        raise RuntimeError("no gui")

    monkeypatch.setattr(hg.cv2, "namedWindow", named)
    monkeypatch.setattr(hg.cv2, "imshow", boom)
    monkeypatch.setattr(hg.cv2, "waitKey", lambda *args, **kwargs: -1)
    monkeypatch.setattr(hg.cv2, "destroyWindow", boom)

    assert hg.has_highgui() is False
    assert isinstance(hg.highgui_probe_reason(), str)
    hg.imshow_safe("x", np.zeros((1, 1, 3), np.uint8))
    hg.destroy_all_windows_safe()

    assert calls["named"] == 1


def test_has_highgui_non_main_thread_short_circuits(monkeypatch):
    hg = _reload_guard(monkeypatch)

    calls = {"named": 0}

    sentinel_main = object()
    sentinel_worker = object()

    monkeypatch.setattr(
        hg,
        "threading",
        type(
            "T",
            (),
            {
                "current_thread": staticmethod(lambda: sentinel_worker),
                "main_thread": staticmethod(lambda: sentinel_main),
            },
        ),
    )
    monkeypatch.setattr(
        hg.cv2,
        "namedWindow",
        lambda *args, **kwargs: calls.__setitem__("named", calls["named"] + 1),
    )
    monkeypatch.setattr(hg.cv2, "imshow", lambda *args, **kwargs: None)
    monkeypatch.setattr(hg.cv2, "waitKey", lambda *args, **kwargs: -1)

    assert hg.has_highgui() is False
    assert hg.highgui_probe_reason() == "probe attempted from non-main thread"
    assert calls["named"] == 0
