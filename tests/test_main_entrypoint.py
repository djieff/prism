"""Tests for Prism application entrypoint wiring."""

from __future__ import annotations

import signal

import pytest

import prism.main as prism_main


def test_main_sets_sigint_and_exits_with_app_code(monkeypatch: pytest.MonkeyPatch) -> None:
    signal_calls: list[tuple[int, object]] = []
    timer_state: dict[str, object] = {
        "interval": None,
        "connected": None,
        "started": False,
    }
    window_state: dict[str, bool] = {"shown": False}

    class _FakeTimeout:
        def connect(self, callback) -> None:
            timer_state["connected"] = callback

    class _FakeTimer:
        def __init__(self) -> None:
            self.timeout = _FakeTimeout()

        def setInterval(self, interval: int) -> None:
            timer_state["interval"] = interval

        def start(self) -> None:
            timer_state["started"] = True

    class _FakeApp:
        def __init__(self, argv) -> None:
            self.argv = argv

        def exec(self) -> int:
            return 7

    class _FakeWindow:
        def show(self) -> None:
            window_state["shown"] = True

    def _fake_signal(sig: int, handler: object) -> None:
        signal_calls.append((sig, handler))

    monkeypatch.setattr(prism_main.signal, "signal", _fake_signal)
    monkeypatch.setattr(prism_main, "QTimer", _FakeTimer)
    monkeypatch.setattr(prism_main, "QApplication", _FakeApp)
    monkeypatch.setattr(prism_main, "MainWindow", _FakeWindow)

    with pytest.raises(SystemExit) as exc_info:
        prism_main.main()

    assert exc_info.value.code == 7
    assert signal_calls == [(signal.SIGINT, signal.SIG_DFL)]
    assert timer_state["interval"] == 100
    assert callable(timer_state["connected"])
    assert timer_state["started"] is True
    assert window_state["shown"] is True


def test_main_uses_runtime_sys_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_argv: list[list[str]] = []

    class _FakeTimeout:
        def connect(self, callback) -> None:
            del callback

    class _FakeTimer:
        def __init__(self) -> None:
            self.timeout = _FakeTimeout()

        def setInterval(self, interval: int) -> None:
            del interval

        def start(self) -> None:
            return

    class _FakeApp:
        def __init__(self, argv) -> None:
            captured_argv.append(list(argv))

        def exec(self) -> int:
            return 0

    class _FakeWindow:
        def show(self) -> None:
            return

    monkeypatch.setattr(prism_main, "QTimer", _FakeTimer)
    monkeypatch.setattr(prism_main, "QApplication", _FakeApp)
    monkeypatch.setattr(prism_main, "MainWindow", _FakeWindow)
    monkeypatch.setattr(prism_main.sys, "argv", ["prism", "--demo", "arg"])

    with pytest.raises(SystemExit) as exc_info:
        prism_main.main()

    assert exc_info.value.code == 0
    assert captured_argv == [["prism", "--demo", "arg"]]
