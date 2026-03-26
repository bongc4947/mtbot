"""
Dashboard launcher for MT5_Bot_mk2.

Responsibilities:
  1. Route log records into StateManager.log_buffer (DashboardHandler).
  2. Suppress engine terminal output so the engine console stays clean.
  3. Launch dashboard_window.py in a NEW console window (subprocess).
"""
import logging
import os
import subprocess
import sys

from interface.state_manager import get_state_manager, StateManager


# ── Logging bridge ────────────────────────────────────────────────────────────

class DashboardHandler(logging.Handler):
    """Feeds every log record into StateManager.log_buffer (thread-safe)."""

    def __init__(self, sm: StateManager):
        super().__init__(level=logging.DEBUG)
        self._sm = sm

    def emit(self, record: logging.LogRecord):
        try:
            self._sm.add_log(record.levelname, record.name, record.getMessage())
        except Exception:
            self.handleError(record)


def _attach_log_routing():
    """
    Add DashboardHandler to root logger and remove all StreamHandlers from
    every existing logger so the engine terminal stays clean.
    """
    sm      = get_state_manager()
    handler = DashboardHandler(sm)

    root = logging.getLogger()

    # Remove StreamHandlers from root
    for h in list(root.handlers):
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            root.removeHandler(h)

    # Remove StreamHandlers from all child loggers already created
    for name in list(logging.Logger.manager.loggerDict.keys()):
        logger = logging.getLogger(name)
        for h in list(logger.handlers):
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                logger.removeHandler(h)

    root.addHandler(handler)

    # Tell utils.logger not to add new StreamHandlers
    try:
        import utils.logger as _ul
        _ul._dashboard_active = True
    except Exception:
        pass


# ── Subprocess launcher ───────────────────────────────────────────────────────

def launch_dashboard_window(state_path: str, cmd_path: str):
    """Open dashboard_window.py in a brand-new console window."""
    script = os.path.join(os.path.dirname(__file__), "dashboard_window.py")
    python = sys.executable

    if sys.platform == "win32":
        subprocess.Popen(
            [python, script, state_path, cmd_path],
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )
    else:
        # Fallback for Linux/Mac: try xterm, then gnome-terminal
        for term in (["xterm", "-e"], ["gnome-terminal", "--"]):
            try:
                subprocess.Popen(term + [python, script, state_path, cmd_path])
                break
            except FileNotFoundError:
                continue


# ── Public API ────────────────────────────────────────────────────────────────

def start_dashboard(state_path: str, cmd_path: str):
    """
    Call once in run_live() before the main loop.
    Attaches log routing, silences the engine terminal,
    and opens the dashboard in a new window.
    """
    _attach_log_routing()
    launch_dashboard_window(state_path, cmd_path)
