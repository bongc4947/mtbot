#!/usr/bin/env python
"""
MT5_Bot_mk2 — Dashboard Window (TradingAgents-style)
Runs in its OWN console window, launched by run.py.
Reads engine state from shared JSON file and renders Rich TUI.
"""
import sys
import os
import time
import threading

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from interface.state_bridge import read_state, write_command, StateSnapshot
from interface.metrics_view import (
    build_progress, build_messages, build_current_report,
    build_status_bar, build_header_title,
)
from interface.state_manager import SystemState

from rich.console import Console
from rich.live    import Live
from rich.layout  import Layout
from rich.panel   import Panel
from rich.text    import Text
from rich.align   import Align
from rich.rule    import Rule

# ── Key bindings ──────────────────────────────────────────────────────────────

_KEY_MAP = {
    b'p': "PAUSE",
    b'r': "RESUME",
    b's': "SHUTDOWN",
    b't': "RETRAIN",
    b'd': "REFRESH_DATA",
    b'c': "CLOSE_PROFITABLE",
    b'l': "CLOSE_LOSING",
    b'x': "CLOSE_ALL",
    # live config — handled via SET_PARAM commands
    b'1': "_CONF_UP",
    b'2': "_CONF_DN",
    b'3': "_TRADES_UP",
    b'4': "_TRADES_DN",
    b'5': "_MODE_PREV",
    b'6': "_MODE_NEXT",
}


_MODES = ["scalp", "balanced", "swing"]

def _kbd_thread(cmd_file: str, stop: threading.Event, state_file: str):
    try:
        import msvcrt
    except ImportError:
        return
    while not stop.is_set():
        try:
            if msvcrt.kbhit():
                raw = msvcrt.getch()
                cmd = _KEY_MAP.get(raw.lower())
                if not cmd:
                    time.sleep(0.05)
                    continue

                if cmd in ("PAUSE", "RESUME", "SHUTDOWN", "RETRAIN", "REFRESH_DATA",
                           "CLOSE_PROFITABLE", "CLOSE_LOSING", "CLOSE_ALL"):
                    write_command(cmd, cmd_file)
                else:
                    # Live config edit — read current values then adjust
                    from interface.state_bridge import read_state
                    data = read_state(state_file) or {}
                    rc   = data.get("runtime_config", {})
                    conf       = float(rc.get("confidence", 0.90))
                    max_trades = int(rc.get("max_trades_10min", 30))
                    mode       = rc.get("mode", "balanced")

                    if cmd == "_CONF_UP":
                        write_command(f"SET_PARAM:confidence:{conf + 0.01:.2f}", cmd_file)
                    elif cmd == "_CONF_DN":
                        write_command(f"SET_PARAM:confidence:{conf - 0.01:.2f}", cmd_file)
                    elif cmd == "_TRADES_UP":
                        write_command(f"SET_PARAM:max_trades_10min:{max_trades + 1}", cmd_file)
                    elif cmd == "_TRADES_DN":
                        write_command(f"SET_PARAM:max_trades_10min:{max_trades - 1}", cmd_file)
                    elif cmd in ("_MODE_PREV", "_MODE_NEXT"):
                        idx   = _MODES.index(mode) if mode in _MODES else 1
                        delta = 1 if cmd == "_MODE_NEXT" else -1
                        new_mode = _MODES[(idx + delta) % len(_MODES)]
                        write_command(f"SET_PARAM:mode:{new_mode}", cmd_file)
        except Exception:
            pass
        time.sleep(0.05)


# ── Layout builder ────────────────────────────────────────────────────────────

def _build_layout(snap: StateSnapshot) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="title",     size=3),
        Layout(name="main"),
        Layout(name="report",    size=14),
        Layout(name="statusbar", size=3),
    )
    layout["main"].split_row(
        Layout(name="progress", ratio=1),
        Layout(name="messages", ratio=1),
    )

    # Title bar
    title_txt = build_header_title(snap)
    layout["title"].update(
        Panel(Align.center(title_txt, vertical="middle"),
              border_style="cyan", style="on grey7")
    )

    layout["progress"].update(build_progress(snap))
    layout["messages"].update(build_messages(snap, n=22))
    layout["report"].update(build_current_report(snap))
    layout["statusbar"].update(
        Panel(Align.center(build_status_bar(snap), vertical="middle"),
              border_style="dim", style="on grey7")
    )
    return layout


_STARTUP_STEPS = [
    "Initializing",
    "GPU/CPU setup",
    "ZMQ bridge",
    "Symbol resolver",
    "Tick buffers",
    "Data export",
    "Model training",
    "Pre-flight",
    "Running",
]


def _waiting_layout(dots: int = 0) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="title",  size=3),
        Layout(name="body"),
    )
    title = Text("MT5_Bot_mk2 | Waiting for engine...", style="bold cyan", justify="center")
    layout["title"].update(Panel(Align.center(title, vertical="middle"),
                                 border_style="cyan", style="on grey7"))
    body = Text(justify="center")
    spinner = "." * (dots % 4)
    body.append(f"\n\nWaiting for MT5_Bot_mk2 engine{spinner}\n\n", style="bold yellow")
    body.append("Make sure  python run.py --mode live  is running.\n", style="dim")
    body.append("\nKeys: ", style="dim")
    for k, lbl in [("P","Pause"),("R","Resume"),("T","Retrain"),("D","Data"),("S","Shutdown"),
                   ("C","Close+"),("L","Close-"),("X","CloseAll"),
                   ("1","Conf+"),("2","Conf-"),("3","Trades+"),("4","Trades-"),
                   ("5","ModePrev"),("6","ModeNext")]:
        body.append(f"[{k}]", style="bold yellow")
        body.append(f" {lbl}   ", style="dim")
    layout["body"].update(Panel(Align.center(body, vertical="middle"), border_style="cyan"))
    return layout


def _startup_layout(snap: "StateSnapshot", stale: bool = False) -> Layout:
    """Show a progress screen while the engine is initializing (IDLE/TRAINING state)."""
    from interface.metrics_view import _SYSTEM_STATE_COLOR
    layout = Layout()
    layout.split_column(
        Layout(name="title",   size=3),
        Layout(name="body"),
        Layout(name="statusbar", size=3),
    )

    # Title
    sc = _SYSTEM_STATE_COLOR.get(snap.state, "yellow")
    title_txt = Text(justify="center")
    title_txt.append("MT5_Bot_mk2", style="bold cyan")
    title_txt.append("  |  STARTING UP  [", style="dim")
    title_txt.append(snap.state.value, style=f"bold {sc}")
    title_txt.append("]", style="dim")
    if stale:
        title_txt.append("  (reconnecting...)", style="dim yellow")
    layout["title"].update(
        Panel(Align.center(title_txt, vertical="middle"),
              border_style="cyan", style="on grey7")
    )

    # Body — progress
    body = Text(overflow="fold")
    pct = max(0, min(100, snap.training_progress))
    filled = int(pct / 5)
    bar = "[" + "#" * filled + "-" * (20 - filled) + "]"
    body.append(f"\n  Overall Progress: ", style="dim")
    body.append(f"{bar} {pct}%\n\n", style="bold cyan")

    # Report text (what the engine is currently doing)
    if snap.current_report:
        for line in snap.current_report.splitlines()[:10]:
            body.append(f"  {line}\n", style="white")

    body.append("\n\n  Startup steps: ", style="dim")
    for step in _STARTUP_STEPS:
        body.append(f"  {step}", style="dim")
    body.append("\n\n  This can take a few minutes on first run (data export + model training).\n", style="dim")

    layout["body"].update(Panel(body, title="[bold cyan]Startup Progress[/bold cyan]",
                                border_style="cyan"))

    # Status bar
    status_txt = Text(justify="center")
    status_txt.append("Keys: ", style="dim")
    for k, lbl in [("P","Pause"),("R","Resume"),("T","Retrain"),("D","Data"),("S","Shutdown"),
                   ("C","Close+"),("L","Close-"),("X","CloseAll")]:
        status_txt.append(f"[{k}]", style="bold yellow")
        status_txt.append(f" {lbl}   ", style="dim")
    layout["statusbar"].update(
        Panel(Align.center(status_txt, vertical="middle"),
              border_style="dim", style="on grey7")
    )
    return layout


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print("Usage: dashboard_window.py <state_file> <cmd_file>")
        sys.exit(1)

    state_file = sys.argv[1]
    cmd_file   = sys.argv[2]

    stop = threading.Event()
    kbd  = threading.Thread(target=_kbd_thread, args=(cmd_file, stop, state_file), daemon=True)
    kbd.start()

    console = Console(force_terminal=True, legacy_windows=False)

    last_snap = None
    dots      = 0

    with Live(_waiting_layout(), console=console, screen=True,
              refresh_per_second=4) as live:
        while True:
            try:
                data = read_state(state_file)
                stale = data is None

                if not stale:
                    snap      = StateSnapshot(data)
                    last_snap = snap
                elif last_snap is not None:
                    snap  = last_snap
                    stale = True
                else:
                    snap = None

                if snap is None:
                    # No data ever received — show waiting screen with spinner
                    dots += 1
                    live.update(_waiting_layout(dots))
                elif snap.state == SystemState.SHUTDOWN:
                    live.update(_waiting_layout())
                    time.sleep(1)
                    break
                elif snap.state in (SystemState.IDLE, SystemState.TRAINING) or stale:
                    # Engine is starting up or we lost connection — show progress screen
                    live.update(_startup_layout(snap, stale=stale))
                else:
                    live.update(_build_layout(snap))
            except Exception:
                pass
            time.sleep(0.25)

    stop.set()
    print("\nDashboard closed.")
    try:
        input("Press Enter to exit...")
    except Exception:
        pass


if __name__ == "__main__":
    main()
