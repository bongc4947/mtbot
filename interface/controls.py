"""
Non-blocking keyboard input handler for Windows.
Maps single keypresses to Command enum values and puts them in the
StateManager command queue.
"""
import sys
import threading
import time

from interface.state_manager import get_state_manager, Command

_KEY_MAP: dict[bytes, Command] = {
    b'p': Command.PAUSE,
    b'r': Command.RESUME,
    b's': Command.SHUTDOWN,
    b't': Command.RETRAIN,
    b'd': Command.REFRESH_DATA,
}


def _input_loop():
    try:
        import msvcrt
    except ImportError:
        return  # not on Windows — skip

    sm = get_state_manager()
    while not sm.shutdown_flag.is_set():
        try:
            if msvcrt.kbhit():
                raw = msvcrt.getch()
                cmd = _KEY_MAP.get(raw.lower())
                if cmd is not None:
                    sm.command_queue.put(cmd)
        except Exception:
            pass
        time.sleep(0.05)


def start_input_handler():
    """Launch the keyboard capture thread (daemon, Windows only)."""
    if sys.platform != "win32":
        return
    t = threading.Thread(target=_input_loop, daemon=True, name="kbd_input")
    t.start()
