"""
Bidirectional ZMQ controller.

Topology (EA binds, Python connects/binds):
  EA  PUSH binds   tcp://*:5557   → Python PULL connects  tcp://127.0.0.1:5557
  Python PUSH binds tcp://*:5558  → EA PULL connects      tcp://127.0.0.1:5558

Why EA binds:
  - EA is always-on inside MT5; Python restarts frequently.
  - EA binding avoids Windows "Permission denied" on well-known ports.
  - Python PUSH binds 5558 (localhost-only, unprivileged port, no conflict).
"""
import json
import threading
import time
import zmq
from utils.logger import get_logger

log = get_logger("zmq_controller")


class ZMQController:
    def __init__(self, cfg: dict, on_message_cb=None):
        self.cfg = cfg
        self.on_message_cb = on_message_cb  # callback(msg: dict)
        self._ctx = None
        self._push = None   # Python → EA  (Python binds 5558)
        self._pull = None   # EA   → Python (Python connects to EA's 5557)
        self._running = False
        self._recv_thread = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self):
        self._ctx = zmq.Context()

        # EA binds 5557 and pushes data to Python — Python connects to it
        ea_push_port = self.cfg.get("ea_push_port", 5557)
        # Python binds 5558 and pushes commands to EA — EA connects to it
        py_push_port = self.cfg.get("py_push_port", 5558)
        ea_host = self.cfg.get("ea_host", "127.0.0.1")

        # Python PULL — connect to EA's bound PUSH port
        self._pull = self._ctx.socket(zmq.PULL)
        self._pull.setsockopt(zmq.RCVHWM, 1000)
        self._pull.setsockopt(zmq.LINGER, 0)
        self._pull.connect(f"tcp://{ea_host}:{ea_push_port}")

        # Python PUSH — bind so EA can connect to it
        self._push = self._ctx.socket(zmq.PUSH)
        self._push.setsockopt(zmq.SNDHWM, 1000)
        self._push.setsockopt(zmq.LINGER, 0)
        self._push.bind(f"tcp://127.0.0.1:{py_push_port}")

        self._running = True
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()
        log.info(f"ZMQ started — PULL connects to EA:{ea_push_port}  "
                 f"PUSH binds :{py_push_port}")

    def stop(self):
        self._running = False
        if self._push:
            self._push.close()
        if self._pull:
            self._pull.close()
        if self._ctx:
            self._ctx.term()
        log.info("ZMQ stopped")

    # ------------------------------------------------------------------
    # Send (Python → EA)
    # ------------------------------------------------------------------
    def send(self, msg: dict):
        with self._lock:
            try:
                self._push.send_string(json.dumps(msg), zmq.NOBLOCK)
            except zmq.ZMQError as e:
                log.error(f"ZMQ send error: {e}")

    def send_trade(self, symbol: str, action: str, lot: float, sl: float, tp: float,
                   comment: str = "", magic: int = 20250002, ticket: int = 0):
        msg = {
            "type": "TRADE",
            "symbol": symbol,
            "action": action,   # BUY | SELL | CLOSE | MODIFY
            "lot": round(lot, 2),
            "sl": sl,
            "tp": tp,
            "comment": comment,
            "magic": magic,
        }
        if ticket:
            msg["ticket"] = ticket  # EA uses ticket for CLOSE/MODIFY when provided
        self.send(msg)

    def send_command(self, cmd: str, **kwargs):
        self.send({"type": "CMD", "cmd": cmd, **kwargs})

    def request_historical_export(self, symbol: str, timeframe: str = "M1", years: int = 5):
        self.send({"type": "CMD", "cmd": "EXPORT_HISTORICAL",
                   "symbol": symbol, "timeframe": timeframe, "years": years})

    def request_symbol_info(self, symbol: str):
        self.send({"type": "CMD", "cmd": "SYMBOL_INFO", "symbol": symbol})

    def send_heartbeat(self):
        self.send({"type": "HEARTBEAT", "ts": time.time()})

    # ------------------------------------------------------------------
    # Receive loop (EA → Python)
    # ------------------------------------------------------------------
    def _recv_loop(self):
        timeout_ms = self.cfg.get("timeout_ms", 3000)
        while self._running:
            try:
                if self._pull.poll(timeout_ms):
                    raw = self._pull.recv_string(zmq.NOBLOCK)
                    msg = json.loads(raw)
                    if self.on_message_cb:
                        self.on_message_cb(msg)
            except zmq.ZMQError as e:
                if self._running:
                    log.error(f"ZMQ recv error: {e}")
            except json.JSONDecodeError as e:
                log.warning(f"JSON decode error: {e}")
            except Exception as e:
                log.error(f"Unexpected recv error: {e}")
