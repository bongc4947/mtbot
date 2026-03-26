"""
MT5_Bot_mk2 -- Entry point.

Usage:
    python run.py --mode live
    python run.py --mode train
    python run.py --mode live --no-dashboard
"""
import argparse
import os
import sys
import time
import threading
import signal as os_signal
import queue

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.config_loader import load_config, load_symbols
from utils.logger import get_logger
from utils.gpu_utils import AccelerationContext
from utils.data_utils import list_available_symbols

from ingestion.zmq_controller import ZMQController
from ingestion.tick_buffer import TickBufferManager
from ingestion.historical_exporter import HistoricalExporter
from ingestion.message_router import MessageRouter
from ingestion.symbol_resolver import SymbolResolver

from models.model_manager import ModelManager
from training.trainer import train_symbol, train_all_symbols
from inference.engine import InferenceEngine

from execution.risk_manager import RiskManager
from execution.position_sizer import PositionSizer
from execution.trailing_manager import TrailingManager
from execution.trade_stacker import TradeStacker
from execution.micro_scalper import MicroScalper

from interface.state_manager import (
    get_state_manager, SystemState, Command, TradeInfo
)
from interface.state_bridge import get_ipc_paths, write_state, read_commands

log = get_logger("main")

SHUTDOWN_FLAG = threading.Event()


def handle_signal(signum, frame):
    log.info(f"Signal {signum} received -- initiating shutdown")
    SHUTDOWN_FLAG.set()
    get_state_manager().shutdown_flag.set()


os_signal.signal(os_signal.SIGINT,  handle_signal)
os_signal.signal(os_signal.SIGTERM, handle_signal)


# ------------------------------------------------------------------------------
# Init-phase helpers — keep dashboard responsive during long startup steps
# ------------------------------------------------------------------------------

def _push_state(sm, state_path):
    """Write state to IPC file so the dashboard window refreshes."""
    try:
        write_state(sm, state_path)
    except Exception:
        pass


def _check_init_commands(sm, cmd_path) -> bool:
    """
    Read and process dashboard commands during initialization.
    Returns True if SHUTDOWN was requested (caller should abort).
    """
    try:
        for raw_cmd in read_commands(cmd_path):
            try:
                cmd = Command[raw_cmd]
                if cmd == Command.SHUTDOWN:
                    log.info("Shutdown requested during initialization")
                    SHUTDOWN_FLAG.set()
                    sm.shutdown_flag.set()
                    return True
            except (KeyError, Exception):
                pass
    except Exception:
        pass
    return SHUTDOWN_FLAG.is_set()


def _resolver_with_countdown(resolver, symbols, timeout, sm, state_path, cmd_path):
    """
    Run symbol resolver while publishing a live countdown to the dashboard.
    The countdown updates every second so the operator knows the bot is alive.
    Returns the resolved symbol list.
    """
    result_holder = []
    done_event    = threading.Event()

    def _run():
        result_holder.extend(resolver.resolve(symbols, timeout=timeout))
        done_event.set()

    thread = threading.Thread(target=_run, daemon=True, name="resolver")
    thread.start()

    t0 = time.time()
    while not done_event.is_set():
        elapsed   = time.time() - t0
        remaining = max(0.0, timeout - elapsed)

        sm.update_agent_status("ZMQ Bridge", "running", "handshake")
        pct_resolver = 18 + int(min(1.0, elapsed / max(timeout, 1)) * 7)
        sm.training_progress = pct_resolver
        sm.set_current_report(
            f"Step 3/8: Connecting to MetaTrader 5 EA...\n"
            f"Waiting for Market Watch symbol list ({remaining:.0f}s remaining)\n\n"
            f"If this hangs: open MT5, attach the Expert Advisor to a chart\n"
            f"and make sure it shows a smiley face (not an X).\n\n"
            f"Press [S] in the dashboard or Ctrl+C here to abort."
        )
        _push_state(sm, state_path)

        if _check_init_commands(sm, cmd_path):
            return []

        done_event.wait(timeout=1.0)

    return result_holder


def _export_with_progress(exporter, resolved_symbols, sm, state_path, cmd_path):
    """
    Run historical export while pushing progress updates to dashboard each second.
    """
    result_holder = []
    done_event    = threading.Event()
    progress_info = {"current": "", "done": 0, "total": len(resolved_symbols)}

    def _run():
        # ensure_all_symbols calls per-symbol export with blocking waits
        ready = exporter.ensure_all_symbols(resolved_symbols)
        result_holder.extend(ready)
        done_event.set()

    thread = threading.Thread(target=_run, daemon=True, name="exporter")
    thread.start()

    sm.update_agent_status("Data Exporter", "running", "exporting CSVs")
    t0 = time.time()
    while not done_event.is_set():
        elapsed = time.time() - t0
        # Animate 32-60% over first 5 minutes; cap at 59 until done
        pct_export = 32 + int(min(1.0, elapsed / 300.0) * 27)
        sm.training_progress = pct_export
        sym_done = progress_info.get("done", 0)
        sym_total = progress_info.get("total", 1)
        sm.set_current_report(
            f"Step 5/8: Exporting historical data from MT5...\n"
            f"Symbols: {sym_done}/{sym_total}  |  Elapsed: {elapsed:.0f}s\n\n"
            f"This may take several minutes for large symbol lists.\n"
            f"Press [S] in dashboard to abort."
        )
        _push_state(sm, state_path)
        if _check_init_commands(sm, cmd_path):
            return []
        done_event.wait(timeout=1.0)

    sm.training_progress = 60
    sm.update_agent_status("Data Exporter", "completed",
                            f"{len(result_holder)} symbols")
    return result_holder


def _preflight_gate(sm, state_path, cmd_path, ready_symbols,
                    auto_start_secs: int = 60) -> bool:
    """
    Pause before live trading begins. Shows readiness summary in dashboard.
    Operator confirms with [R] in dashboard or [Enter]/[S]/[R] in this console.
    Auto-starts after `auto_start_secs` if no input (non-blocking default).

    Returns True  -> proceed to trading
    Returns False -> shutdown requested
    """
    sm.set_state(SystemState.PAUSED)
    sm.training_progress = 95
    log.info("-" * 60)
    log.info(f"  PRE-FLIGHT CHECK  |  {len(ready_symbols)} symbols ready")
    log.info(f"  Press [R] in the dashboard  OR  [Enter]/[S]/[R] here")
    log.info(f"  Auto-start in {auto_start_secs}s  |  [S] in dashboard to abort")
    log.info("-" * 60)

    confirmed = threading.Event()

    # Console keyboard thread (main terminal window)
    def _console_kbd():
        try:
            import msvcrt
            while not confirmed.is_set() and not SHUTDOWN_FLAG.is_set():
                if msvcrt.kbhit():
                    key = msvcrt.getch().lower()
                    if key in (b'\r', b'\n', b's', b'r'):
                        log.info("Start confirmed via console keyboard")
                        confirmed.set()
                    elif key == b'\x03':  # Ctrl+C
                        SHUTDOWN_FLAG.set()
                        get_state_manager().shutdown_flag.set()
                        confirmed.set()
                time.sleep(0.05)
        except Exception:
            pass  # msvcrt not available (non-Windows)

    kbd = threading.Thread(target=_console_kbd, daemon=True, name="preflight_kbd")
    kbd.start()

    deadline = time.time() + auto_start_secs
    while not SHUTDOWN_FLAG.is_set() and not confirmed.is_set():
        remaining = max(0.0, deadline - time.time())

        # Build a summary of trained models for the report
        model_lines = []
        for sym, info in list(sm.model_info.items())[:8]:
            wr  = info.get("win_rate", 0.0)
            exp = info.get("expectancy", 0.0)
            model_lines.append(f"  {sym:12s}  WR:{wr:.0%}  E:{exp:+.2f}")
        model_summary = "\n".join(model_lines) if model_lines else "  (no models trained yet)"

        sm.set_current_report(
            f"Step 8/8: Pre-flight Ready\n"
            f"Symbols ready: {len(ready_symbols)}  |  Auto-start in: {remaining:.0f}s\n\n"
            f"Model summary:\n{model_summary}\n\n"
            f"Press [R] in dashboard OR [Enter]/[R]/[S] in this console to start.\n"
            f"Press [S] in dashboard or Ctrl+C to abort."
        )
        _push_state(sm, state_path)

        # Check dashboard commands
        for raw_cmd in read_commands(cmd_path):
            try:
                cmd = Command[raw_cmd]
                if cmd in (Command.RESUME,):
                    log.info("Start confirmed from dashboard [R]")
                    confirmed.set()
                elif cmd == Command.SHUTDOWN:
                    SHUTDOWN_FLAG.set()
                    sm.shutdown_flag.set()
                    confirmed.set()
            except Exception:
                pass

        if remaining <= 0:
            log.info(f"Pre-flight timeout ({auto_start_secs}s) -- auto-starting")
            break

        SHUTDOWN_FLAG.wait(timeout=0.5)

    confirmed.set()  # signal kbd thread to stop

    if SHUTDOWN_FLAG.is_set():
        return False

    log.info("Pre-flight passed -- starting live trading")
    return True


# ------------------------------------------------------------------------------
# Background retrain helper
# ------------------------------------------------------------------------------

def _do_retrain(symbols: list, raw_dir: str, model_manager, cfg: dict):
    sm    = get_state_manager()
    total = max(1, len(symbols))
    sm.set_state(SystemState.TRAINING)
    log.info(f"[retrain] Starting for {total} symbols")

    for i, sym in enumerate(symbols):
        sm.training_symbol   = sym
        sm.training_progress = int(i / total * 100)
        min_samples = cfg.get("model", {}).get("min_train_samples", 5000)
        window      = cfg.get("model", {}).get("input_window", 60)
        train_symbol(sym, raw_dir, model_manager, min_samples, window)

    sm.training_symbol   = ""
    sm.training_progress = 100
    sm.set_state(SystemState.RUNNING)
    log.info("[retrain] Complete -- resuming RUNNING state")


# ------------------------------------------------------------------------------
# Command handler (called each cycle from the main loop)
# ------------------------------------------------------------------------------

def _process_commands(sm, zmq, ready_symbols, raw_dir, model_manager, cfg,
                       trailing=None, risk=None):
    """Drain the command queue and act on each command."""
    try:
        while True:
            cmd = sm.command_queue.get_nowait()
            if cmd == Command.PAUSE:
                if sm.state == SystemState.RUNNING:
                    sm.set_state(SystemState.PAUSED)
                    log.info(">>> PAUSED by operator")

            elif cmd == Command.RESUME:
                if sm.state == SystemState.PAUSED:
                    sm.set_state(SystemState.RUNNING)
                    log.info(">>> RESUMED by operator")

            elif cmd == Command.SHUTDOWN:
                log.info(">>> SHUTDOWN by operator")
                SHUTDOWN_FLAG.set()
                sm.shutdown_flag.set()

            elif cmd == Command.RETRAIN:
                if sm.state in (SystemState.RUNNING, SystemState.PAUSED):
                    log.info(">>> RETRAIN requested by operator")
                    t = threading.Thread(
                        target=_do_retrain,
                        args=(list(ready_symbols), raw_dir, model_manager, cfg),
                        daemon=True, name="retrain",
                    )
                    t.start()

            elif cmd == Command.REFRESH_DATA:
                log.info(">>> DATA REFRESH requested by operator")
                for sym in ready_symbols:
                    zmq.send_command("EXPORT_HISTORY", symbol=sym, years=1)

            elif cmd == Command.CLOSE_PROFITABLE:
                log.info(">>> CLOSE PROFITABLE by operator")
                _close_trades_by_condition(sm, zmq, trailing, risk, only_profitable=True)

            elif cmd == Command.CLOSE_LOSING:
                log.info(">>> CLOSE LOSING by operator")
                _close_trades_by_condition(sm, zmq, trailing, risk, only_losing=True)

            elif cmd == Command.CLOSE_ALL:
                log.info(">>> CLOSE ALL by operator")
                _close_trades_by_condition(sm, zmq, trailing, risk)

    except queue.Empty:
        pass


def _close_trades_by_condition(sm, zmq, trailing, risk,
                                only_profitable=False, only_losing=False):
    """Send CLOSE to MT5 for matching active trades."""
    for ticket, trade in list(sm.active_trades.items()):
        if only_profitable and trade.profit_pips <= 0:
            continue
        if only_losing and trade.profit_pips >= 0:
            continue
        zmq.send_trade(
            symbol=trade.symbol,
            action="CLOSE",
            lot=trade.lot,
            sl=0.0,
            tp=0.0,
            comment="manual_close",
            magic=ticket,
            ticket=ticket,
        )
        if trailing:
            trailing.unregister(ticket)
        if risk:
            risk.register_close(trade.symbol, ticket,
                                 trade.profit_pips)  # profit_pips as proxy
        log.info(f"[{trade.symbol}] Manual close sent ticket={ticket} "
                 f"pips={trade.profit_pips:.1f}")


# ------------------------------------------------------------------------------
# LIVE MODE
# ------------------------------------------------------------------------------

def run_live(cfg: dict, symbols: list, use_dashboard: bool = True):
    sm = get_state_manager()
    sm.set_state(SystemState.IDLE)

    log.info("=" * 60)
    log.info("  MT5_Bot_mk2  |  LIVE MODE")
    log.info("=" * 60)

    # ── 1. IPC paths (needed before dashboard launch) ────────────────
    state_path, cmd_path = get_ipc_paths()

    # Clear stale commands from previous run to prevent immediate shutdown
    if os.path.exists(cmd_path):
        try:
            os.remove(cmd_path)
        except OSError:
            pass

    # Write initial idle state so dashboard shows something immediately
    sm.training_progress = 3
    sm.set_current_report("Initializing MT5_Bot_mk2...\nStarting subsystems.")
    _push_state(sm, state_path)

    # ── 2. Dashboard (separate window) ──────────────────────────────
    if use_dashboard:
        try:
            from interface.dashboard import start_dashboard
            start_dashboard(state_path, cmd_path)
            log.info("Dashboard launched in new window -- use [P/R/T/D/S] keys")
            # Brief pause so the dashboard window has time to open and render
            time.sleep(1.5)
        except Exception as e:
            log.warning(f"Dashboard unavailable: {e} -- running headless")

    # ── 3. GPU/Acceleration ──────────────────────────────────────────
    sm.update_agent_status("ZMQ Bridge",    "initializing")
    sm.training_progress = 8
    sm.set_current_report("Step 1/8: Initializing GPU/CPU acceleration context...")
    _push_state(sm, state_path)
    accel = AccelerationContext(cfg.get("gpu", {}))
    sm.training_progress = 10
    _push_state(sm, state_path)

    # ── 4. ZMQ ──────────────────────────────────────────────────────
    sm.update_agent_status("ZMQ Bridge", "running", "connecting")
    sm.training_progress = 13
    sm.set_current_report("Step 2/8: Starting ZMQ bridge to MetaTrader 5...")
    _push_state(sm, state_path)

    router = MessageRouter()
    zmq = ZMQController(cfg.get("zmq", {}), on_message_cb=router.route)
    zmq.start()
    sm.zmq_connected = True
    sm.update_agent_status("ZMQ Bridge", "running", "connected")
    _push_state(sm, state_path)

    if _check_init_commands(sm, cmd_path):
        zm_stop_and_exit(zmq, sm)
        return

    # ── 5. Symbol resolver (live countdown in dashboard) ─────────────
    resolver = SymbolResolver(zmq)
    router.attach(symbol_resolver=resolver)

    log.info("Sending HANDSHAKE to EA...")
    zmq.send_command("HANDSHAKE", version="mk2")
    time.sleep(1.5)  # give EA time to respond before resolver starts
    _push_state(sm, state_path)

    sm.training_progress = 18
    _push_state(sm, state_path)
    log.info("Fetching Market Watch symbol list from MT5...")
    resolved_symbols = _resolver_with_countdown(
        resolver, symbols, timeout=30, sm=sm,
        state_path=state_path, cmd_path=cmd_path,
    )

    if SHUTDOWN_FLAG.is_set():
        zm_stop_and_exit(zmq, sm)
        return

    if not resolved_symbols:
        log.warning("Symbol resolver timed out -- falling back to configured list")
        resolved_symbols = symbols  # fall back so bot can still continue

    # ea_online = True only when the resolver got a real MARKETWATCH_LIST from EA.
    # Used by HistoricalExporter to skip the 60s MT5 wait when EA is offline.
    ea_online = bool(resolver._broker_symbols)
    sm.mt5_connected  = ea_online
    sm.symbols_active = len(resolved_symbols)
    sm.training_progress = 26
    sm.update_agent_status("ZMQ Bridge", "completed",
                           f"{len(resolved_symbols)} symbols" if ea_online else "offline — using local data")
    log.info(f"EA online={ea_online}  Using {len(resolved_symbols)} symbols")
    _push_state(sm, state_path)

    # ── 6. Tick buffers ──────────────────────────────────────────────
    sm.training_progress = 28
    sm.set_current_report("Step 4/8: Allocating tick buffers...")
    sm.update_agent_status("Tick Buffer", "running", "allocating")
    _push_state(sm, state_path)
    tick_buffers = TickBufferManager(resolved_symbols, cfg)
    router.attach(tick_buffers=tick_buffers)
    sm.training_progress = 30
    sm.update_agent_status("Tick Buffer", "completed", f"{len(resolved_symbols)} buffers")
    _push_state(sm, state_path)

    # ── 7. Historical data export (live progress in dashboard) ────────
    exporter = HistoricalExporter(zmq, cfg)
    exporter.ea_online = ea_online   # skip MT5 export waits when EA is offline
    router.attach(exporter=exporter)

    log.info("Requesting historical data export...")
    sm.training_progress = 32
    sm.update_agent_status("Data Exporter", "running", "starting")
    _push_state(sm, state_path)

    ready_symbols = _export_with_progress(
        exporter, resolved_symbols, sm=sm,
        state_path=state_path, cmd_path=cmd_path,
    )

    if SHUTDOWN_FLAG.is_set():
        zm_stop_and_exit(zmq, sm)
        return

    if not ready_symbols:
        log.error("No symbols have historical data. Cannot proceed.")
        zm_stop_and_exit(zmq, sm)
        return

    raw_dir = cfg.get("data", {}).get("raw_dir", "data/raw")

    # ── 8. Model training / loading ──────────────────────────────────
    sm.set_state(SystemState.TRAINING)
    sm.training_progress = 62
    sm.set_current_report("Step 6/8: Loading ensemble models...")
    sm.update_agent_status("Trainer", "running", "loading models")
    _push_state(sm, state_path)

    model_manager = ModelManager(cfg, accel)
    for sym in ready_symbols:
        model_manager.load_or_create(sym)
        if _check_init_commands(sm, cmd_path):
            zm_stop_and_exit(zmq, sm)
            return

    needs_train = [s for s in ready_symbols if model_manager.needs_retrain(s)]
    if needs_train:
        total = len(needs_train)
        log.info(f"Training ensemble models for {total} symbols...")
        for i, sym in enumerate(needs_train):
            if SHUTDOWN_FLAG.is_set():
                break
            sm.training_symbol   = sym
            # Map symbol index to 65-90% range
            sm.training_progress = 65 + int(i / total * 25)
            sm.set_current_report(
                f"Step 7/8: Training ensemble models\n"
                f"Symbol {i+1}/{total}: {sym}\n"
                f"Overall progress: {sm.training_progress}%"
            )
            _push_state(sm, state_path)
            _check_init_commands(sm, cmd_path)

            min_s  = cfg.get("model", {}).get("min_train_samples", 5000)
            window = cfg.get("model", {}).get("input_window", 60)
            train_symbol(sym, raw_dir, model_manager, min_s, window, cfg)

        sm.training_symbol   = ""
        sm.training_progress = 90
        sm.update_agent_status("Trainer", "completed", f"{total} symbols")
    else:
        sm.training_progress = 90
        sm.update_agent_status("Trainer", "completed", "all cached")

    if SHUTDOWN_FLAG.is_set():
        zm_stop_and_exit(zmq, sm)
        return

    _push_state(sm, state_path)

    # ── 9. Pre-flight gate ───────────────────────────────────────────
    #   Operator reviews readiness and confirms before trading starts.
    #   Auto-starts after 60 seconds if no input.
    auto_secs = cfg.get("startup", {}).get("preflight_auto_start_secs", 60)
    if not _preflight_gate(sm, state_path, cmd_path, ready_symbols,
                           auto_start_secs=auto_secs):
        zm_stop_and_exit(zmq, sm)
        return

    # ── 10. Inference & execution engines ────────────────────────────
    sm.training_progress = 100
    sm.set_state(SystemState.RUNNING)
    sm.update_agent_status("Ensemble",     "running")
    sm.update_agent_status("Trade Stacker","running")
    sm.update_agent_status("Micro Scalper","running")
    sm.update_agent_status("Trailing Mgr", "running")
    sm.update_agent_status("Risk Manager", "running")
    sm.update_agent_status("Portfolio Mgr","running")

    inference = InferenceEngine(model_manager, tick_buffers, cfg)
    risk      = RiskManager(cfg)
    risk.initialize(balance=10000.0)
    sizer    = PositionSizer(cfg)
    trailing = TrailingManager(zmq, cfg)
    stacker  = TradeStacker(zmq, risk, trailing, sizer, cfg)
    scalper  = MicroScalper(zmq, trailing, cfg)

    router.attach(risk_manager=risk, trailing_manager=trailing)

    # ── 11. Subscribe ticks ──────────────────────────────────────────
    zmq.send_command("GET_ACCOUNT_INFO")
    for sym in ready_symbols:
        zmq.request_symbol_info(sym)
        zmq.send_command("SUBSCRIBE_TICKS", symbol=sym)

    log.info(f"Live trading started -- {len(ready_symbols)} symbols active")
    log.info("Entering main inference/execution loop...")
    sm.set_current_report(
        f"Live Trading Active\n"
        f"Symbols: {len(ready_symbols)}\n"
        f"Confidence threshold: {cfg.get('inference',{}).get('confidence_threshold',0.90):.0%}\n"
        f"Stack threshold: {cfg.get('inference',{}).get('stack_confidence_threshold',0.97):.0%}"
    )
    _push_state(sm, state_path)

    # ── 12. Main loop ─────────────────────────────────────────────────
    cycle_interval      = 0.5
    last_retrain_check  = time.time()
    last_heartbeat      = time.time()
    last_daily_reset    = time.time()
    last_positions_sync = time.time()
    hb_interval         = cfg.get("zmq",   {}).get("heartbeat_interval",     5)
    retrain_interval    = cfg.get("model", {}).get("retrain_interval_hours", 4) * 3600
    positions_sync_interval = 10  # request position sync every 10 s

    while not SHUTDOWN_FLAG.is_set():
        t_cycle = time.perf_counter()

        # ── Read dashboard commands ──────────────────────────────────
        for raw_cmd in read_commands(cmd_path):
            try:
                if raw_cmd.startswith("SET_PARAM:"):
                    # Format: SET_PARAM:key:value
                    parts = raw_cmd.split(":", 2)
                    if len(parts) == 3:
                        _, key, val = parts
                        sm.update_runtime_param(key, val)
                        log.info(f">>> Runtime param updated: {key}={val}")
                else:
                    sm.command_queue.put_nowait(Command[raw_cmd])
            except (KeyError, Exception):
                pass

        _process_commands(sm, zmq, ready_symbols, raw_dir, model_manager, cfg,
                          trailing=trailing, risk=risk)

        if SHUTDOWN_FLAG.is_set():
            break

        # ── Risk shutdown ────────────────────────────────────────────
        if risk.is_shutdown:
            log.critical("Risk shutdown triggered -- stopping bot")
            sm.set_state(SystemState.ERROR, "Risk drawdown limit hit")
            break

        # ── Daily reset ──────────────────────────────────────────────
        now = time.time()
        if now - last_daily_reset > 86400:
            risk.reset_daily()
            last_daily_reset = now

        # ── Heartbeat ────────────────────────────────────────────────
        if now - last_heartbeat > hb_interval:
            zmq.send_heartbeat()
            zmq.send_command("GET_ACCOUNT_INFO")
            sm.last_heartbeat = now
            last_heartbeat = now

        # ── Position sync (detect manual closes in MT5) ──────────────
        if now - last_positions_sync > positions_sync_interval:
            zmq.send_command("GET_POSITIONS")
            last_positions_sync = now

        # ── Scheduled retrain ────────────────────────────────────────
        if now - last_retrain_check > retrain_interval:
            if sm.state == SystemState.RUNNING:
                needs = [s for s in ready_symbols if model_manager.needs_retrain(s)]
                if needs:
                    t = threading.Thread(
                        target=_do_retrain,
                        args=(needs, raw_dir, model_manager, cfg),
                        daemon=True, name="retrain_sched",
                    )
                    t.start()
            last_retrain_check = now

        # ── Update dashboard metrics ─────────────────────────────────
        sm.drawdown_pct         = risk.daily_drawdown_pct
        sm.inference_latency_ms = inference.avg_latency_ms

        # ── Inference + execution (only when RUNNING) ────────────────
        if sm.state == SystemState.RUNNING:
            signals    = inference.run_batch(ready_symbols)
            tradeable  = inference.filter_tradeable(signals)
            signal_map = {s.symbol: s for s in signals}

            balance = float(router.account_info.get("balance", 10000.0))
            dd_pct  = risk.daily_drawdown_pct

            for sig in tradeable:
                buf = tick_buffers.get(sig.symbol)
                if buf:
                    stacker.execute_signal(sig, buf, balance, dd_pct)

            trailing.update_all(tick_buffers, inference)

            reentry_fns = {}
            for sig in tradeable:
                buf = tick_buffers.get(sig.symbol)
                if buf:
                    def _make_reentry(s=sig, b=buf):
                        stacker.execute_signal(s, b, balance, dd_pct)
                    reentry_fns[sig.symbol] = _make_reentry
            scalper.run_cycle(tick_buffers, trailing, signal_map, reentry_fns)

            for ticket, state in list(trailing._trades.items()):
                sig = signal_map.get(state.symbol)
                if sig and trailing.should_early_exit(ticket, sig.action, sig.confidence):
                    zmq.send_trade(
                        symbol=state.symbol, action="CLOSE",
                        lot=state.lot, sl=0.0, tp=0.0,
                        comment=f"early_exit_t{ticket}",
                        magic=state.ticket,
                        ticket=ticket,
                    )
                    trailing.unregister(ticket)
                    sm.register_trade_close(ticket, 0.0, reason="early_exit")

        else:
            # PAUSED -- still maintain trailing on open positions
            trailing.update_all(tick_buffers, inference)

        # ── Publish state to dashboard ───────────────────────────────
        _push_state(sm, state_path)

        # ── Pace loop ────────────────────────────────────────────────
        elapsed = time.perf_counter() - t_cycle
        sleep_t = max(0.0, cycle_interval - elapsed)
        if sleep_t > 0:
            SHUTDOWN_FLAG.wait(timeout=sleep_t)

    log.info("Main loop exited -- shutting down...")
    sm.set_state(SystemState.SHUTDOWN)
    sm.shutdown_flag.set()
    _push_state(sm, state_path)
    time.sleep(0.8)
    zmq.stop()
    log.info("MT5_Bot_mk2 stopped.")


def zm_stop_and_exit(zmq, sm):
    sm.set_state(SystemState.ERROR)
    sm.shutdown_flag.set()
    zmq.stop()


# ------------------------------------------------------------------------------
# TRAIN MODE
# ------------------------------------------------------------------------------

def run_train(cfg: dict, symbols: list):
    sm = get_state_manager()
    log.info("=" * 60)
    log.info("  MT5_Bot_mk2  |  TRAIN MODE")
    log.info("=" * 60)

    raw_dir   = cfg.get("data", {}).get("raw_dir", "data/raw")
    available = list_available_symbols(raw_dir)
    if not available:
        log.error(f"No CSV data found in {raw_dir}. Run --mode live first.")
        return

    from ingestion.symbol_resolver import _strip_suffix
    canonical_set = {s.upper() for s in symbols}
    train_targets = [
        s for s in available
        if s.upper() in canonical_set or _strip_suffix(s) in canonical_set
    ]
    if not train_targets:
        train_targets = available
        log.info(f"No configured symbols matched -- training all {len(train_targets)}")

    accel         = AccelerationContext(cfg.get("gpu", {}))
    model_manager = ModelManager(cfg, accel)
    for sym in train_targets:
        model_manager.load_or_create(sym)

    sm.set_state(SystemState.TRAINING)
    results = train_all_symbols(train_targets, raw_dir, model_manager, cfg)
    sm.set_state(SystemState.IDLE)

    ok = [r for r in results if r.get("status") == "ok"]
    log.info(f"\nTraining summary: {len(ok)}/{len(results)} succeeded")
    for r in ok:
        log.info(
            f"  {r['symbol']:12s} "
            f"win_rate={r.get('win_rate',0):.3f}  "
            f"expectancy={r.get('expectancy',0):+.3f}  "
            f"pf={r.get('profit_factor',0):.2f}  "
            f"trades={r.get('trade_count',0)}  "
            f"conf={r.get('avg_confidence',0):.3f}"
        )


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MT5_Bot_mk2")
    parser.add_argument("--mode",         choices=["live", "train", "backtest"],
                        default="live")
    parser.add_argument("--config",       default="config/settings.yaml")
    parser.add_argument("--symbols",      default="config/symbols.yaml")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="Disable the Rich TUI dashboard")
    args = parser.parse_args()

    cfg     = load_config(args.config)
    symbols = load_symbols(args.symbols)

    os.makedirs("logs", exist_ok=True)

    log.info(f"MT5_Bot_mk2 starting in [{args.mode}] mode")
    log.info(f"Symbols: {len(symbols)} configured")

    if args.mode == "live":
        run_live(cfg, symbols, use_dashboard=not args.no_dashboard)
    elif args.mode == "train":
        run_train(cfg, symbols)
    elif args.mode == "backtest":
        log.info("Backtest mode not yet implemented")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
