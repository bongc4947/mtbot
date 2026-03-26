"""
TradingAgents-style Rich panel builders for MT5_Bot_mk2 dashboard.
Layout mirrors the TradingAgents UI: Progress | Messages | Current Report | Status Bar
"""
import time
from rich.panel   import Panel
from rich.table   import Table
from rich.text    import Text
from rich.layout  import Layout
from rich.align   import Align
from rich         import box

from interface.state_manager import StateManager, SystemState

# ── Color maps ────────────────────────────────────────────────────────────────

_STATUS_STYLE = {
    "idle":         "dim",
    "initializing": "yellow",
    "running":      "bright_green",
    "completed":    "green",
    "training":     "cyan",
    "error":        "bold red",
    "paused":       "yellow",
}

_TYPE_STYLE = {
    "SIGNAL": "yellow",
    "TRADE":  "bright_green",
    "INFO":   "white",
    "WARN":   "yellow",
    "ERROR":  "bold red",
    "MODEL":  "cyan",
    "DEBUG":  "dim",
}

_MSG_CATEGORY = {
    "trade_stacker":    "TRADE",
    "micro_scalper":    "TRADE",
    "trailing_manager": "TRADE",
    "inference_engine": "SIGNAL",
    "trainer":          "MODEL",
    "ensemble":         "MODEL",
    "lgbm_model":       "MODEL",
    "mlp_model":        "MODEL",
}

_SYSTEM_STATE_COLOR = {
    SystemState.IDLE:     "yellow",
    SystemState.TRAINING: "cyan",
    SystemState.RUNNING:  "bright_green",
    SystemState.PAUSED:   "yellow",
    SystemState.ERROR:    "red",
    SystemState.SHUTDOWN: "red",
}


# ── Header ────────────────────────────────────────────────────────────────────

def build_header_title(sm: StateManager) -> Text:
    sc = _SYSTEM_STATE_COLOR.get(sm.state, "white")
    t = Text(justify="center")
    t.append("MT5_Bot_mk2", style="bold cyan")
    t.append(" | ", style="dim")
    t.append("Live Ensemble Trading System", style="cyan")
    t.append("  [", style="dim")
    t.append(sm.state.value, style=f"bold {sc}")
    t.append("]", style="dim")
    return t


# ── Progress panel (left) ─────────────────────────────────────────────────────

def build_progress(sm: StateManager) -> Panel:
    tbl = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style="bold cyan",
        expand=True,
        padding=(0, 1),
        show_lines=False,
    )
    tbl.add_column("Team",   style="cyan",  ratio=3, no_wrap=True)
    tbl.add_column("Agent",  style="cyan",  ratio=3, no_wrap=True)
    tbl.add_column("Status", justify="right", ratio=2, no_wrap=True)

    last_team = ""
    for name, info in sm.agent_status.items():
        team   = info["team"]
        status = info["status"]
        detail = info.get("info", "")

        team_cell = team if team != last_team else ""
        last_team = team

        sty = _STATUS_STYLE.get(status, "white")
        status_txt = Text()
        status_txt.append(status, style=sty)
        if detail:
            status_txt.append(f" {detail[:12]}", style="dim")

        tbl.add_row(team_cell, name, status_txt)

    return Panel(
        tbl,
        title="[bold cyan]Progress[/bold cyan]",
        border_style="cyan",
    )


# ── Messages & Signals panel (right) ──────────────────────────────────────────

def build_messages(sm: StateManager, n: int = 20) -> Panel:
    tbl = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style="bold cyan",
        expand=True,
        padding=(0, 1),
        show_lines=False,
    )
    tbl.add_column("Time",    style="dim",   ratio=2, no_wrap=True)
    tbl.add_column("Type",    ratio=2,        no_wrap=True)
    tbl.add_column("Content", ratio=8,        no_wrap=True)

    entries = list(sm.log_buffer)[-n:]
    for e in reversed(entries):  # newest first like TradingAgents
        ts  = time.strftime("%H:%M:%S", time.localtime(e.timestamp))
        cat = _MSG_CATEGORY.get(e.name, e.level[:5])
        cat_sty = _TYPE_STYLE.get(cat, "white")
        # Truncate long messages
        msg = e.message[:80] + "..." if len(e.message) > 80 else e.message
        type_txt = Text(cat, style=cat_sty)
        tbl.add_row(ts, type_txt, msg)

    return Panel(
        tbl,
        title="[bold cyan]Messages & Signals[/bold cyan]",
        border_style="cyan",
    )


# ── Current Report panel (bottom main) ────────────────────────────────────────

def build_current_report(sm: StateManager) -> Panel:
    txt = Text(overflow="fold")

    # Title line
    sc = _SYSTEM_STATE_COLOR.get(sm.state, "white")
    txt.append("Portfolio Trading Decision\n", style="bold white")
    txt.append(f"State: ", style="dim")
    txt.append(f"{sm.state.value}  ", style=f"bold {sc}")

    zmq = "connected" if sm.zmq_connected else "disconnected"
    mt5 = "connected" if sm.mt5_connected else "disconnected"
    zc  = "bright_green" if sm.zmq_connected else "red"
    mc  = "bright_green" if sm.mt5_connected else "red"
    txt.append("ZMQ [", style="dim")
    txt.append(zmq, style=zc)
    txt.append("]  MT5 [", style="dim")
    txt.append(mt5, style=mc)
    txt.append("]\n\n", style="dim")

    # Metrics summary
    pnl_col  = "bright_green" if sm.daily_pnl >= 0 else "red"
    pnl_sign = "+" if sm.daily_pnl >= 0 else ""
    wr_col   = "bright_green" if sm.win_rate >= 0.60 else ("yellow" if sm.win_rate >= 0.50 else "red")
    dd_col   = "bright_green" if sm.drawdown_pct < 2 else ("yellow" if sm.drawdown_pct < 5 else "red")
    pf       = sm.profit_factor

    txt.append(f"  Balance: ${sm.balance:,.2f}  |  Equity: ${sm.equity:,.2f}  |  ", style="white")
    txt.append(f"Daily P/L: {pnl_sign}${sm.daily_pnl:,.2f}", style=pnl_col)
    txt.append("\n")

    txt.append(f"  Win Rate: ", style="dim")
    txt.append(f"{sm.win_rate:.1%}", style=wr_col)
    txt.append(f"  |  Profit Factor: {pf:.2f}  |  Drawdown: ", style="dim")
    txt.append(f"{sm.drawdown_pct:.2f}%", style=dd_col)
    txt.append(f"  |  Trades: {sm.total_trades}  |  Active: {len(sm.active_trades)}\n")

    # Ensemble weights
    ew = sm.ensemble_weights
    txt.append("\n  Ensemble: ", style="dim")
    txt.append(f"N-HiTS {ew.get('nhits', 0.30):.0%}  ", style="cyan")
    txt.append(f"LightGBM {ew.get('lgbm', 0.40):.0%}  ", style="yellow")
    txt.append(f"MLP {ew.get('mlp', 0.30):.0%}", style="bright_green")

    # Runtime config (live-editable)
    rc = getattr(sm, "runtime_config", {})
    mode       = rc.get("mode", "balanced")
    conf       = rc.get("confidence", 0.90)
    max_trades = rc.get("max_trades_10min", 30)
    MODE_COLORS = {"scalp": "bright_cyan", "balanced": "bright_green", "swing": "yellow"}
    mc = MODE_COLORS.get(mode, "white")
    txt.append("\n\n  [LIVE CONFIG]  ", style="bold white")
    txt.append("Confidence: ", style="dim")
    txt.append(f"{conf:.2f}  ", style="bold cyan")
    txt.append("Max/10min: ", style="dim")
    txt.append(f"{max_trades}  ", style="bold cyan")
    txt.append("Mode: ", style="dim")
    txt.append(f"{mode.upper()}", style=f"bold {mc}")
    txt.append("   ", style="dim")
    txt.append("[1/-] conf  [3/4] trades  [5/6] mode", style="dim yellow")

    # Training info
    if sm.state == SystemState.TRAINING and sm.training_symbol:
        prog = sm.training_progress
        bar  = "#" * (prog // 5) + "." * (20 - prog // 5)
        txt.append(f"\n\n  Training: ", style="dim")
        txt.append(f"{sm.training_symbol} ", style="cyan")
        txt.append(f"[{bar}] {prog}%", style="cyan")

    # Current report / analysis text
    if sm.current_report:
        txt.append(f"\n\n  Latest Analysis:\n", style="dim")
        # Wrap at 100 chars
        for line in sm.current_report.splitlines()[:6]:
            txt.append(f"  {line}\n", style="white")

    # Active trades summary
    active = list(sm.active_trades.values())[:8]
    if active:
        txt.append("\n  Active Trades: ", style="dim")
        for t in active:
            pp = getattr(t, "profit_pips", 0.0)
            pp_col = "bright_green" if pp >= 0 else "red"
            txt.append(f"{t.symbol} {t.direction} ", style="bold")
            txt.append(f"{pp:+.1f}p  ", style=pp_col)

    # Rejected trades (last 5)
    rejected = list(getattr(sm, "rejected_trades", []))[:5]
    if rejected:
        txt.append("\n  Rejected: ", style="dim red")
        for r in rejected:
            sym = getattr(r, "symbol", "?")
            reason = getattr(r, "reason", "?")
            txt.append(f"{sym}:{reason}  ", style="dim red")

    # Recent closed trades (last 5)
    closed = list(sm.recent_closed)[:5]
    if closed:
        txt.append("\n  Recent Closed: ", style="dim")
        for c in closed:
            profit = getattr(c, "profit", 0.0)
            reason = getattr(c, "close_reason", "")
            pc = "bright_green" if profit >= 0 else "red"
            label = f"({reason})" if reason else ""
            txt.append(f"{c.symbol} {c.direction} ", style="bold")
            txt.append(f"{profit:+.2f}{label}  ", style=pc)

    # Recent model info
    if sm.model_info:
        txt.append("\n  Symbol Performance: ", style="dim")
        items = list(sm.model_info.items())[:5]
        for sym, info in items:
            wr   = info.get("win_rate", 0.0)
            wc   = "bright_green" if wr >= 0.60 else ("yellow" if wr >= 0.50 else "red")
            exp  = info.get("expectancy", 0.0)
            txt.append(f"{sym} ", style="bold")
            txt.append(f"WR:{wr:.0%} ", style=wc)
            txt.append(f"E:{exp:+.2f}  ", style="dim")

    return Panel(
        txt,
        title="[bold cyan]Current Analysis[/bold cyan]",
        border_style="cyan",
    )


# ── Status bar (bottom strip) ─────────────────────────────────────────────────

def build_status_bar(sm: StateManager) -> Text:
    t10 = sm.trades_10min
    tpm = sm.trades_per_minute
    sig = sm.signal_count
    lat = sm.inference_latency_ms

    t = Text(justify="center")
    t.append(f"Signals: {sig}", style="cyan")
    t.append("  |  ", style="dim")
    t.append(f"Trades/10min: {t10}", style="bright_green" if t10 >= 15 else "yellow")
    t.append("  |  ", style="dim")
    t.append(f"Trades/min: {tpm:.1f}", style="cyan")
    t.append("  |  ", style="dim")
    wr_col = "bright_green" if sm.win_rate >= 0.60 else "yellow"
    t.append(f"Win Rate: {sm.win_rate:.1%}", style=wr_col)
    t.append("  |  ", style="dim")
    pnl_col = "bright_green" if sm.daily_pnl >= 0 else "red"
    sign = "+" if sm.daily_pnl >= 0 else ""
    t.append(f"P&L: {sign}${sm.daily_pnl:,.2f}", style=pnl_col)
    t.append("  |  ", style="dim")
    t.append(f"Latency: {lat:.1f}ms", style="cyan")
    t.append("  |  ", style="dim")
    t.append("Keys: ", style="dim")
    for k, lbl in [("P","Pause"),("R","Resume"),("T","Retrain"),("D","Data"),("S","Shutdown"),
                   ("C","Close+"),("L","Close-"),("X","CloseAll")]:
        t.append(f"[{k}]", style="bold yellow")
        t.append(f"{lbl} ", style="dim")
    return t
