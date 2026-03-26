"""
Routes incoming ZMQ messages to appropriate handlers.
"""
import time
from utils.logger import get_logger
from interface.state_manager import get_state_manager

log = get_logger("message_router")


class MessageRouter:
    def __init__(self):
        self._tick_buffers = None
        self._exporter = None
        self._risk_manager = None
        self._trailing_manager = None
        self._symbol_resolver = None
        self._account_info = {}

    def attach(self, tick_buffers=None, exporter=None,
               risk_manager=None, trailing_manager=None,
               symbol_resolver=None):
        if tick_buffers:
            self._tick_buffers = tick_buffers
        if exporter:
            self._exporter = exporter
        if risk_manager:
            self._risk_manager = risk_manager
        if trailing_manager:
            self._trailing_manager = trailing_manager
        if symbol_resolver:
            self._symbol_resolver = symbol_resolver

    def route(self, msg: dict):
        mtype = msg.get("type", "")
        try:
            if mtype == "TICK":
                self._on_tick(msg)
            elif mtype == "MARKETWATCH_LIST":
                self._on_marketwatch_list(msg)
            elif mtype == "EXPORT_DONE":
                self._on_export_done(msg)
            elif mtype == "TRADE_CONFIRM":
                self._on_trade_confirm(msg)
            elif mtype == "TRADE_CLOSED":
                self._on_trade_closed(msg)
            elif mtype == "POSITIONS_SYNC":
                self._on_positions_sync(msg)
            elif mtype == "ACCOUNT_INFO":
                self._on_account_info(msg)
            elif mtype == "SYMBOL_INFO":
                self._on_symbol_info(msg)
            elif mtype == "HANDSHAKE_ACK":
                log.debug("Handshake acknowledged by EA")
                sm = get_state_manager()
                sm.mt5_connected  = True
                sm.zmq_connected  = True
            elif mtype == "HEARTBEAT":
                pass
            else:
                log.debug(f"Unknown message type: {mtype}")
        except Exception as e:
            log.error(f"Router error on {mtype}: {e}")

    def _on_tick(self, msg: dict):
        symbol = msg.get("symbol")
        if not symbol or not self._tick_buffers:
            return
        self._tick_buffers.push_tick(
            symbol=symbol,
            bid=float(msg.get("bid", 0)),
            ask=float(msg.get("ask", 0)),
            spread=float(msg.get("spread", 0)),
            volume=float(msg.get("volume", 0)),
            ts=float(msg.get("ts", 0))
        )
        get_state_manager().last_tick_time = time.time()

    def _on_marketwatch_list(self, msg: dict):
        if not self._symbol_resolver:
            return
        symbols = msg.get("symbols", [])
        if isinstance(symbols, list):
            self._symbol_resolver.on_marketwatch_list(symbols)

    def _on_export_done(self, msg: dict):
        if not self._exporter:
            return
        symbol = msg.get("symbol", "")
        success = bool(msg.get("success", False))
        self._exporter.on_export_done(symbol, success)

    def _on_trade_confirm(self, msg: dict):
        if not self._risk_manager:
            return
        symbol = msg.get("symbol", "")
        ticket = int(msg.get("ticket", 0))
        magic  = int(msg.get("magic", 0))
        if ticket and magic:
            # Replace the magic placeholder with the real MT5 ticket (no double-count)
            self._risk_manager.replace_ticket(symbol, magic, ticket)
            if self._trailing_manager:
                self._trailing_manager.replace_ticket(magic, ticket)
            get_state_manager().replace_trade_ticket(magic, ticket)
            log.info(f"[{symbol}] Trade confirmed — ticket={ticket} magic={magic}")

    def _on_trade_closed(self, msg: dict):
        if not self._risk_manager:
            return
        symbol = msg.get("symbol", "")
        ticket = int(msg.get("ticket", 0))
        profit = float(msg.get("profit", 0))
        magic  = int(msg.get("magic", 0))
        self._risk_manager.register_close(symbol, ticket, profit)
        # Also try to close by magic in case ticket mapping wasn't updated yet
        if magic and magic != ticket:
            self._risk_manager.register_close(symbol, magic, profit)
        if self._trailing_manager:
            self._trailing_manager.unregister(ticket)
            if magic:
                self._trailing_manager.unregister(magic)
        reason = msg.get("reason", "")  # e.g. "tp", "sl", "manual"
        get_state_manager().register_trade_close(ticket, profit, reason=reason)
        log.info(f"[{symbol}] Trade closed — ticket={ticket} magic={magic} profit={profit:.2f} reason={reason}")

    def _on_positions_sync(self, msg: dict):
        """Reconcile Python's open trade list against MT5's live positions."""
        if not self._risk_manager:
            return
        positions = msg.get("positions", [])
        self._risk_manager.sync_positions(positions)
        # Also sync trailing manager: remove stale tickets
        if self._trailing_manager:
            live_tickets = {int(p.get("ticket", 0)) for p in positions if p.get("ticket")}
            self._trailing_manager.sync_positions(live_tickets)
        sm = get_state_manager()
        # Build profit map from live positions
        live_profit = {int(p.get("ticket", 0)): float(p.get("profit", 0.0))
                       for p in positions if p.get("ticket")}
        live_tickets = set(live_profit.keys())
        for ticket in list(sm.active_trades.keys()):
            if ticket >= 10000 and ticket not in live_tickets:
                # Trade was manually closed in MT5 — use last known profit_pips
                trade = sm.active_trades.get(ticket)
                profit = 0.0
                if trade and trade.current_price > 0 and trade.open_price > 0:
                    d = 1 if trade.direction == "BUY" else -1
                    profit = (trade.current_price - trade.open_price) * d * trade.lot * 100000
                sm.register_trade_close(ticket, profit, reason="manual")
                if self._risk_manager:
                    self._risk_manager.register_close(
                        trade.symbol if trade else "", ticket, profit)
                if self._trailing_manager:
                    self._trailing_manager.unregister(ticket)
                log.info(f"Reconciled: manually closed ticket={ticket} profit≈{profit:.2f}")

    def _on_account_info(self, msg: dict):
        balance = float(msg.get("balance", 0))
        equity  = float(msg.get("equity",  balance))
        self._account_info = msg
        if self._risk_manager and balance > 0:
            self._risk_manager.update_balance(balance)
        get_state_manager().update_account(balance, equity)

    def _on_symbol_info(self, msg: dict):
        symbol = msg.get("symbol", "")
        if symbol and self._tick_buffers:
            self._tick_buffers.add_symbol(symbol)
        log.debug(f"Symbol info received: {symbol}")

    @property
    def account_info(self) -> dict:
        return self._account_info
