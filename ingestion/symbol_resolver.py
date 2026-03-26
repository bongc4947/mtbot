"""
Symbol resolver: fetches the broker's exact Market Watch symbol names from MT5
and maps them to the canonical names defined in symbols.yaml.

Why this matters:
  Brokers append suffixes to symbol names (e.g. EURUSDm, EURUSD.i, EURUSD#,
  XAUUSDz). Using a hardcoded list breaks on those brokers. This resolver pulls
  the live list first, then fuzzy-matches so the rest of the pipeline always
  uses the broker-exact name.

Flow:
  Python → CMD GET_MARKETWATCH → EA
  EA     → MARKETWATCH_LIST    → Python (list of exact broker symbol strings)
  SymbolResolver.resolve(canonical_list) → list of broker-exact names
"""
import re
import threading
from utils.logger import get_logger

log = get_logger("symbol_resolver")

# Suffix patterns brokers commonly add after the base currency pair.
# Order matters: try longer patterns first to avoid partial stripping.
_SUFFIX_PATTERNS = re.compile(
    r'[mMzZiIrRsS]$'           # single trailing letter suffix (EURUSDm)
    r'|[#\+\-]$'               # special char suffix
    r'|\.[a-zA-Z]{1,3}$'       # dot + up to 3 letters (EURUSD.i, EURUSD.raw)
    r'|\d+$'                   # trailing digits (EURUSD2)
)


def _strip_suffix(broker_name: str) -> str:
    """
    Iteratively strip known broker suffixes until the name stabilises.
    Returns the stripped base name (uppercase).
    """
    name = broker_name.upper()
    for _ in range(4):          # max 4 stripping passes
        stripped = _SUFFIX_PATTERNS.sub("", name)
        if stripped == name or len(stripped) < 3:
            break
        name = stripped
    return name


def _match_score(canonical: str, broker: str) -> int:
    """
    Return a match score (higher = better).
    -1 means no match.
    """
    c = canonical.upper()
    b = broker.upper()
    if c == b:
        return 100                      # exact match
    if _strip_suffix(b) == c:
        suffix_len = len(b) - len(c)
        return max(1, 90 - suffix_len)  # suffix match, penalise longer suffixes
    return -1


class SymbolResolver:
    def __init__(self, zmq_ctrl):
        self.zmq = zmq_ctrl
        self._event = threading.Event()
        self._broker_symbols: list[str] = []

    # ------------------------------------------------------------------
    # Called by MessageRouter when MARKETWATCH_LIST arrives
    # ------------------------------------------------------------------
    def on_marketwatch_list(self, symbols: list[str]):
        self._broker_symbols = symbols
        self._event.set()
        log.info(f"Received {len(symbols)} symbols from Market Watch")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fetch(self, timeout: int = 30) -> list[str]:
        """
        Request and wait for the broker's Market Watch symbol list.
        Returns the raw broker symbol list, or [] on timeout.
        """
        self._event.clear()
        self._broker_symbols = []
        self.zmq.send_command("GET_MARKETWATCH")
        if not self._event.wait(timeout=timeout):
            log.error(f"Timed out waiting for Market Watch list after {timeout}s")
            return []
        return self._broker_symbols

    def resolve(self, canonical_symbols: list[str], timeout: int = 30) -> list[str]:
        """
        Fetch the broker list, then map each canonical name in
        canonical_symbols to its broker-exact equivalent.

        Returns a list of broker-exact names in the same relative order
        as canonical_symbols. Unmatched canonicals are logged and skipped.
        """
        broker_list = self.fetch(timeout=timeout)
        if not broker_list:
            log.warning("No broker symbols received — falling back to canonical names")
            return canonical_symbols

        log.info(f"Resolving {len(canonical_symbols)} canonical symbols "
                 f"against {len(broker_list)} broker symbols...")

        resolved = []
        unmatched = []

        for canonical in canonical_symbols:
            best_broker = None
            best_score = -1

            for broker in broker_list:
                score = _match_score(canonical, broker)
                if score > best_score:
                    best_score = score
                    best_broker = broker

            if best_broker is not None and best_score >= 0:
                if best_broker != canonical:
                    log.info(f"  {canonical:12s} -> {best_broker}  (score={best_score})")
                else:
                    log.debug(f"  {canonical} -> exact match")
                resolved.append(best_broker)
            else:
                unmatched.append(canonical)

        if unmatched:
            log.warning(f"No broker match found for: {unmatched}")

        log.info(f"Resolved {len(resolved)}/{len(canonical_symbols)} symbols")
        return resolved
