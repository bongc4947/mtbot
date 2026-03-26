"""
Reference spread and pip constants derived from:
  Market Watch 20260325 132856.csv  (AvaTrade MT5, live session)

SYMBOL_SPREADS : ea_pips (same unit the EA sends in tick data col-2)
  Formula: ea_pips = csv_spread_points / 10
  (MT5 5-digit convention: 1 pip = 10 points for all instruments)

SYMBOL_PIP : pip_mult — price units per pip
  pip_mult = (ask - bid) / ea_pips  (verified live per symbol)

Usage:
  from utils.symbol_info import get_ref_spread, get_ref_pip
  ref_spr = get_ref_spread("EURUSD")   # → 0.8
  ref_pip = get_ref_pip("BTCUSD")      # → 0.10
"""

# Symbol → reference spread in ea_pips (Market Watch 20260325)
SYMBOL_SPREADS: dict[str, float] = {
    # ── Forex majors ──────────────────────────────────────────────────
    "EURUSD":      0.8,
    "GBPUSD":      1.2,
    "USDJPY":      1.3,
    "USDCHF":      1.3,
    "AUDUSD":      0.9,
    "USDCAD":      1.8,
    "NZDUSD":      2.1,
    # ── Forex minors ──────────────────────────────────────────────────
    "EURGBP":      1.2,
    "EURJPY":      1.8,
    "EURCHF":      1.6,
    "EURCAD":      2.3,
    "EURAUD":      2.5,
    "EURNZD":      3.3,
    "GBPJPY":      2.2,
    "GBPCHF":      2.2,
    "GBPCAD":      3.3,
    "GBPAUD":      3.2,
    "GBPNZD":      4.9,
    "AUDJPY":      1.9,
    "AUDCAD":      2.4,
    "AUDCHF":      3.2,
    "AUDNZD":      2.4,
    "CADJPY":      2.3,
    "CADCHF":      1.8,
    "CHFJPY":      2.5,
    "NZDJPY":      2.5,
    "NZDCAD":      2.4,
    "NZDCHF":      1.9,
    # ── Metals ────────────────────────────────────────────────────────
    "GOLD":        4.5,
    "XAUUSD":      4.5,   # alias
    "SILVER":      6.5,
    "XAGUSD":      6.5,   # alias
    "COPPER":      7.5,
    # ── Energies ──────────────────────────────────────────────────────
    "CRUDEOIL":    0.4,
    "CRUOIL":      0.4,
    "BRENT_OIL":   0.5,
    "NATURAL_GAS": 0.7,
    # ── Indices ───────────────────────────────────────────────────────
    "US_500":      5.0,
    "US500":       5.0,
    "US_TECH100":  25.0,
    "NAS100":      30.0,
    "US_30":       0.6,
    "US30":        0.6,
    "UK_100":      1.5,
    "GERMANY_40":  2.0,
    "FRANCE_40":   2.0,
    "JAPAN_225":   3.5,
    "AUS_200":     0.4,
    "HK_50":       1.2,
    # ── Crypto ────────────────────────────────────────────────────────
    "BTCUSD":      286.6,
    "ETHUSD":      11.0,
    "LTCUSD":      1.8,
    "SOLUSD":      2.8,
    "XRPUSD":      2.6,
    # ── Other ─────────────────────────────────────────────────────────
    "DOLLAR_INDX": 3.0,
}

# Symbol → pip_mult (price units per pip) — verified live against Market Watch
SYMBOL_PIP: dict[str, float] = {
    # Forex: pip = 0.0001 (4th decimal)
    "EURUSD": 0.0001, "GBPUSD": 0.0001, "USDCHF": 0.0001,
    "AUDUSD": 0.0001, "USDCAD": 0.0001, "NZDUSD": 0.0001,
    "EURGBP": 0.0001, "EURCHF": 0.0001, "EURCAD": 0.0001,
    "EURAUD": 0.0001, "EURNZD": 0.0001, "GBPCHF": 0.0001,
    "GBPCAD": 0.0001, "GBPAUD": 0.0001, "GBPNZD": 0.0001,
    "AUDCAD": 0.0001, "AUDCHF": 0.0001, "AUDNZD": 0.0001,
    "CADCHF": 0.0001, "NZDCAD": 0.0001, "NZDCHF": 0.0001,
    "XRPUSD": 0.0001,
    # JPY pairs: pip = 0.01
    "USDJPY": 0.01, "EURJPY": 0.01, "GBPJPY": 0.01,
    "AUDJPY": 0.01, "CADJPY": 0.01, "CHFJPY": 0.01,
    "NZDJPY": 0.01,
    # Metals / energies / indices: pip = 0.10
    "GOLD": 0.10, "XAUUSD": 0.10, "SILVER": 0.01, "XAGUSD": 0.01,
    "CRUDEOIL": 0.10, "CRUOIL": 0.10, "BRENT_OIL": 0.10,
    "US_500": 0.10, "US500": 0.10, "US_TECH100": 0.10,
    "NAS100": 0.10, "UK_100": 0.10, "GERMANY_40": 0.10,
    "FRANCE_40": 0.10, "AUS_200": 0.10, "HK_50": 0.10,
    "BTCUSD": 0.10, "ETHUSD": 0.10, "LTCUSD": 0.10, "SOLUSD": 0.10,
    # Pip = 10.0
    "US_30": 10.0, "US30": 10.0, "JAPAN_225": 10.0,
    # Others
    "COPPER": 0.001, "NATURAL_GAS": 0.001, "DOLLAR_INDX": 0.001,
}

_DEFAULT_SPREAD = 5.0   # conservative fallback for unknown symbols
_DEFAULT_PIP    = 0.0001


def get_ref_spread(symbol: str) -> float:
    """Reference spread in ea_pips for symbol (Market Watch 20260325).
    Tries exact match, then upper-case, then prefix match."""
    s = symbol.strip()
    v = SYMBOL_SPREADS.get(s) or SYMBOL_SPREADS.get(s.upper())
    if v is not None:
        return v
    # prefix fallback (e.g. "CrudeOIL" → "CRUDEOIL")
    su = s.upper()
    for key, val in SYMBOL_SPREADS.items():
        if su.startswith(key) or key.startswith(su):
            return val
    return _DEFAULT_SPREAD


def get_ref_pip(symbol: str) -> float:
    """Reference pip_mult for symbol."""
    s = symbol.strip()
    v = SYMBOL_PIP.get(s) or SYMBOL_PIP.get(s.upper())
    if v is not None:
        return v
    su = s.upper()
    for key, val in SYMBOL_PIP.items():
        if su.startswith(key) or key.startswith(su):
            return val
    return _DEFAULT_PIP


def effective_spread(symbol: str, live_ea_pips: float) -> float:
    """Return max(live, reference) spread in ea_pips.
    Ensures SL/TP floors and scalp targets are never below the known
    minimum spread for this symbol, even if live feed is momentarily zero."""
    return max(live_ea_pips, get_ref_spread(symbol))


def effective_pip(symbol: str, live_pip: float) -> float:
    """Return live pip_mult if valid, else fall back to reference."""
    if live_pip > 0:
        return live_pip
    return get_ref_pip(symbol)
