"""
Synthetic 5-minute bar generator for MT5_Bot_mk2.

Produces 100,000 bars per symbol across all configured symbols.
Output format matches MT5 CSV export (timestamp, open, high, low, close,
tick_volume, spread) so the training pipeline consumes it unchanged.

Spreads sourced from AvaTrade "SPREADS AND MARGINS" spreadsheet (2026-03).
Spread unit: pips as reported by the EA formula (ask-bid)*10^(digits-1),
which equals pips directly for all symbol types.

Usage:
    python tools/generate_synthetic_data.py
    python tools/generate_synthetic_data.py --out-dir data/raw --bars 100000
    python tools/generate_synthetic_data.py --symbols EURUSD GBPUSD XAUUSD
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Symbol parameters ─────────────────────────────────────────────────────────
# (base_price, annual_volatility, pip_size_in_price, spread_min_pips, spread_max_pips)
#
# pip_size_in_price is used only for CSV decimal rounding:
#   0.00001 -> 5 dp  (standard forex, 5-digit broker)
#   0.001   -> 4 dp  (JPY crosses)
#   0.10    -> 2 dp  (Gold XAUUSD)
#   0.01    -> 3 dp  (Silver XAGUSD, CrudeOil)
#   1.0     -> 2 dp  (indices US30, NAS100, US500)
#
# Spreads are in PIPS matching the EA's formula: (ask-bid)*10^(digits-1).
# Source: AvaTrade "SPREADS AND MARGINS" spreadsheet (Sheet8=forex, Sheet6=metals/energy).
# spread_min = London/NY overlap (tightest)
# spread_max = Asian/news spike (widest, used as hard cap)
SYMBOL_PARAMS = {
    # sym            base_px    ann_vol  pip_sz    spd_min  spd_max
    # ──────────────────────────────────────────────────────────────────────────
    # ALL values calibrated from: Market Watch 20260325 132856.csv (live broker data)
    # base_px  = live bid price at capture time (realistic simulation anchor)
    # pip_sz   = pip_size used for CSV decimal rounding only (not for SL/TP in live)
    #            SL/TP pip_mult is derived dynamically in trade_stacker from bid/ask/ea_pips
    # spd_min  = ea_pips at London/NY session (tightest observed)
    # spd_max  = ea_pips during news/Asia/weekend (widest, used as hard cap)
    # ea_pips  = (ask-bid) * 10^(digits-1)  ← EA formula
    # ── Forex Majors ────────────────────────────────────────────────────────
    "EURUSD":   (1.1585,    0.07,  0.00001,   0.8,    4.0),
    "GBPUSD":   (1.3388,    0.09,  0.00001,   1.2,    5.5),
    "USDJPY":   (159.03,    0.08,  0.001,     1.3,    5.5),
    "USDCHF":   (0.7906,    0.07,  0.00001,   1.3,    5.5),
    "AUDUSD":   (0.6961,    0.10,  0.00001,   0.9,    4.0),
    "USDCAD":   (1.3782,    0.07,  0.00001,   1.8,    7.0),
    "NZDUSD":   (0.5816,    0.10,  0.00001,   2.1,    7.0),
    # ── Forex Minors ────────────────────────────────────────────────────────
    "EURGBP":   (0.8653,    0.05,  0.00001,   1.2,    5.0),
    "EURJPY":   (184.25,    0.10,  0.001,     1.8,    7.0),
    "EURCHF":   (0.9159,    0.05,  0.00001,   1.6,    6.5),
    "EURCAD":   (1.5966,    0.08,  0.00001,   2.3,    8.5),
    "EURAUD":   (1.6641,    0.09,  0.00001,   2.5,    9.5),
    "EURNZD":   (1.9915,    0.09,  0.00001,   3.3,   12.0),
    "GBPJPY":   (212.92,    0.12,  0.001,     2.2,    9.0),
    "GBPCHF":   (1.0585,    0.08,  0.00001,   2.2,    8.0),
    "GBPCAD":   (1.8450,    0.09,  0.00001,   3.3,   12.0),
    "GBPAUD":   (1.9230,    0.10,  0.00001,   3.2,   12.0),
    "GBPNZD":   (2.3014,    0.10,  0.00001,   4.9,   18.0),
    "AUDJPY":   (110.71,    0.10,  0.001,     1.9,    7.0),
    "AUDCAD":   (0.9593,    0.08,  0.00001,   2.4,    8.5),
    "AUDCHF":   (0.5503,    0.08,  0.00001,   3.2,   10.0),
    "AUDNZD":   (1.1967,    0.07,  0.00001,   2.4,    8.5),
    "CADJPY":   (115.39,    0.08,  0.001,     2.3,    8.5),
    "CADCHF":   (0.5736,    0.07,  0.00001,   1.8,    7.0),
    "CHFJPY":   (201.14,    0.08,  0.001,     2.5,    9.0),
    "NZDJPY":   (92.50,     0.10,  0.001,     2.5,    9.0),
    "NZDCAD":   (0.8016,    0.09,  0.00001,   2.4,    8.5),
    "NZDCHF":   (0.4598,    0.08,  0.00001,   1.9,    7.5),
    # ── Metals (verified broker names from Market Watch) ─────────────────────
    # GOLD  bid=4567.62  ea_pips=4.5  pip_mult=0.10  (AvaTrade uses GOLD, not XAUUSD)
    # SILVER bid=72.884  ea_pips=6.5  pip_mult=0.01
    # COPPER bid=5.5225  ea_pips=7.5  pip_mult=0.001
    "GOLD":     (4567.62,   0.18,  0.10,      4.5,   18.0),
    "SILVER":   (72.88,     0.25,  0.01,      6.5,   22.0),
    "COPPER":   (5.52,      0.22,  0.001,     7.5,   25.0),
    # ── Energies ─────────────────────────────────────────────────────────────
    # CrudeOIL  bid=87.74   ea_pips=0.4  pip_mult=0.10
    # BRENT_OIL bid=99.10   ea_pips=0.5  pip_mult=0.10
    # NATURAL_GAS bid=2.865 ea_pips=0.7  pip_mult=0.001
    "CrudeOIL":     (87.74,  0.35,  0.10,     0.4,    5.0),
    "BRENT_OIL":    (99.10,  0.30,  0.10,     0.5,    5.0),
    "NATURAL_GAS":  (2.865,  0.50,  0.001,    0.7,    8.0),
    # ── Indices (pip_sz matches live pip_mult from Market Watch) ─────────────
    # US_500     pip_mult=0.10  → pip_sz=0.10
    # US_TECH100 pip_mult=0.10  → pip_sz=0.10
    # US_30      pip_mult=10.0  → pip_sz=10.0  (digits=0, each pip=$10)
    # NAS100     pip_mult=0.10  → pip_sz=0.10
    # JAPAN_225  pip_mult=10.0  → pip_sz=10.0  (similar to US_30)
    # Others     pip_mult=0.10
    "US_500":     (6660.25,  0.18,  0.10,     5.0,   20.0),
    "US_TECH100": (24432.25, 0.22,  0.10,    25.0,   80.0),
    "US_30":      (46859.0,  0.15,  10.0,     0.6,    3.0),
    "NAS100":     (12625.50, 0.22,  0.10,    30.0,   90.0),
    "UK_100":     (10095.5,  0.15,  0.10,     1.5,    7.0),
    "GERMANY_40": (23207.5,  0.18,  0.10,     2.0,    9.0),
    "FRANCE_40":  (7859.0,   0.16,  0.10,     2.0,    9.0),
    "JAPAN_225":  (53575.0,  0.18,  10.0,     3.5,   15.0),
    "AUS_200":    (8609.0,   0.14,  0.10,     0.4,    4.0),
    "HK_50":      (25262.0,  0.20,  0.10,     1.2,    8.0),
    # ── Crypto (ea_pips from live Market Watch) ──────────────────────────────
    # BTCUSD bid=71596.32 ea_pips=286.6  pip_mult=0.10
    # ETHUSD bid=2181.50  ea_pips=11.0   pip_mult=0.10
    # LTCUSD bid=56.39    ea_pips=1.8    pip_mult=0.10
    # SOLUSD bid=92.81    ea_pips=2.8    pip_mult=0.10
    # XRPUSD bid=1.4213   ea_pips=2.6    pip_mult=0.0001
    "BTCUSD":  (71596.32,   0.80,  0.10,   286.6, 1000.0),
    "ETHUSD":  (2181.50,    0.65,  0.10,    11.0,   40.0),
    "LTCUSD":  (56.39,      0.60,  0.10,     1.8,    8.0),
    "SOLUSD":  (92.81,      0.70,  0.10,     2.8,   12.0),
    "XRPUSD":  (1.4213,     0.80,  0.00001,  2.6,   10.0),
}

BARS_PER_SYMBOL = 100_000
BARS_PER_WEEK   = 5 * 24 * 12   # 5 days × 24h × 12 bars/h (5-min)
MARKET_OPEN_DAY  = 0   # Monday
MARKET_CLOSE_DAY = 4   # Friday (close at 22:00 UTC)

# Start date for synthetic data
START_DATE = datetime(2018, 1, 1, 0, 0, 0)


# ── Market calendar helpers ───────────────────────────────────────────────────

def _next_market_open(dt: datetime) -> datetime:
    """Advance dt to the next valid 5-min market bar (skip weekends)."""
    # Skip Saturday and Sunday (weekday 5 = Sat, 6 = Sun)
    # Also skip Friday after 22:00 UTC
    while True:
        wd = dt.weekday()
        if wd == 5:                             # Saturday — skip to Sunday 22:00
            dt = dt.replace(hour=22, minute=0, second=0, microsecond=0)
            dt += timedelta(days=1)
            continue
        if wd == 6 and dt.hour < 22:            # Sunday before 22:00 — jump forward
            dt = dt.replace(hour=22, minute=0, second=0)
            continue
        if wd == 4 and dt.hour >= 22:           # Friday after 22:00 — jump to Sun 22:00
            days_ahead = 2
            dt = dt.replace(hour=22, minute=0, second=0) + timedelta(days=days_ahead)
            continue
        break
    return dt


def generate_timestamps(n_bars: int, start: datetime) -> list:
    """Generate n_bars × 5-min timestamps, skipping weekends."""
    stamps = []
    dt = _next_market_open(start)
    while len(stamps) < n_bars:
        stamps.append(dt)
        dt += timedelta(minutes=5)
        dt = _next_market_open(dt)
    return stamps


# ── Price simulation ──────────────────────────────────────────────────────────

def _regime_vol(base_vol: float, t: int, period: int = 3000) -> float:
    """Oscillate volatility between 0.5× and 1.5× base over a long cycle."""
    phase = np.sin(2 * np.pi * t / period)
    return base_vol * (1.0 + 0.5 * phase)


def simulate_prices(base_price: float, annual_vol: float,
                    n_bars: int, rng: np.random.Generator) -> np.ndarray:
    """
    Simulate close prices with:
    - Geometric Brownian Motion core
    - Regime-switching volatility (high/low vol phases)
    - Mean reversion impulse every ~500 bars to prevent runaway drift
    """
    bar_vol = annual_vol / np.sqrt(252 * 24 * 12)   # 5-min bar vol
    closes  = np.zeros(n_bars)
    closes[0] = base_price

    for i in range(1, n_bars):
        rv  = _regime_vol(bar_vol, i)
        ret = rng.normal(0.0, rv)

        # Mild mean-reversion pull every 500 bars
        if i % 500 == 0:
            drift = -0.005 * (closes[i-1] / base_price - 1.0)
            ret += drift

        closes[i] = closes[i-1] * np.exp(ret)

    return closes


def build_ohlcv(closes: np.ndarray, annual_vol: float,
                rng: np.random.Generator) -> tuple:
    """
    Build O, H, L, V arrays from close prices.
    O  = previous close
    H  = max(O, C) + random wick
    L  = min(O, C) - random wick
    V  = realistic tick volume 50–3000
    """
    n       = len(closes)
    bar_vol = annual_vol / np.sqrt(252 * 24 * 12)

    opens = np.empty(n)
    opens[0] = closes[0] * (1 + rng.normal(0, bar_vol * 0.5))
    opens[1:] = closes[:-1]

    # Wicks: 0.2× to 1.5× of |C-O|, minimum 0.1× bar_vol × price
    body = np.abs(closes - opens)
    wick_scale = rng.uniform(0.2, 1.5, n)
    min_wick   = np.abs(closes) * bar_vol * 0.1
    wick       = np.maximum(body * wick_scale, min_wick)

    highs = np.maximum(opens, closes) + wick * rng.uniform(0.3, 1.0, n)
    lows  = np.minimum(opens, closes) - wick * rng.uniform(0.3, 1.0, n)

    # Clamp: low > 0, high > low
    lows  = np.maximum(lows, closes * 0.5)
    highs = np.maximum(highs, lows + closes * bar_vol * 0.01)

    # Volume: log-normal centred on 500 ticks, occasional spikes
    base_vol_v = rng.lognormal(mean=np.log(500), sigma=0.8, size=n).astype(int)
    spike_mask = rng.random(n) < 0.02                    # 2% of bars are high-vol
    base_vol_v[spike_mask] = rng.integers(2000, 8000, spike_mask.sum())
    volumes = np.clip(base_vol_v, 10, 10000)

    return opens, highs, lows, volumes


def build_spread(n_bars: int, smin: float, smax: float,
                 timestamps: list, rng: np.random.Generator) -> np.ndarray:
    """
    Simulate spreads in pips (float, matching the EA's output format).
    - Intraday pattern (tighter London/NY, wider Asia/overnight)
    - Random news spikes (~0.3% of bars)
    - Weekend-open gap spike
    """
    spreads = np.full(n_bars, smin, dtype=np.float32)

    for i, ts in enumerate(timestamps):
        hour = ts.hour

        # Intraday spread multiplier
        if   7 <= hour < 8:   mult = 2.5   # pre-London
        elif 8 <= hour < 12:  mult = 1.0   # London: tight
        elif 12 <= hour < 17: mult = 1.05  # London/NY overlap: tightest
        elif 17 <= hour < 20: mult = 1.5   # NY afternoon
        elif 20 <= hour < 22: mult = 2.0   # NY close
        else:                 mult = 2.8   # Asia/overnight

        base  = smin + (smax - smin) * 0.3 * mult
        noise = rng.uniform(0, (smax - smin) * 0.25)
        spreads[i] = min(smax, base + noise)

    # Weekend-open gap: spike to smax
    for i in range(1, n_bars):
        if timestamps[i].weekday() == 6 and timestamps[i].hour == 22:
            spreads[i] = smax

    # News spikes (~0.3% of bars): 2–5× normal spread
    spike_idx = rng.choice(n_bars, size=max(1, n_bars // 333), replace=False)
    spreads[spike_idx] = np.minimum(
        spreads[spike_idx] * rng.uniform(2.0, 5.0, len(spike_idx)),
        smax * 3.0
    )

    return np.round(spreads, 1)


# ── Main generator ────────────────────────────────────────────────────────────

def generate_symbol(symbol: str, params: tuple,
                    n_bars: int, out_dir: str,
                    rng: np.random.Generator) -> str:
    base_price, annual_vol, pip_size, smin, smax = params

    print(f"  [{symbol}] Generating {n_bars:,} bars...", end="", flush=True)

    timestamps = generate_timestamps(n_bars, START_DATE)
    closes     = simulate_prices(base_price, annual_vol, n_bars, rng)
    opens, highs, lows, volumes = build_ohlcv(closes, annual_vol, rng)
    spreads    = build_spread(n_bars, smin, smax, timestamps, rng)

    # Format timestamps as MT5 format: 2024.01.15 08:30:00
    ts_strings = [t.strftime("%Y.%m.%d %H:%M:%S") for t in timestamps]

    df = pd.DataFrame({
        "timestamp":   ts_strings,
        "open":        np.round(opens,  _digits(pip_size)),
        "high":        np.round(highs,  _digits(pip_size)),
        "low":         np.round(lows,   _digits(pip_size)),
        "close":       np.round(closes, _digits(pip_size)),
        "tick_volume": volumes,
        "spread":      spreads,
    })

    path = os.path.join(out_dir, f"{symbol}.csv")
    df.to_csv(path, index=False, encoding="utf-8")

    print(f" done -> {path}  (date range: {ts_strings[0]} -> {ts_strings[-1]})")
    return path


def _digits(pip_size: float) -> int:
    """Number of decimal places from pip_size."""
    if pip_size >= 10.0:  return 0   # US_30, JAPAN_225 (integer points)
    if pip_size >= 1.0:   return 1   # other large-pip indices
    if pip_size >= 0.1:   return 2   # GOLD, CrudeOIL, US_500, BTCUSD
    if pip_size >= 0.01:  return 3   # SILVER, JPY pairs
    if pip_size >= 0.001: return 4   # COPPER, NATURAL_GAS
    return 5                         # standard 5-digit forex


def main():
    parser = argparse.ArgumentParser(description=f"Generate synthetic 5-min bars for training ({len(SYMBOL_PARAMS)} symbols available)")
    parser.add_argument("--out-dir", default="data/raw",
                        help="Output directory (default: data/raw)")
    parser.add_argument("--bars", type=int, default=BARS_PER_SYMBOL,
                        help=f"Bars per symbol (default: {BARS_PER_SYMBOL:,})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--symbols", nargs="+",
                        help="Subset of symbols (default: all)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing CSV files")
    args = parser.parse_args()

    # Resolve out_dir relative to repo root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root  = os.path.dirname(script_dir)
    out_dir    = os.path.join(repo_root, args.out_dir) \
                 if not os.path.isabs(args.out_dir) else args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    symbols = args.symbols or list(SYMBOL_PARAMS.keys())
    unknown = [s for s in symbols if s not in SYMBOL_PARAMS]
    if unknown:
        print(f"Unknown symbols: {unknown}")
        print(f"Available: {list(SYMBOL_PARAMS.keys())}")
        sys.exit(1)

    rng        = np.random.default_rng(args.seed)
    total_bars = len(symbols) * args.bars

    print(f"MT5_Bot_mk2 — Synthetic Data Generator")
    print(f"  Symbols  : {len(symbols)} -> {symbols}")
    print(f"  Bars each: {args.bars:,}")
    print(f"  Total    : {total_bars:,}")
    print(f"  Out dir  : {out_dir}")
    print(f"  Seed     : {args.seed}")
    print()

    skipped = []
    generated = []

    for sym in symbols:
        path = os.path.join(out_dir, f"{sym}.csv")
        if os.path.exists(path) and not args.overwrite:
            print(f"  [{sym}] Already exists — skipping (use --overwrite to replace)")
            skipped.append(sym)
            continue
        generated.append(generate_symbol(sym, SYMBOL_PARAMS[sym], args.bars, out_dir, rng))

    print()
    print(f"Done.  Generated: {len(generated)}  Skipped: {len(skipped)}")
    print(f"Total bars written: {len(generated) * args.bars:,}")
    if skipped:
        print(f"Skipped (already exist): {skipped}")
    print()
    print("Run training with:")
    print("  python run.py --mode train")


if __name__ == "__main__":
    main()
