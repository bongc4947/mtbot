import os
import numpy as np
import pandas as pd
from utils.logger import get_logger

log = get_logger("data_utils")

REQUIRED_COLS = ["timestamp", "open", "high", "low", "close", "tick_volume", "spread"]


def detect_session_gaps(df: pd.DataFrame, gap_minutes: float = 2.0) -> np.ndarray:
    """
    Return a boolean array of length len(df).
    True at index i means bar i follows a session gap
    (time elapsed from bar i-1 to bar i exceeds gap_minutes).
    Used to prevent rolling features and targets from spanning market open/close boundaries.
    First element is always False.
    """
    if len(df) < 2 or "timestamp" not in df.columns:
        return np.zeros(len(df), dtype=bool)
    ts = pd.to_datetime(df["timestamp"])
    diffs_sec = ts.diff().dt.total_seconds().fillna(0).values
    return (diffs_sec > gap_minutes * 60).astype(bool)


def load_csv(path: str) -> pd.DataFrame:
    """
    Load an MT5-exported CSV robustly.

    Handles:
    - UTF-16 LE (MQL5 default without FILE_ANSI) and UTF-8 / UTF-8-BOM
    - MT5 date format: '2025.12.15 02:55:00' (dots in date part)
    - Standard ISO format: '2025-12-15 02:55:00'
    """
    # --- Encoding detection --------------------------------------------------
    df = None
    for enc in ("utf-8-sig", "utf-8", "utf-16", "utf-16-le"):
        try:
            df = pd.read_csv(path, encoding=enc)
            # Quick sanity: header must contain 'timestamp'
            if "timestamp" not in df.columns:
                df = None
                continue
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception:
            continue

    if df is None:
        raise ValueError(f"Cannot decode CSV: {path}")

    # Strip any stray whitespace from column names (UTF-16 BOM can leave junk)
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]

    # --- Timestamp parsing ----------------------------------------------------
    # MT5 writes '2025.12.15 02:55:00'; try both dot and dash formats
    ts_raw = df["timestamp"].astype(str).str.strip()
    parsed = pd.to_datetime(ts_raw, format="%Y.%m.%d %H:%M:%S", errors="coerce")
    if parsed.isna().mean() > 0.05:   # too many failures — try ISO format
        parsed = pd.to_datetime(ts_raw, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    if parsed.isna().mean() > 0.05:   # last resort: let pandas infer
        parsed = pd.to_datetime(ts_raw, errors="coerce")

    df["timestamp"] = parsed
    df.dropna(subset=["timestamp"], inplace=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def validate_dataset(df: pd.DataFrame, symbol: str, min_bars: int = 10000) -> bool:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        log.warning(f"[{symbol}] Missing columns: {missing}")
        return False
    if len(df) < min_bars:
        log.warning(f"[{symbol}] Only {len(df)} bars, need {min_bars}")
        return False
    null_pct = df[REQUIRED_COLS].isnull().mean().max()
    if null_pct > 0.01:
        log.warning(f"[{symbol}] High null rate: {null_pct:.1%}")
        return False
    return True


def list_available_symbols(raw_dir: str) -> list:
    if not os.path.isdir(raw_dir):
        return []
    files = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]
    return [os.path.splitext(f)[0] for f in files]


class CircularBuffer:
    """Fixed-size float32 ring buffer for tick data."""

    def __init__(self, capacity: int, n_features: int):
        self.capacity = capacity
        self.n_features = n_features
        self._buf = np.zeros((capacity, n_features), dtype=np.float32)
        self._head = 0
        self._size = 0

    def push(self, row: np.ndarray):
        self._buf[self._head] = row
        self._head = (self._head + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def get(self, n: int) -> np.ndarray:
        """Return last n rows in chronological order."""
        if n > self._size:
            n = self._size
        end = self._head
        start = (end - n) % self.capacity
        if start < end:
            return self._buf[start:end]
        return np.concatenate([self._buf[start:], self._buf[:end]], axis=0)

    @property
    def size(self) -> int:
        return self._size

    def is_full(self) -> bool:
        return self._size >= self.capacity
