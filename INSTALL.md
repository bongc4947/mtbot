# MT5_Bot_mk2 — Installation & Operations Guide

> **Live trading system. Bugs cause real financial loss. Read this guide fully before starting.**

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Python Environment](#2-python-environment)
3. [MetaTrader 5 Setup](#3-metatrader-5-setup)
4. [ZMQ Library Installation](#4-zmq-library-installation)
5. [Compile and Attach the Expert Advisor](#5-compile-and-attach-the-expert-advisor)
6. [Configuration](#6-configuration)
7. [Running the Bot](#7-running-the-bot)
8. [Dashboard & Startup Progress](#8-dashboard--startup-progress)
9. [Operations Reference](#9-operations-reference)
10. [Troubleshooting](#10-troubleshooting)
11. [Directory Layout](#11-directory-layout)
12. [Architecture Overview](#12-architecture-overview)

---

## 1. System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10 | Windows 11 |
| Python | 3.10 | 3.11+ |
| RAM | 4 GB | 8 GB |
| CPU | AMD A8 (4 cores) | Any modern 4+ core |
| MetaTrader 5 | Any broker build | IronFX demo / live |
| Disk | 5 GB free | 10 GB free |

> GPU acceleration is optional. The bot runs on CPU-only with no loss in correctness.

---

## 2. Python Environment

```bash
cd path\to\MT5_bot_mk2f

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

**Key dependencies installed:**

| Package | Purpose |
|---------|---------|
| `numpy`, `pandas` | Data manipulation |
| `scikit-learn` | Model training, class balancing |
| `lightgbm` | Tabular ensemble component |
| `pyzmq` | ZeroMQ bridge to MT5 EA |
| `pyyaml` | Config loading |
| `rich` | Terminal dashboard (TUI) |
| `joblib` | Model save/load |

**Optional — CPU-only PyTorch** (for N-HiTS neural model):

```bash
pip install torch==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu
```

**Optional — DirectML GPU acceleration** (Windows only):

```bash
pip install torch-directml
```

---

## 3. MetaTrader 5 Setup

1. Open MT5 → **Tools → Options → Expert Advisors**
2. Enable all three checkboxes:
   - ✅ Allow automated trading
   - ✅ Allow DLL imports
   - ✅ Allow external imports
3. Click OK and restart MT5 if prompted.

> The bot will not trade without "Allow automated trading" enabled on the EA smiley face icon.

---

## 4. ZMQ Library Installation

All required ZMQ files are bundled in the `MT5 Include Libraries/` folder in this repo.

**Step 1 — MQL5 Include files** (copy once, skip if already present):

```
Copy FROM:
  MT5 Include Libraries\Include\Zmq\
    AtomicCounter.mqh, Context.mqh, Errno.mqh, Socket.mqh,
    SocketOptions.mqh, Z85.mqh, Zmq.mqh, ZmqMsg.mqh

Copy TO:
  <MT5 Data Folder>\MQL5\Include\Zmq\
```

**Step 2 — DLL files** (copy once, skip if already present):

```
Copy FROM:
  MT5 Include Libraries\Libraries\
    libzmq.dll, libsodium.dll

Copy TO:
  <MT5 Data Folder>\MQL5\Libraries\
```

> Find the MT5 Data Folder: in MT5 go to **File → Open Data Folder**

**ZMQ Port Layout** (no conflict with mk1):

```
EA   PUSH  binds  tcp://*:5557    →   Python PULL  connects 127.0.0.1:5557
Python PUSH binds tcp://127.0.0.1:5558  →   EA PULL  connects 5558
```

Both ports bind to `127.0.0.1` only — never exposed to the network.

---

## 5. Compile and Attach the Expert Advisor

1. Copy the EA file to MT5:
   ```
   ea\MT5_Bot_mk2_EA.mq5  →  <MT5 Data Folder>\MQL5\Experts\
   ```

2. Open **MetaEditor** (press F4 in MT5)

3. Open `MT5_Bot_mk2_EA.mq5` from the Experts folder

4. Press **F7** to compile — must show **0 errors, 0 warnings**

5. Back in MT5 → **Navigator → Expert Advisors** → find `MT5_Bot_mk2_EA`

6. Drag to a chart (e.g. EURUSD M1)

7. In the EA dialog:
   - ✅ Allow DLL imports
   - ✅ Allow live trading
   - Set `PushPort = 5557` and `PullPort = 5558`

8. Confirm the EA is running: the chart should show a **smiley face icon** (not ❌)

---

## 6. Configuration

### `config/settings.yaml` — Key Parameters

```yaml
zmq:
  ea_host: "127.0.0.1"
  ea_push_port: 5557          # EA → Python
  py_push_port: 5558          # Python → EA

inference:
  confidence_threshold: 0.90        # Minimum signal confidence to open a trade
  stack_confidence_threshold: 0.97  # Threshold for stacking multiple lots

trading:
  base_lot: 0.01
  max_total_trades: 50
  max_daily_drawdown_pct: 5.0       # Bot self-stops at this drawdown %
  label_sl_pips: 10                 # Stop-loss used for outcome labeling (training)
  label_tp_pips: 20                 # Take-profit used for outcome labeling
  label_horizon: 20                 # Max bars to wait for TP/SL during labeling
  min_trades_per_10min: 15          # Target micro-scalp frequency

model:
  input_window: 60                  # Bars of lookback per inference
  min_train_samples: 5000
  retrain_interval_hours: 4

ensemble:
  nhits_weight: 0.30                # N-HiTS (macro trend)
  lgbm_weight:  0.40                # LightGBM (microstructure)
  mlp_weight:   0.30                # MLP (nonlinear)

startup:
  preflight_auto_start_secs: 60     # Seconds to wait at pre-flight before auto-starting
```

### `config/symbols.yaml` — Symbol List

Symbols are grouped by category. On startup the bot fuzzy-matches these against your
broker's Market Watch list (handles suffixes like `EURUSDm`, `EURUSD.i`, `XAUUSD#`).

To reduce startup time and RAM, comment out symbol groups you don't need:

```yaml
symbols:
  forex_majors:
    - EURUSD
    - GBPUSD
    # ... add or remove symbols here
```

---

## 7. Running the Bot

**Prerequisites:** MT5 is open and the EA shows a smiley face.

```bash
# Activate environment first
venv\Scripts\activate

# Live trading — launches dashboard in a separate window
python run.py --mode live

# Live trading — no dashboard (headless, logs only)
python run.py --mode live --no-dashboard

# Training only — download data and train models, then exit
python run.py --mode train
```

### What Happens on First Run

The startup sequence takes several minutes on first run:

| Step | Progress | What it does |
|------|----------|--------------|
| 1/8 | 3–10% | Initialize GPU/CPU acceleration context |
| 2/8 | 10–18% | Start ZMQ bridge, connect to MT5 EA |
| 3/8 | 18–26% | Fetch broker symbol list (Market Watch) |
| 4/8 | 26–30% | Allocate per-symbol tick buffers |
| 5/8 | 30–60% | Export 5 years of M1 history per symbol from MT5 |
| 6/8 | 60–65% | Load cached ensemble models (or create new ones) |
| 7/8 | 65–90% | Train N-HiTS + LightGBM + MLP ensemble per symbol |
| 8/8 | 90–95% | Pre-flight gate — review model summary before trading |
| — | 100% | Live trading begins |

On subsequent runs, steps 5–7 are skipped if cached data and models exist (startup ~30s).

---

## 8. Dashboard & Startup Progress

The dashboard opens automatically in a **separate terminal window**.

### Startup Screen

While the engine is initializing, the dashboard shows:

```
MT5_Bot_mk2  |  STARTING UP  [TRAINING]

  Overall Progress: [########--------] 43%

  Step 5/8: Exporting historical data from MT5...
  Symbols: 3/12  |  Elapsed: 47s
```

If the engine connection drops briefly, the dashboard shows the **last known state** with
a `(reconnecting...)` badge rather than going blank.

### Live Trading Screen (4-panel layout)

```
┌─────────────────────────────────────────────────────┐
│           MT5_Bot_mk2  |  Live Ensemble  [RUNNING]  │
├──────────────────────┬──────────────────────────────┤
│  Progress (agents)   │  Messages & Signals          │
│  Team | Agent | Status│  Time | Type | Content      │
│  ...                 │  ...                         │
├──────────────────────┴──────────────────────────────┤
│  Current Analysis                                   │
│  Balance / Equity / P&L / Win Rate / Ensemble Weights│
│  Symbol performance table                           │
├─────────────────────────────────────────────────────┤
│  Signals: 42 | Trades/10min: 17 | Win: 61% | ...    │
└─────────────────────────────────────────────────────┘
```

### Dashboard Key Bindings

| Key | Action |
|-----|--------|
| `P` | Pause trading (open positions stay open) |
| `R` | Resume trading |
| `T` | Trigger model retrain in background |
| `D` | Refresh historical data |
| `S` | Shutdown bot and close positions |

> These keys work in both the dashboard window and the main `run.py` console.

---

## 9. Operations Reference

### Normal Operating Metrics

| Metric | Healthy | Warning | Critical |
|--------|---------|---------|---------|
| Win Rate | ≥ 60% | 50–60% | < 50% |
| Profit Factor | ≥ 1.5 | 1.0–1.5 | < 1.0 |
| Trades/10min | ≥ 15 | 5–15 | < 5 |
| Daily Drawdown | < 2% | 2–5% | ≥ 5% (auto-stop) |
| Inference Latency | < 50ms | 50–100ms | > 100ms |

### Retrain Schedule

Models auto-retrain every `retrain_interval_hours` (default: 4 hours) in the background.
You can force an immediate retrain by pressing `T` in the dashboard.

### Auto-Stop Conditions

The bot will automatically pause trading and set `SHUTDOWN` state when:
- Daily drawdown exceeds `max_daily_drawdown_pct` (default 5%)
- Risk manager detects `is_shutdown = True`
- ZMQ connection to MT5 is lost for > `timeout_ms` (default 3000ms)

### Historical Data Location

MT5 exports CSVs to:
```
%APPDATA%\MetaQuotes\Terminal\<TerminalID>\MQL5\Files\MT5_Bot_mk2\data\raw\
```

Python reads from `data/raw/<SYMBOL>.csv`. If these don't match, either:
- Copy files manually, or
- Update `config/settings.yaml → data.raw_dir`

---

## 10. Troubleshooting

### EA smiley shows ❌ (red X)

- Check MT5 → Tools → Options → Expert Advisors → all boxes ticked
- Check the EA's "Allow DLL imports" checkbox in the EA dialog
- Verify `libzmq.dll` and `libsodium.dll` are in `MQL5\Libraries\`

### "Waiting for Market Watch list" hangs at startup

1. Confirm the EA is attached to a chart and shows a smiley face
2. Check MT5 Experts tab for errors
3. After 30s timeout, the bot falls back to the configured symbol list and continues

### Win rate shows 0% after training

Caused by spread unit mismatch. Verify `config/settings.yaml`:
```yaml
trading:
  label_sl_pips: 10
  label_tp_pips: 20
```
The trainer auto-detects integer-point spreads from MT5 (e.g. `10` points = `0.0001` price)
and converts them. If win rate stays 0%, check your broker's spread for the symbol.

### Inference shows all HOLD signals

- Lower `confidence_threshold` in `settings.yaml` (default 0.90; try 0.70 to diagnose)
- Check `logs/mt5_bot_mk2.log` for `[inference] non-HOLD=0` lines
- Run `/diagnose` in Claude Code for a full diagnosis

### Dashboard shows "Waiting for engine..." indefinitely

- Confirm `python run.py --mode live` is running in another terminal
- Check `logs/mt5_bot_mk2.log` for startup errors
- The startup screen appears during the 8-step initialization — this is normal

### UnicodeEncodeError in logs

The logger writes UTF-8 to both the console and log file. If you see encoding errors,
run the bot in a UTF-8 terminal:
```bash
chcp 65001
python run.py --mode live
```

### LightGBM not installed

```bash
pip install lightgbm>=4.0.0
```
The bot has a `sklearn GradientBoostingClassifier` fallback but LightGBM gives better
microstructure edge detection.

---

## 11. Directory Layout

```
MT5_bot_mk2f/
├── run.py                        ← Entry point
├── requirements.txt              ← Python dependencies
├── INSTALL.md                    ← This file
├── CLAUDE.md                     ← Claude Code project context
│
├── config/
│   ├── settings.yaml             ← All tunable parameters
│   └── symbols.yaml              ← Symbol list by category
│
├── ea/
│   └── MT5_Bot_mk2_EA.mq5        ← Copy to MT5 Experts folder
│
├── ingestion/
│   ├── zmq_controller.py         ← ZMQ PUSH/PULL bridge (ports 5557/5558)
│   ├── message_router.py         ← Routes ZMQ messages to handlers
│   ├── tick_buffer.py            ← Per-symbol circular tick buffer
│   ├── historical_exporter.py    ← Requests CSV export from EA
│   └── symbol_resolver.py        ← Fuzzy-matches broker symbol names
│
├── features/
│   └── encoder.py                ← encode_ohlcv() → (N, 15) features; gap-aware
│
├── training/
│   ├── trainer.py                ← Trains ensemble per symbol; class balancing
│   └── labeler.py                ← TP/SL outcome labeling (replaces direction threshold)
│
├── models/
│   ├── nhits.py                  ← N-HiTS neural network (macro trend, weight 0.30)
│   ├── lgbm_model.py             ← LightGBM tabular (microstructure, weight 0.40)
│   ├── mlp.py                    ← MLP neural network (nonlinear, weight 0.30)
│   ├── ensemble.py               ← Weighted ensemble of all three models
│   └── model_manager.py          ← Loads/saves/caches ensemble per symbol
│
├── inference/
│   └── engine.py                 ← Per-tick feature encoding; batched inference
│
├── execution/
│   ├── risk_manager.py           ← Drawdown guard; SHUTDOWN flag
│   ├── position_sizer.py         ← Lot size from equity
│   ├── trailing_manager.py       ← Trailing SL/TP management
│   ├── trade_stacker.py          ← Confidence-band stacking (0.97→2x, 0.99→6x lots)
│   └── micro_scalper.py          ← High-frequency 2-pip scalp re-entry
│
├── interface/
│   ├── state_manager.py          ← Singleton state store (thread-safe)
│   ├── state_bridge.py           ← File-based IPC: engine ↔ dashboard window
│   ├── dashboard.py              ← Launches dashboard_window.py in new console
│   ├── dashboard_window.py       ← Standalone Rich TUI (reads state.json)
│   └── metrics_view.py           ← Panel builders: Progress, Messages, Report, Status
│
├── utils/
│   ├── logger.py                 ← Rotating file + UTF-8 console handler
│   ├── config_loader.py          ← load_config(), load_symbols()
│   ├── gpu_utils.py              ← DirectML / CUDA / CPU acceleration context
│   └── data_utils.py             ← load_csv(), detect_session_gaps()
│
├── .claude/
│   ├── agents/                   ← Specialized Claude Code subagents
│   ├── commands/                 ← Slash commands (/diagnose, /retrain, /code-review)
│   └── rules/trading-safety.md  ← Hard rules enforced on every code change
│
├── data/
│   ├── raw/                      ← MT5-exported CSVs (auto-created)
│   ├── processed/                ← Feature arrays (auto-created)
│   └── cache/models/             ← Saved ensemble model files (auto-created)
│
└── logs/
    └── mt5_bot_mk2.log           ← Rotating log (10 MB × 5 backups)
```

---

## 12. Architecture Overview

```
MetaTrader 5 EA  ←──ZMQ 5557/5558──→  Python (run.py)
                                              │
                    ┌─────────────────────────┤
                    │                         │
             ingestion/                  interface/
           ZMQ bridge                  StateManager
           Tick buffers               state_bridge.py ──→ state.json
           Symbol resolver                              ←── cmds.json
           Historical export                                    │
                    │                                    dashboard_window.py
             features/                                  (separate window)
           Feature encoder
           (15 features/tick)
                    │
             training/
           Labeler (TP/SL sim)
           Trainer + balancer
                    │
             models/
           N-HiTS  (0.30)
           LightGBM (0.40)
           MLP      (0.30)
            ↓ Ensemble
             inference/
           _encode_tick_sequence()
           Batched prediction
                    │
             execution/
           RiskManager  ← MUST pass through
           TradeStacker → ZMQ → MT5 EA
           MicroScalper
           TrailingManager
```

### Ensemble Model

Three models vote on every signal, weighted by validation performance:

| Model | Type | Strength | Default Weight |
|-------|------|----------|----------------|
| N-HiTS | Neural (sequence) | Macro trend, multi-scale | 0.30 |
| LightGBM | Gradient boosted tree | Microstructure patterns | 0.40 |
| MLP | Neural (tabular) | Nonlinear feature interaction | 0.30 |

Weights auto-adjust per symbol based on recent validation win rate.

### Trade Stacking (Confidence Bands)

When confidence is very high, the bot opens multiple lots:

| Confidence | Lots |
|-----------|------|
| ≥ 0.97 | 2× base_lot |
| ≥ 0.975 | 3× base_lot |
| ≥ 0.98 | 4× base_lot |
| ≥ 0.99 | 6× base_lot |

### Safety Gates

```
Signal → RiskManager.check() → TradeStacker.execute() → ZMQ → EA → MT5
              ↑
         is_shutdown check
         drawdown check
         max_trades check
```

No code path should call `zmq.send_trade()` without passing through `RiskManager` first.
