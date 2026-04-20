"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║   SYNTHETIC QUANT ELITE v4.3 — Calibrated Precision Edition                   ║
║   All v4.1 features preserved + v4.3 surgical fixes                            ║
║   Fix1: Empirical target cap · Fix2: Dynamic RR · Fix3: Horizon-aware windows  ║
║   Fix4: Z-score neutral gate · Fix5: Ensemble agreement gate                   ║
║   Fix6: HybridTargetRefiner cold-memory path                                   ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""
import os
import asyncio
import json
import logging
import math
import os
import pickle
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import websockets
from numba import njit, prange
from scipy.stats import norm, kstest, linregress, gaussian_kde

from telegram import (
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("SQEv43")

# Dummy HTTP server so Render keeps service alive
if os.environ.get("RENDER"):
    from threading import Thread
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"SQE Bot Running")

    def run_server():
        port = int(os.environ.get("PORT", 10000))
        HTTPServer(("0.0.0.0", port), Handler).serve_forever()

    Thread(target=run_server, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# CREDENTIALS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN = "8533545784:AAG1EE1PCu1d_IlzrpfVfnsSl9XzpmJwLj0"
DERIV_APP_ID   = "1089"
DERIV_WS_URL   = (
    f"wss://ws.derivws.com/websockets/v3"
    f"?app_id={DERIV_APP_ID}"
)

BUFFER_SIZE    = 10_000
HISTORY_COUNT  = 2_000
MC_PATHS       = 10_000
MC_STEPS_MAX   = 600
JUMP_THRESHOLD = 4.5
MIN_TICKS      = 80
ALERT_WARMUP   = 30
EWMA_LAMBDA    = 0.94

# FIX v4.3: Ensemble agreement flag and floor
ENABLE_ENSEMBLE          = True
ENSEMBLE_AGREEMENT_FLOOR = 0.35

# FIX v4.3: Empirical cap constants
EMPIRICAL_CAP_PERCENTILE = 80
EMPIRICAL_CAP_MULTIPLIER = 1.5
EMPIRICAL_MIN_WINDOWS    = 20

# FIX v4.3: Refiner cold-memory threshold
REFINER_MIN_PATTERNS     = 5

# ── Signal Quality Gates (v4.3) ───────────────────────────────────────────────
MIN_PROB_TARGET  = 0.53
MIN_EDGE_PCT     = 0.0
MIN_EV           = 0.0
MIN_TARGET_MOVE  = 0.001
MIN_RR_GLOBAL    = 1.4

# ── Time-of-Day config ────────────────────────────────────────────────────────
TOD_BUCKETS  = 48
TOD_ALPHA    = 0.12
TOD_MULT_MIN = 0.88
TOD_MULT_MAX = 1.12

PERSIST_FILE = "sqe_v43_state.pkl"
PATTERN_FILE = "pattern_memory_v43.pkl"

# ─────────────────────────────────────────────────────────────────────────────
# TIMEFRAMES & SYMBOLS
# ─────────────────────────────────────────────────────────────────────────────
TIMEFRAMES: Dict[str, str] = {
    "1m":  "1 Minute",
    "5m":  "5 Minutes",
    "15m": "15 Minutes",
    "30m": "30 Minutes",
    "1h":  "1 Hour",
}
TF_SECONDS: Dict[str, int] = {
    "1m": 60,   "5m": 300,
    "15m": 900, "30m": 1800, "1h": 3600,
}
HORIZONS: Dict[str, int] = {
    "50t": 50, "100t": 100,
    "300t": 300, "600t": 600,
}
HORIZON_LABELS: Dict[str, str] = {
    "50t":  "50 Ticks",
    "100t": "100 Ticks",
    "300t": "300 Ticks",
    "600t": "600 Ticks",
}

VOLATILITY_SYMBOLS: Dict[str, str] = {
    "V10":      "R_10",
    "V25":      "R_25",
    "V50":      "R_50",
    "V75":      "R_75",
    "V100":     "R_100",
    "V10(1s)":  "1HZ10V",
    "V25(1s)":  "1HZ25V",
    "V50(1s)":  "1HZ50V",
    "V75(1s)":  "1HZ75V",
    "V100(1s)": "1HZ100V",
    "V250":     "R_250",
}
BOOM_SYMBOLS: Dict[str, str] = {
    "Boom300":  "BOOM300N",
    "Boom500":  "BOOM500",
    "Boom600":  "BOOM600N",
    "Boom900":  "BOOM900",
    "Boom1000": "BOOM1000",
}
CRASH_SYMBOLS: Dict[str, str] = {
    "Crash300":  "CRASH300N",
    "Crash500":  "CRASH500",
    "Crash600":  "CRASH600N",
    "Crash900":  "CRASH900",
    "Crash1000": "CRASH1000",
}
STEP_SYMBOLS: Dict[str, str] = {
    "Step Index": "stpRNG",
}
JUMP_SYMBOLS: Dict[str, str] = {
    "Jump10":  "JD10",
    "Jump25":  "JD25",
    "Jump50":  "JD50",
    "Jump75":  "JD75",
    "Jump100": "JD100",
}
ALL_SYMBOLS: Dict[str, str] = {
    **VOLATILITY_SYMBOLS,
    **BOOM_SYMBOLS,
    **CRASH_SYMBOLS,
    **STEP_SYMBOLS,
    **JUMP_SYMBOLS,
}
REVERSE_MAP: Dict[str, str] = {
    v: k for k, v in ALL_SYMBOLS.items()
}
EXPECTED_JUMP_FREQ: Dict[str, int] = {
    "BOOM300N": 300,  "BOOM500": 500,
    "BOOM600N": 600,  "BOOM900": 900,
    "BOOM1000": 1000,
    "CRASH300N": 300, "CRASH500": 500,
    "CRASH600N": 600, "CRASH900": 900,
    "CRASH1000": 1000,
    "JD10": 10,  "JD25": 25,  "JD50": 50,
    "JD75": 75,  "JD100": 100,
}

ADVERTISED_VOL: Dict[str, float] = {
    "R_10": 0.10,    "R_25": 0.25,
    "R_50": 0.50,    "R_75": 0.75,
    "R_100": 1.00,   "R_250": 2.50,
    "1HZ10V": 0.10,  "1HZ25V": 0.25,
    "1HZ50V": 0.50,  "1HZ75V": 0.75,
    "1HZ100V": 1.00,
    "JD10": 0.10,    "JD25": 0.25,
    "JD50": 0.50,    "JD75": 0.75,
    "JD100": 1.00,   "stpRNG": 0.01,
}

# ── Chart theme ───────────────────────────────────────────────────────────────
BG  = "#0a0e1a"; AX  = "#0f1629"
GR  = "#00ff88"; RD  = "#ff3355"
BL  = "#3d8eff"; YL  = "#ffd700"
OR  = "#ff8c00"; PU  = "#b44fff"
CY  = "#00e5ff"; GY  = "#5a6680"
WH  = "#e8eaf6"; DG  = "#1e2540"
GR2 = "#00cc6a"; RD2 = "#cc2244"

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _cat(sym: str) -> str:
    f = REVERSE_MAP.get(sym, sym)
    if "Boom"  in f: return "boom"
    if "Crash" in f: return "crash"
    if "Jump"  in f: return "jump"
    if "Step"  in f: return "step"
    return "vol"

def _friendly(sym: str) -> str:
    return REVERSE_MAP.get(sym, sym)

def _resolve(s: str) -> Optional[str]:
    s = s.strip().upper().replace(" ", "")
    for k, v in ALL_SYMBOLS.items():
        if (k.upper().replace(" ", "") == s
                or v.upper() == s):
            return v
    for k, v in ALL_SYMBOLS.items():
        if s in k.upper() or s in v.upper():
            return v
    return None

def _se(sig: str) -> str:
    return {
        "bullish":        "🟢",
        "bearish":        "🔴",
        "jump_imminent":  "⚡",
        "spike_imminent": "🚨",
        "neutral":        "⬜",
    }.get(sig, "❓")

def _esc(text: str) -> str:
    return (str(text)
            .replace("`", "'")
            .replace("*", "")
            .replace("_", ""))

def _safe_md(val, fmt: str) -> str:
    return _esc(format(val, fmt))

def _tod_bucket() -> int:
    now = datetime.now(timezone.utc)
    return now.hour * 2 + (1 if now.minute >= 30 else 0)

# FIX v4.3: Empirical move cap helper
# Computes max realistic target from actual
# historical rolling window moves
def _empirical_move_cap(
        prices: np.ndarray,
        horizon: int,
        cap_percentile: float = EMPIRICAL_CAP_PERCENTILE,
        cap_multiplier: float = EMPIRICAL_CAP_MULTIPLIER,
        min_windows: int      = EMPIRICAL_MIN_WINDOWS,
) -> Optional[float]:
    """
    Returns empirical maximum move as a fraction
    e.g. 0.025 = 2.5% by rolling through actual
    price history in horizon-sized windows.
    Returns None if insufficient data.
    """
    if prices is None or len(prices) < horizon * 2:
        return None
    step  = max(1, horizon // 4)
    moves = []
    for i in range(0, len(prices) - horizon, step):
        p0 = prices[i]
        p1 = prices[i + horizon]
        if p0 > 0:
            moves.append(abs(p1 - p0) / p0)
    if len(moves) < min_windows:
        return None
    pct_move = float(np.percentile(moves, cap_percentile))
    return pct_move * cap_multiplier

RISK_MSG = (
    "\n\n*Risk Warning:* No model guarantees profit. "
    "Max 1-2% account risk per trade. "
    "Statistical edge does not guarantee "
    "future results."
)
MAX_CAPTION = 950

async def _safe_send_photo(
        bot, chat_id: int,
        photo, caption: str, **kwargs):
    try:
        if len(caption) <= MAX_CAPTION:
            return await bot.send_photo(
                chat_id, photo=photo,
                caption=caption,
                parse_mode=ParseMode.MARKDOWN,
                **kwargs)
        short = caption[:MAX_CAPTION].rsplit("\n", 1)[0]
        await bot.send_photo(
            chat_id, photo=photo,
            caption=short,
            parse_mode=ParseMode.MARKDOWN,
            **kwargs)
        rest = caption[len(short):]
        if rest.strip():
            await _safe_send_message(bot, chat_id, rest)
    except Exception as e:
        log.warning(f"send_photo error: {e}")
        try:
            plain = caption.replace("*", "").replace("`", "")
            await bot.send_photo(
                chat_id, photo=photo,
                caption=plain[:MAX_CAPTION])
        except Exception as e2:
            log.error(f"send_photo fallback: {e2}")

async def _safe_send_message(
        bot, chat_id: int,
        text: str, **kwargs):
    MAX_MSG = 4000
    chunks = [
        text[i:i+MAX_MSG]
        for i in range(0, len(text), MAX_MSG)
    ]
    for idx, chunk in enumerate(chunks):
        kw = kwargs.copy() if idx == 0 else {}
        try:
            await bot.send_message(
                chat_id, chunk,
                parse_mode=ParseMode.MARKDOWN,
                **kw)
        except Exception as e:
            if any(x in str(e).lower()
                   for x in ["parse", "entity", "can't"]):
                plain = (chunk
                         .replace("*", "")
                         .replace("`", "")
                         .replace("_", ""))
                try:
                    await bot.send_message(chat_id, plain)
                except Exception as e2:
                    log.error(f"send_message fallback: {e2}")
            else:
                log.error(f"send_message: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# TIME-OF-DAY PROFILE
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TodBucket:
    count:       float = 0.0
    win_rate:    float = 0.50
    avg_edge:    float = 0.0
    avg_ev:      float = 0.0
    avg_vol_dev: float = 0.0
    tick_rate:   float = 0.5
    multiplier:  float = 1.0

class TimeOfDayProfile:
    def __init__(self):
        self._buckets: Dict[str, List[TodBucket]] = {}

    def _ensure(self, sym: str):
        if sym not in self._buckets:
            self._buckets[sym] = [
                TodBucket() for _ in range(TOD_BUCKETS)]

    def update(
            self, sym: str,
            win: bool, edge: float,
            ev: float, vol_dev: float,
            tick_rate: float,
            bucket: Optional[int] = None):
        self._ensure(sym)
        b_idx = bucket if bucket is not None else _tod_bucket()
        b     = self._buckets[sym][b_idx]
        a     = TOD_ALPHA
        b.count      += 1.0
        b.win_rate    = (1-a)*b.win_rate + a*(1.0 if win else 0.0)
        b.avg_edge    = (1-a)*b.avg_edge    + a*edge
        b.avg_ev      = (1-a)*b.avg_ev      + a*ev
        b.avg_vol_dev = (1-a)*b.avg_vol_dev + a*vol_dev
        b.tick_rate   = (1-a)*b.tick_rate   + a*tick_rate
        wr_factor   = float(np.clip(1.0+(b.win_rate-0.50)*0.60, 0.80, 1.25))
        edge_factor = float(np.clip(1.0+b.avg_edge*0.30, 0.90, 1.15))
        conf_weight = float(np.clip(b.count/20.0, 0.0, 1.0))
        raw_mult    = 1.0 + conf_weight*((wr_factor*edge_factor)-1.0)
        b.multiplier = float(np.clip(raw_mult, TOD_MULT_MIN, TOD_MULT_MAX))

    def multiplier(self, sym: str, bucket: Optional[int] = None) -> float:
        self._ensure(sym)
        b_idx = bucket if bucket is not None else _tod_bucket()
        b     = self._buckets[sym][b_idx]
        if b.count < 3:
            return 1.0
        return b.multiplier

    def prob_adjustment(self, sym: str, bucket: Optional[int] = None) -> float:
        mult = self.multiplier(sym, bucket)
        return float(np.clip((mult-1.0)*0.05, -0.03, 0.03))

    def best_horizon(self, sym: str, available: List[int]) -> int:
        self._ensure(sym)
        b_idx = _tod_bucket()
        b     = self._buckets[sym][b_idx]
        if b.count < 5:
            return 300
        if b.avg_vol_dev > 0.15:
            preferred = [h for h in available if h <= 100]
            return preferred[0] if preferred else 300
        elif b.avg_vol_dev < -0.10:
            preferred = [h for h in available if h >= 300]
            return preferred[0] if preferred else 300
        return 300

    def summary(self, sym: str, bucket: Optional[int] = None) -> str:
        self._ensure(sym)
        b_idx = bucket if bucket is not None else _tod_bucket()
        b     = self._buckets[sym][b_idx]
        hour  = b_idx // 2
        mins  = "30" if b_idx % 2 else "00"
        return (
            f"ToD [{hour:02d}:{mins} UTC] "
            f"n={b.count:.0f} wr={b.win_rate:.0%} "
            f"edge={b.avg_edge:.3f} mult={b.multiplier:.3f}"
        )

    def to_dict(self) -> dict:
        return {
            sym: [
                {
                    "count":       bk.count,
                    "win_rate":    bk.win_rate,
                    "avg_edge":    bk.avg_edge,
                    "avg_ev":      bk.avg_ev,
                    "avg_vol_dev": bk.avg_vol_dev,
                    "tick_rate":   bk.tick_rate,
                    "multiplier":  bk.multiplier,
                }
                for bk in buckets
            ]
            for sym, buckets in self._buckets.items()
        }

    def from_dict(self, d: dict):
        for sym, blist in d.items():
            self._buckets[sym] = []
            for bk in blist:
                tb             = TodBucket()
                tb.count       = bk.get("count",       0.0)
                tb.win_rate    = bk.get("win_rate",     0.50)
                tb.avg_edge    = bk.get("avg_edge",     0.0)
                tb.avg_ev      = bk.get("avg_ev",       0.0)
                tb.avg_vol_dev = bk.get("avg_vol_dev",  0.0)
                tb.tick_rate   = bk.get("tick_rate",    0.5)
                tb.multiplier  = bk.get("multiplier",   1.0)
                self._buckets[sym].append(tb)

# ─────────────────────────────────────────────────────────────────────────────
# ASSET CLASS PROFILE SYSTEM
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class AssetProfile:
    name: str
    model_type: str            = "GBM"
    edge_min: float            = 0.20
    direction_gap: float       = 0.008
    ci_width_min: float        = 0.000005
    signal_edge_min: float     = 0.15
    strong_edge: float         = 0.25
    strong_prob: float         = 0.58
    strong_rr: float           = 2.0
    moderate_edge: float       = 0.13
    moderate_prob: float       = 0.54
    moderate_rr: float         = 1.4
    trade_enabled: bool        = True
    min_target_pct: float      = 0.001
    min_stop_pct: float        = 0.001
    min_rr: float              = 1.4
    target_percentile: float   = 60.0
    stop_percentile: float     = 40.0
    direction_bias: str        = "NONE"
    spike_enabled: bool        = False
    spike_min_prob: float      = 0.45
    spike_min_magnitude: float = 0.03
    spike_lam_dev_min: float   = 8.0
    spike_hazard_high: float   = 0.65
    spike_is_primary: bool     = False
    alert_edge_min: float      = 0.20
    alert_cooldown: float      = 300.0
    alert_min_ticks: int       = 400
    alert_min_rr: float        = 1.4
    alert_strength: List[str]  = field(
        default_factory=lambda: ["STRONG", "MODERATE"])
    mc_paths: int                = 10_000
    sigma_multiplier_base: float = 1.5
    drift_signal_enabled: bool   = False
    drift_tstat_min: float       = 2.0
    drift_window: int            = 200
    drift_edge_weight: float     = 0.55
    vol_regime_enabled: bool     = False
    vol_k_min: float             = 2.0
    vol_k_max: float             = 2.8
    resolve_threshold_pct: float = 0.05
    tod_enabled: bool            = True

PROFILE_VOL = AssetProfile(
    name="VOLATILITY",
    model_type="GBM",
    edge_min=0.12,
    direction_gap=0.003,
    ci_width_min=0.000001,
    signal_edge_min=0.10,
    strong_edge=0.25,
    strong_prob=0.58,
    strong_rr=2.0,
    moderate_edge=0.12,
    moderate_prob=0.54,
    moderate_rr=1.4,
    trade_enabled=True,
    min_target_pct=0.001,
    min_stop_pct=0.001,
    min_rr=1.4,
    target_percentile=65.0,
    stop_percentile=35.0,
    direction_bias="NONE",
    spike_enabled=False,
    alert_edge_min=0.12,
    alert_cooldown=360.0,
    alert_min_ticks=400,
    alert_min_rr=1.4,
    alert_strength=["STRONG", "MODERATE"],
    mc_paths=10_000,
    sigma_multiplier_base=2.2,
    drift_signal_enabled=True,
    drift_tstat_min=2.0,
    drift_window=200,
    drift_edge_weight=0.55,
    vol_regime_enabled=True,
    vol_k_min=2.0,
    vol_k_max=2.8,
    resolve_threshold_pct=0.02,
    tod_enabled=True,
)

PROFILE_BOOM = AssetProfile(
    name="BOOM",
    model_type="JumpDiffusion",
    edge_min=0.25,
    direction_gap=0.010,
    ci_width_min=0.00001,
    signal_edge_min=0.15,
    strong_edge=0.40,
    strong_prob=0.58,
    strong_rr=2.0,
    moderate_edge=0.20,
    moderate_prob=0.54,
    moderate_rr=1.4,
    trade_enabled=True,
    min_target_pct=0.0025,
    min_stop_pct=0.005,
    min_rr=1.4,
    target_percentile=65.0,
    stop_percentile=35.0,
    direction_bias="BUY",
    spike_enabled=True,
    spike_min_prob=0.45,
    spike_min_magnitude=0.03,
    spike_lam_dev_min=8.0,
    spike_hazard_high=0.65,
    spike_is_primary=True,
    alert_edge_min=0.20,
    alert_cooldown=300.0,
    alert_min_ticks=400,
    alert_min_rr=1.4,
    alert_strength=["STRONG", "MODERATE"],
    mc_paths=10_000,
    sigma_multiplier_base=2.2,
    drift_signal_enabled=False,
    vol_regime_enabled=False,
    resolve_threshold_pct=0.10,
    tod_enabled=True,
)

PROFILE_CRASH = AssetProfile(
    name="CRASH",
    model_type="JumpDiffusion",
    edge_min=0.25,
    direction_gap=0.010,
    ci_width_min=0.00001,
    signal_edge_min=0.15,
    strong_edge=0.40,
    strong_prob=0.58,
    strong_rr=2.0,
    moderate_edge=0.20,
    moderate_prob=0.54,
    moderate_rr=1.4,
    trade_enabled=True,
    min_target_pct=0.0025,
    min_stop_pct=0.005,
    min_rr=1.4,
    target_percentile=35.0,
    stop_percentile=65.0,
    direction_bias="SELL",
    spike_enabled=True,
    spike_min_prob=0.45,
    spike_min_magnitude=0.03,
    spike_lam_dev_min=8.0,
    spike_hazard_high=0.65,
    spike_is_primary=True,
    alert_edge_min=0.20,
    alert_cooldown=300.0,
    alert_min_ticks=400,
    alert_min_rr=1.4,
    alert_strength=["STRONG", "MODERATE"],
    mc_paths=10_000,
    sigma_multiplier_base=2.2,
    drift_signal_enabled=False,
    vol_regime_enabled=False,
    resolve_threshold_pct=0.10,
    tod_enabled=True,
)

PROFILE_JUMP = AssetProfile(
    name="JUMP",
    model_type="JumpDiffusion",
    edge_min=0.15,
    direction_gap=0.005,
    ci_width_min=0.000005,
    signal_edge_min=0.12,
    strong_edge=0.28,
    strong_prob=0.57,
    strong_rr=2.0,
    moderate_edge=0.15,
    moderate_prob=0.54,
    moderate_rr=1.4,
    trade_enabled=True,
    min_target_pct=0.001,
    min_stop_pct=0.005,
    min_rr=1.4,
    target_percentile=62.0,
    stop_percentile=38.0,
    direction_bias="NONE",
    spike_enabled=False,
    spike_is_primary=False,
    alert_edge_min=0.15,
    alert_cooldown=180.0,
    alert_min_ticks=200,
    alert_min_rr=1.4,
    alert_strength=["STRONG", "MODERATE"],
    mc_paths=10_000,
    sigma_multiplier_base=2.0,
    drift_signal_enabled=True,
    drift_tstat_min=1.8,
    drift_window=150,
    drift_edge_weight=0.45,
    vol_regime_enabled=False,
    resolve_threshold_pct=0.05,
    tod_enabled=True,
)

PROFILE_STEP = AssetProfile(
    name="STEP",
    model_type="Step",
    edge_min=0.10,
    direction_gap=0.002,
    ci_width_min=0.0000001,
    signal_edge_min=0.08,
    strong_edge=0.22,
    strong_prob=0.56,
    strong_rr=1.5,
    moderate_edge=0.12,
    moderate_prob=0.54,
    moderate_rr=1.4,
    trade_enabled=True,
    min_target_pct=0.0001,
    min_stop_pct=0.00005,
    min_rr=1.4,
    target_percentile=65.0,
    stop_percentile=35.0,
    direction_bias="NONE",
    spike_enabled=False,
    alert_edge_min=0.10,
    alert_cooldown=360.0,
    alert_min_ticks=300,
    alert_min_rr=1.4,
    alert_strength=["STRONG", "MODERATE"],
    mc_paths=8_000,
    sigma_multiplier_base=1.0,
    drift_signal_enabled=True,
    drift_tstat_min=2.2,
    drift_window=300,
    drift_edge_weight=0.55,
    vol_regime_enabled=False,
    resolve_threshold_pct=0.01,
    tod_enabled=True,
)

@dataclass
class SymbolTune:
    alert_cooldown_override: Optional[float]  = None
    min_target_pct_override: Optional[float]  = None
    min_rr_override: Optional[float]          = None
    spike_min_prob_override: Optional[float]  = None
    mc_paths_override: Optional[int]          = None
    sigma_mult_override: Optional[float]      = None
    drift_tstat_override: Optional[float]     = None
    drift_window_override: Optional[int]      = None
    vol_k_min_override: Optional[float]       = None
    vol_k_max_override: Optional[float]       = None

SYMBOL_TUNES: Dict[str, SymbolTune] = {
    "R_10":    SymbolTune(
        min_target_pct_override=0.001,
        min_rr_override=1.4,
        sigma_mult_override=1.2,
        drift_tstat_override=2.0,
        drift_window_override=300,
        vol_k_min_override=1.8,
        vol_k_max_override=2.5),
    "R_25":    SymbolTune(
        min_target_pct_override=0.0015,
        min_rr_override=1.4,
        sigma_mult_override=1.3,
        drift_tstat_override=2.0,
        vol_k_min_override=1.9,
        vol_k_max_override=2.6),
    "R_50":    SymbolTune(
        min_target_pct_override=0.002,
        min_rr_override=1.4,
        sigma_mult_override=1.6,
        vol_k_min_override=1.8,
        vol_k_max_override=2.5),
    "R_75":    SymbolTune(
        min_target_pct_override=0.002,
        min_rr_override=1.4,
        sigma_mult_override=1.8,
        vol_k_min_override=2.0,
        vol_k_max_override=2.6),
    "R_100":   SymbolTune(
        min_target_pct_override=0.003,
        min_rr_override=1.4,
        sigma_mult_override=2.2,
        vol_k_min_override=2.2,
        vol_k_max_override=2.8),
    "R_250":   SymbolTune(
        min_target_pct_override=0.005,
        min_rr_override=1.4,
        sigma_mult_override=2.5,
        vol_k_min_override=2.3,
        vol_k_max_override=2.8),
    "1HZ10V":  SymbolTune(
        min_target_pct_override=0.001,
        min_rr_override=1.4,
        sigma_mult_override=1.2,
        drift_tstat_override=2.0,
        vol_k_min_override=1.8,
        vol_k_max_override=2.5),
    "1HZ25V":  SymbolTune(
        min_target_pct_override=0.0015,
        min_rr_override=1.4,
        sigma_mult_override=1.3,
        vol_k_min_override=1.9,
        vol_k_max_override=2.6),
    "1HZ50V":  SymbolTune(
        min_target_pct_override=0.002,
        min_rr_override=1.4,
        sigma_mult_override=1.6,
        vol_k_min_override=1.8,
        vol_k_max_override=2.5),
    "1HZ75V":  SymbolTune(
        min_target_pct_override=0.002,
        min_rr_override=1.4,
        sigma_mult_override=1.8,
        vol_k_min_override=2.0,
        vol_k_max_override=2.6),
    "1HZ100V": SymbolTune(
        min_target_pct_override=0.003,
        min_rr_override=1.4,
        sigma_mult_override=2.2,
        vol_k_min_override=2.2,
        vol_k_max_override=2.8),
    "BOOM300N":  SymbolTune(
        alert_cooldown_override=180.0,
        spike_min_prob_override=0.50),
    "BOOM500":   SymbolTune(
        alert_cooldown_override=240.0,
        spike_min_prob_override=0.45),
    "BOOM600N":  SymbolTune(
        alert_cooldown_override=270.0,
        spike_min_prob_override=0.45),
    "BOOM900":   SymbolTune(
        alert_cooldown_override=360.0,
        spike_min_prob_override=0.42),
    "BOOM1000":  SymbolTune(
        alert_cooldown_override=400.0,
        spike_min_prob_override=0.40),
    "CRASH300N": SymbolTune(
        alert_cooldown_override=180.0,
        spike_min_prob_override=0.50),
    "CRASH500":  SymbolTune(
        alert_cooldown_override=240.0,
        spike_min_prob_override=0.45),
    "CRASH600N": SymbolTune(
        alert_cooldown_override=270.0,
        spike_min_prob_override=0.45),
    "CRASH900":  SymbolTune(
        alert_cooldown_override=360.0,
        spike_min_prob_override=0.42),
    "CRASH1000": SymbolTune(
        alert_cooldown_override=400.0,
        spike_min_prob_override=0.40),
    "JD10":  SymbolTune(
        alert_cooldown_override=60.0,
        mc_paths_override=8_000,
        drift_tstat_override=1.8,
        drift_window_override=100),
    "JD25":  SymbolTune(
        alert_cooldown_override=90.0,
        drift_tstat_override=1.8),
    "JD50":  SymbolTune(alert_cooldown_override=120.0),
    "JD75":  SymbolTune(alert_cooldown_override=150.0),
    "JD100": SymbolTune(alert_cooldown_override=180.0),
}

def get_profile(sym: str) -> AssetProfile:
    cat = _cat(sym)
    return {
        "vol":   PROFILE_VOL,
        "boom":  PROFILE_BOOM,
        "crash": PROFILE_CRASH,
        "jump":  PROFILE_JUMP,
        "step":  PROFILE_STEP,
    }.get(cat, PROFILE_VOL)

def get_tune(sym: str) -> SymbolTune:
    return SYMBOL_TUNES.get(sym, SymbolTune())

def effective_profile(sym: str) -> AssetProfile:
    import copy
    p    = copy.deepcopy(get_profile(sym))
    tune = get_tune(sym)
    if tune.alert_cooldown_override is not None:
        p.alert_cooldown = tune.alert_cooldown_override
    if tune.min_target_pct_override is not None:
        p.min_target_pct = tune.min_target_pct_override
    if tune.min_rr_override is not None:
        p.min_rr       = tune.min_rr_override
        p.alert_min_rr = tune.min_rr_override
    if tune.spike_min_prob_override is not None:
        p.spike_min_prob = tune.spike_min_prob_override
    if tune.mc_paths_override is not None:
        p.mc_paths = tune.mc_paths_override
    if tune.sigma_mult_override is not None:
        p.sigma_multiplier_base = tune.sigma_mult_override
    if tune.drift_tstat_override is not None:
        p.drift_tstat_min = tune.drift_tstat_override
    if tune.drift_window_override is not None:
        p.drift_window = tune.drift_window_override
    if tune.vol_k_min_override is not None:
        p.vol_k_min = tune.vol_k_min_override
    if tune.vol_k_max_override is not None:
        p.vol_k_max = tune.vol_k_max_override
    return p

# ─────────────────────────────────────────────────────────────────────────────
# NUMBA KERNELS
# ─────────────────────────────────────────────────────────────────────────────
@njit(parallel=True, cache=True, fastmath=True)
def _gbm_kernel(S0, mu, sigma, dt, n_paths, n_steps, Z):
    paths = np.empty((n_paths, n_steps+1), dtype=np.float64)
    drift = (mu - 0.5*sigma*sigma)*dt
    vol   = sigma*math.sqrt(dt)
    for i in prange(n_paths):
        paths[i, 0] = S0
        for t in range(n_steps):
            paths[i, t+1] = paths[i, t]*math.exp(drift + vol*Z[i, t])
    return paths

@njit(parallel=True, cache=True, fastmath=True)
def _jd_kernel(S0, mu, sigma, lam, jmean, jstd, jsign,
               dt, n_paths, n_steps, Z, U, Zj):
    paths  = np.empty((n_paths, n_steps+1), dtype=np.float64)
    drift  = (mu - 0.5*sigma*sigma)*dt
    vol    = sigma*math.sqrt(dt)
    lam_dt = lam*dt
    for i in prange(n_paths):
        paths[i, 0] = S0
        for t in range(n_steps):
            gbm  = drift + vol*Z[i, t]
            jump = 0.0
            if U[i, t] < lam_dt:
                raw  = jmean + jstd*abs(Zj[i, t])
                jump = jsign*raw
            paths[i, t+1] = paths[i, t]*math.exp(gbm+jump)
    return paths

# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class GBMParams:
    mu: float             = 0.0
    sigma: float          = 0.01
    sigma_ewma: float     = 0.01
    n_obs: int            = 0
    advertised_vol: float = 0.0
    ks_pvalue: float      = 1.0
    fit_quality: float    = 1.0

@dataclass
class VolRegimeResult:
    regime: str
    realized_sigma: float
    target_sigma: float
    vol_deviation: float
    normalized_momentum: float
    momentum_direction: str
    k_adaptive: float
    target_up: float
    target_down: float
    stop_up: float
    stop_down: float
    edge_score: float
    signal: str
    signal_strength: str
    tod_multiplier: float

@dataclass
class DriftSignal:
    direction: str
    tstat: float
    pvalue: float
    drift_per_tick: float
    drift_pct: float
    regime: str
    ewma_momentum: float
    window_used: int
    edge_score: float
    signal_strength: str

@dataclass
class JumpParams:
    mu: float               = 0.0
    sigma: float            = 0.01
    lam: float              = 0.002
    lam_posterior: float    = 0.002
    jump_mean: float        = 0.02
    jump_std: float         = 0.01
    jump_sign: float        = 1.0
    n_jumps: int            = 0
    n_obs: int              = 0
    expected_lam: float     = 0.002
    lam_deviation: float    = 0.0
    recent_cluster: float   = 0.0
    inter_arrival_cv: float = 1.0
    hazard_intensity: float = 0.0
    counter_drift: float    = 0.0
    spike_mag_p60: float    = 0.02
    spike_mag_p75: float    = 0.03

@dataclass
class SpikeAlert:
    symbol: str
    fname: str
    direction: str
    confidence: float
    poisson_prob: float
    hazard_intensity: float
    ticks_lo: int
    ticks_hi: int
    time_lo_str: str
    time_hi_str: str
    time_center_str: str
    magnitude_lo: float
    magnitude_hi: float
    magnitude_p60: float
    magnitude_p75: float
    reasons: List[str]
    is_imminent: bool    = True
    counter_drift: float = 0.0
    tod_multiplier: float = 1.0
    timestamp: float     = field(default_factory=time.time)
    resolved: bool       = False
    was_correct: bool    = False

@dataclass
class TradeSetup:
    direction: str
    entry: float
    target: float
    invalidation: float
    target_pct: float
    stop_pct: float
    rr_ratio: float
    prob_target: float
    prob_stop: float
    edge_pct: float
    expected_value: float
    signal_strength: str
    timeframe_label: str
    horizon_ticks: int
    regime: str
    profile_name: str     = "UNKNOWN"
    signal_source: str    = "MC"
    tod_multiplier: float = 1.0

@dataclass
class MCResult:
    symbol: str
    S0: float
    horizon: int
    horizon_label: str
    paths: np.ndarray
    p5:  np.ndarray
    p25: np.ndarray
    p50: np.ndarray
    p75: np.ndarray
    p95: np.ndarray
    target_up: float
    target_down: float
    target_label: str
    prob_up_horizon: float
    prob_hit_up: float
    prob_hit_down: float
    sigma_horizon: float
    ci_width_pct: float
    prob_jump: Optional[float]         = None
    exp_jump_tick: Optional[float]     = None
    jump_magnitude_lo: Optional[float] = None
    jump_magnitude_hi: Optional[float] = None
    edge_score: float                  = 0.0
    signal: str                        = "neutral"
    implied_edge_pct: float            = 0.0
    trade_setup: Optional[TradeSetup]  = None
    timeframe_key: Optional[str]       = None
    regime: str                        = "NORMAL"
    profile_name: str                  = "UNKNOWN"
    drift_signal: Optional[DriftSignal]   = None
    vol_regime: Optional[VolRegimeResult] = None
    tod_multiplier: float                 = 1.0
    ens_agreement: float                  = 1.0

@dataclass
class PatternRecord:
    symbol: str
    horizon: int
    timestamp: float
    prediction: str
    prob: float
    edge_score: float
    S0: float
    target_up: float
    target_down: float
    resolved: bool     = False
    correct: bool      = False
    outcome_pct: float = 0.0
    result_label: str  = ""

@dataclass
class UserState:
    pending_sym: Optional[str] = None
    horizon_key: str           = "300t"
    timeframe_key: str         = "5m"
    watchlist: List[str]       = field(default_factory=list)
    risk_pct: float            = 1.0
    alerts_enabled: bool       = False

# ─────────────────────────────────────────────────────────────────────────────
# PERSISTENCE ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class PersistenceEngine:
    def __init__(self):
        self.data: dict = {}
        self._load()

    def _load(self):
        for fn in (
            PERSIST_FILE,
            "sqe_v42_state.pkl",
            "sqe_v41_state.pkl",
            "sqe_v40_state.pkl",
        ):
            if os.path.exists(fn):
                try:
                    with open(fn, "rb") as f:
                        self.data = pickle.load(f)
                    log.info(f"Persistence: loaded {fn}")
                    return
                except Exception as e:
                    log.warning(f"Persistence load {fn}: {e}")
                    self.data = {}

    def save(self):
        try:
            with open(PERSIST_FILE, "wb") as f:
                pickle.dump(self.data, f)
        except Exception as e:
            log.debug(f"Persistence save: {e}")

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value
        self.save()

    def get_user_states(self) -> Dict[int, UserState]:
        return self.data.get("user_states", {})

    def save_user_states(self, states: Dict[int, UserState]):
        self.data["user_states"] = states
        self.save()

    def get_alert_last(self) -> Dict[str, float]:
        return self.data.get("alert_last", {})

    def save_alert_last(self, d: Dict[str, float]):
        self.data["alert_last"] = d
        self.save()

    def get_chats(self) -> Set[int]:
        return self.data.get("chats", set())

    def save_chats(self, chats: Set[int]):
        self.data["chats"] = chats
        self.save()

    def get_tod_data(self) -> dict:
        return self.data.get("tod_profiles", {})

    def save_tod_data(self, d: dict):
        self.data["tod_profiles"] = d
        self.save()

# ─────────────────────────────────────────────────────────────────────────────
# PATTERN MEMORY
# ─────────────────────────────────────────────────────────────────────────────
class PatternMemory:
    def __init__(self, tod_profile: Optional[TimeOfDayProfile] = None):
        self.records: List[PatternRecord] = []
        self._tod = tod_profile
        self._load()

    def _load(self):
        for fn in (
            PATTERN_FILE,
            "pattern_memory_v42.pkl",
            "pattern_memory_v41.pkl",
            "pattern_memory_v40.pkl",
        ):
            if os.path.exists(fn):
                try:
                    with open(fn, "rb") as f:
                        self.records = pickle.load(f)
                    log.info(f"Pattern memory: {len(self.records)} records.")
                    return
                except Exception:
                    self.records = []

    def _save(self):
        try:
            with open(PATTERN_FILE, "wb") as f:
                pickle.dump(self.records, f)
        except Exception as e:
            log.debug(f"Pattern save: {e}")

    def record(self, sym, horizon, pred, prob, edge, S0, tup, tdn) -> PatternRecord:
        r = PatternRecord(
            symbol=sym, horizon=horizon,
            timestamp=time.time(),
            prediction=pred, prob=prob,
            edge_score=edge, S0=S0,
            target_up=tup, target_down=tdn,
            result_label="PENDING",
        )
        self.records.append(r)
        self._save()
        return r

    def resolve(self, sym: str, price: float,
                bot=None, chats: Optional[Set[int]] = None):
        now     = time.time()
        profile = effective_profile(sym)
        changed = False
        for r in self.records:
            if r.symbol != sym or r.resolved:
                continue
            if now - r.timestamp < r.horizon * 1.2:
                continue
            r.resolved    = True
            changed       = True
            r.outcome_pct = (price - r.S0) / r.S0 * 100
            thresh        = profile.resolve_threshold_pct / 100
            if r.prediction == "bullish":
                r.correct = price >= r.target_up
            elif r.prediction == "bearish":
                r.correct = price <= r.target_down
            elif r.prediction in ("spike_imminent", "jump_imminent"):
                r.correct = abs(r.outcome_pct / 100) > thresh
            else:
                r.correct = False
            r.result_label = "WIN" if r.correct else "LOSS"
            if self._tod is not None:
                bucket = int((r.timestamp % 86400) / 1800)
                self._tod.update(
                    sym, r.correct, r.edge_score,
                    r.outcome_pct, 0.0, 0.5,
                    bucket=bucket % TOD_BUCKETS)
            if bot and chats:
                icon = "✅ WIN" if r.correct else "❌ LOSS"
                msg  = (
                    f"*Signal Outcome — {_friendly(sym)}*\n\n"
                    f"Result: *{icon}*\n"
                    f"Prediction: `{r.prediction.upper()}`\n"
                    f"Entry: `{r.S0:.6f}`\n"
                    f"Exit:  `{price:.6f}`\n"
                    f"Move:  `{r.outcome_pct:+.4f}%`\n"
                    + ("Correct." if r.correct else "Incorrect.")
                )
                for cid in list(chats):
                    asyncio.ensure_future(
                        _safe_send_message(bot, cid, msg))
        if changed:
            self._save()

    def stats(self, sym, horizon=None) -> Tuple[int, float, float]:
        done = [
            r for r in self.records
            if r.symbol == sym and r.resolved
            and (horizon is None or abs(r.horizon - horizon) < 50)
        ]
        if not done:
            return 0, 0.5, 0.0
        wr = sum(r.correct for r in done) / len(done)
        ae = sum(r.edge_score for r in done) / len(done)
        return len(done), wr, ae

    def win_rate_weight(self, sym, horizon=None) -> float:
        n, wr, _ = self.stats(sym, horizon)
        if n < 5:
            return 1.0
        return float(np.clip(0.85 + (wr*0.30), 0.70, 1.20))

    def note(self, sym, horizon=None) -> str:
        n, wr, ae = self.stats(sym, horizon)
        if n < 3:
            return "Pattern memory: building (need 3+ resolved predictions)."
        e      = "🟢" if wr >= 0.55 else ("🔴" if wr < 0.45 else "🟡")
        wins   = sum(1 for r in self.records
                     if r.symbol == sym and r.resolved and r.correct)
        losses = n - wins
        return (
            f"{e} Accuracy: *{wr:.0%}* | "
            f"W:{wins} L:{losses} ({n} total) | "
            f"Avg edge: {ae:.2f}"
        )

    def recent_results(self, sym: str, limit: int = 5) -> str:
        done = sorted(
            [r for r in self.records
             if r.symbol == sym and r.resolved],
            key=lambda x: x.timestamp,
            reverse=True)[:limit]
        if not done:
            return "No resolved predictions yet."
        lines = [f"*Recent Results — {_friendly(sym)}:*"]
        for r in done:
            icon = "✅" if r.correct else "❌"
            dt   = datetime.fromtimestamp(
                r.timestamp, tz=timezone.utc
            ).strftime("%m-%d %H:%M")
            lines.append(
                f"{icon} {dt} | {r.prediction.upper()} | "
                f"{r.outcome_pct:+.3f}% | {r.result_label}")
        return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# HYBRID TARGET REFINER v4.3
# FIX v4.3 PROBLEM 1: Cold-memory path uses empirical
# cap instead of KDE on unreliable MC paths
# ─────────────────────────────────────────────────────────────────────────────
class HybridTargetRefiner:
    """
    Refines MC targets using pattern memory and empirical price history.

    Cold path  (n < REFINER_MIN_PATTERNS):
        Use empirical 80th-pct rolling move x 1.5 as primary target cap.
        Do NOT run KDE on MC paths — unreliable with no ground truth.

    Warm path  (n >= REFINER_MIN_PATTERNS):
        Blend KDE mode on MC finals (50%) with MC regime target (50%).
        Additional empirical cap still applied.
    """

    def refine(
            self,
            sym: str,
            S0: float,
            is_bull: bool,
            mc_target: float,
            mc_stop: float,
            finals: np.ndarray,
            prices: np.ndarray,
            horizon: int,
            records: list,
    ) -> Tuple[float, float, str]:
        n_records = len([
            r for r in records
            if r.symbol == sym and r.resolved])

        # FIX v4.3: Cold path
        pattern_memory_cold = (n_records < REFINER_MIN_PATTERNS)
        emp_cap = _empirical_move_cap(prices, horizon)

        if pattern_memory_cold:
            if emp_cap is not None:
                direction = 1.0 if is_bull else -1.0
                refined_target = S0 * (1.0 + direction * emp_cap)
                refined_stop   = S0 * (1.0 - direction * emp_cap * 0.40)
                log.debug(
                    f"HybridRefiner [{sym}]: COLD "
                    f"emp_cap={emp_cap*100:.3f}% n={n_records}")
                return refined_target, refined_stop, "empirical_cold"
            else:
                return mc_target, mc_stop, "mc_fallback"

        # FIX v4.3: Warm path — blend KDE with MC
        try:
            kde      = gaussian_kde(finals)
            x        = np.linspace(finals.min(), finals.max(), 500)
            kde_mode = float(x[np.argmax(kde(x))])
            blended  = 0.50 * kde_mode + 0.50 * mc_target

            if emp_cap is not None:
                raw_move = abs(blended - S0) / S0
                if raw_move > emp_cap:
                    direction = 1.0 if is_bull else -1.0
                    blended   = S0 * (1.0 + direction * emp_cap)

            log.debug(
                f"HybridRefiner [{sym}]: WARM "
                f"kde={kde_mode:.6f} blend={blended:.6f} n={n_records}")
            return blended, mc_stop, "kde_warm"

        except Exception as e:
            log.debug(f"HybridRefiner KDE err: {e}")
            return mc_target, mc_stop, "mc_fallback"

# ─────────────────────────────────────────────────────────────────────────────
# DATA MANAGER
# ─────────────────────────────────────────────────────────────────────────────
class DataManager:
    def __init__(self):
        self.buffers: Dict[str, deque] = {
            s: deque(maxlen=BUFFER_SIZE)
            for s in ALL_SYMBOLS.values()
        }
        self.timestamps: Dict[str, deque] = {
            s: deque(maxlen=BUFFER_SIZE)
            for s in ALL_SYMBOLS.values()
        }
        self.latest: Dict[str, float] = {}
        self._tick_times: Dict[str, deque] = {
            s: deque(maxlen=300)
            for s in ALL_SYMBOLS.values()
        }
        self._ws        = None
        self._running   = False
        self._connected = False
        self._init_done = False
        self._req_id    = 0
        self._id_lock   = asyncio.Lock()
        self._pending: Dict[int, asyncio.Future] = {}
        self._tick_cbs: List = []

    def add_tick_cb(self, fn):
        self._tick_cbs.append(fn)

    def prices(self, sym) -> np.ndarray:
        return np.array(list(self.buffers[sym]), dtype=np.float64)

    def log_returns(self, sym) -> np.ndarray:
        p = self.prices(sym)
        if len(p) < 2:
            return np.array([])
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.diff(np.log(p))
        return lr[np.isfinite(lr)]

    def last(self, sym) -> Optional[float]:
        return self.latest.get(sym)

    def n(self, sym) -> int:
        return len(self.buffers[sym])

    def observed_tick_rate(self, sym: str) -> float:
        times = list(self._tick_times[sym])
        if len(times) < 10:
            cat = _cat(sym)
            return {"boom": 0.5, "crash": 0.5,
                    "jump": 1.0, "step": 1.0}.get(cat, 0.5)
        diffs = np.diff(times)
        diffs = diffs[diffs > 0]
        if len(diffs) == 0:
            return 0.5
        return 1.0 / max(float(np.median(diffs)), 0.1)

    def ticks_for_timeframe(self, sym: str, tf_key: str) -> Tuple[int, str]:
        seconds = TF_SECONDS.get(tf_key, 300)
        rate    = self.observed_tick_rate(sym)
        ticks   = max(int(seconds * rate), 50)
        ticks   = min(ticks, MC_STEPS_MAX)
        label   = TIMEFRAMES.get(tf_key, tf_key)
        return ticks, label

    async def _next_id(self) -> int:
        async with self._id_lock:
            self._req_id += 1
            return self._req_id

    async def _send_wait(self, payload: dict, timeout: float = 30.0) -> dict:
        rid = await self._next_id()
        payload["req_id"] = rid
        loop = asyncio.get_event_loop()
        fut  = loop.create_future()
        self._pending[rid] = fut
        await self._ws.send(json.dumps(payload))
        try:
            return await asyncio.wait_for(asyncio.shield(fut), timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(rid, None)
            if not fut.done():
                fut.cancel()
            raise asyncio.TimeoutError(f"req_id={rid} timed out")

    async def _receive_loop(self):
        try:
            async for raw in self._ws:
                if not self._running:
                    break
                try:
                    msg   = json.loads(raw)
                    rid   = msg.get("req_id")
                    mtype = msg.get("msg_type", "")
                    if rid and rid in self._pending:
                        fut = self._pending.pop(rid)
                        if not fut.done():
                            fut.set_result(msg)
                    if mtype == "tick":
                        tick = msg["tick"]
                        sym  = tick.get("symbol", "")
                        if sym in self.buffers:
                            prc = float(tick["quote"])
                            ts  = float(tick["epoch"])
                            self.buffers[sym].append(prc)
                            self.timestamps[sym].append(ts)
                            self._tick_times[sym].append(ts)
                            self.latest[sym] = prc
                            for cb in self._tick_cbs:
                                asyncio.ensure_future(cb(sym, prc, ts))
                except Exception as e:
                    log.debug(f"Receive dispatch: {e}")
        except Exception as e:
            log.debug(f"Receive loop exited: {e}")

    async def _initialise(self):
        try:
            r = await self._send_wait({"ping": 1}, timeout=8.0)
            log.info(f"WS ping OK: {r.get('ping','?')}")
        except Exception as e:
            log.warning(f"Ping: {e}")

        log.info("Backfilling tick history ...")
        sem = asyncio.Semaphore(3)

        async def _fill(sym: str):
            async with sem:
                try:
                    resp = await self._send_wait({
                        "ticks_history": sym,
                        "count": HISTORY_COUNT,
                        "end": "latest",
                        "style": "ticks",
                        "adjust_start_time": 1,
                    }, timeout=30.0)
                    if "error" in resp:
                        log.warning(f"  {sym}: {resp['error'].get('message','?')}")
                        return
                    hist   = resp.get("history", {})
                    prices = hist.get("prices", [])
                    times  = hist.get("times",  [])
                    for p, t in zip(prices, times):
                        self.buffers[sym].append(float(p))
                        self.timestamps[sym].append(float(t))
                        self._tick_times[sym].append(float(t))
                    if prices:
                        self.latest[sym] = float(prices[-1])
                    fn = REVERSE_MAP.get(sym, sym)
                    log.info(f"  OK {fn:<14} {len(prices):>5} ticks")
                except asyncio.TimeoutError:
                    log.warning(f"  {sym}: backfill timeout")
                except Exception as e:
                    log.warning(f"  {sym}: {e}")

        await asyncio.gather(*[_fill(s) for s in ALL_SYMBOLS.values()])
        total = sum(self.n(s) for s in ALL_SYMBOLS.values())
        log.info(f"Backfill complete — {total:,} ticks")

        for sym in ALL_SYMBOLS.values():
            rid = await self._next_id()
            try:
                await self._ws.send(json.dumps({
                    "ticks": sym, "subscribe": 1, "req_id": rid,
                }))
                await asyncio.sleep(0.03)
            except Exception as e:
                log.warning(f"  subscribe {sym}: {e}")

        self._init_done = True
        log.info("Live subscriptions active OK")

    async def run(self):
        self._running = True
        backoff = 2.0
        while self._running:
            try:
                log.info(f"Connecting to {DERIV_WS_URL}")
                async with websockets.connect(
                    DERIV_WS_URL,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                    max_size=2**23,
                    extra_headers={"Origin": "https://smarttrader.deriv.com"},
                ) as ws:
                    self._ws        = ws
                    self._connected = True
                    self._init_done = False
                    backoff         = 2.0
                    log.info("WebSocket connected OK")
                    recv_task = asyncio.ensure_future(self._receive_loop())
                    try:
                        await self._initialise()
                    except Exception as e:
                        log.warning(f"Initialise: {e}")
                    await recv_task
            except asyncio.CancelledError:
                log.info("DataManager cancelled.")
                break
            except Exception as exc:
                self._connected = False
                self._ws        = None
                for fut in list(self._pending.values()):
                    if not fut.done():
                        fut.cancel()
                self._pending.clear()
                log.warning(f"WS: {exc!r} — retry in {backoff:.1f}s")
                try:
                    await asyncio.sleep(backoff)
                except asyncio.CancelledError:
                    break
                backoff = min(backoff * 2, 60.0)

    async def stop(self):
        self._running = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass

# ─────────────────────────────────────────────────────────────────────────────
# VOLATILITY MODEL
# ─────────────────────────────────────────────────────────────────────────────
class VolatilityModel:
    ADVERTISED = ADVERTISED_VOL

    def _ewma_sigma(self, r: np.ndarray, lam: float = EWMA_LAMBDA) -> float:
        if len(r) < 2:
            return 0.01
        var = float(np.var(r[:20], ddof=1)) if len(r) >= 20 else float(np.var(r, ddof=1))
        if var <= 0:
            var = 1e-10
        for ret in r[20:]:
            var = lam*var + (1-lam)*ret*ret
        return max(math.sqrt(var), 1e-10)

    def fit(self, sym: str, lr: np.ndarray) -> GBMParams:
        r = lr[np.isfinite(lr)]
        if len(r) < MIN_TICKS:
            return GBMParams(advertised_vol=self.ADVERTISED.get(sym, 0.0))
        mu       = float(np.mean(r))
        sig      = max(float(np.std(r, ddof=1)), 1e-10)
        sig_ewma = self._ewma_sigma(r)
        try:
            ksp = float(kstest((r-mu)/sig, "norm").pvalue)
        except Exception:
            ksp = 1.0
        n_half = len(r) // 2
        sig1   = float(np.std(r[:n_half], ddof=1)) if n_half > 10 else sig
        sig2   = float(np.std(r[n_half:], ddof=1)) if n_half > 10 else sig
        stab   = 1.0 - min(abs(sig1-sig2)/(sig+1e-12), 1.0)
        fq     = float(np.clip((ksp+stab)/2.0, 0.10, 1.0))
        return GBMParams(
            mu=mu, sigma=sig, sigma_ewma=sig_ewma,
            n_obs=len(r),
            advertised_vol=self.ADVERTISED.get(sym, 0.0),
            ks_pvalue=ksp, fit_quality=fq,
        )

    def ann_vol(self, p: GBMParams, tpy: int = 86400*365) -> float:
        return p.sigma_ewma * math.sqrt(tpy)

    def deviation(self, p: GBMParams, tpy: int = 86400*365) -> float:
        if p.advertised_vol == 0:
            return 0.0
        return (self.ann_vol(p, tpy) - p.advertised_vol) / p.advertised_vol * 100

    def detect_regime(self, r: np.ndarray) -> str:
        if len(r) < 50:
            return "NORMAL"
        recent_std = float(np.std(r[-50:], ddof=1))
        old_std    = float(np.std(r[:-50], ddof=1)) if len(r) > 100 else recent_std
        ratio      = recent_std / max(old_std, 1e-10)
        if ratio > 1.5:
            return "VOLATILE"
        if ratio < 0.6:
            return "RANGING"
        drift = float(np.mean(r[-50:]))
        if abs(drift) > 2.5 * recent_std / math.sqrt(50):
            return "TRENDING"
        return "NORMAL"

# ─────────────────────────────────────────────────────────────────────────────
# VOLATILITY REGIME ENGINE v4.3
# FIX v4.3: window_override param (PROBLEM 3)
# FIX v4.3: empirical cap on targets (PROBLEM 1)
# FIX v4.3: momentum blending preserved
# ─────────────────────────────────────────────────────────────────────────────
class VolatilityRegimeEngine:
    def __init__(self, vm: VolatilityModel):
        self.vm = vm

    def _target_sigma_per_tick(
            self, sym: str,
            tpy: int = 86400*365) -> float:
        adv = ADVERTISED_VOL.get(sym, 0.0)
        if adv == 0:
            return 0.01 / math.sqrt(tpy)
        return adv / math.sqrt(tpy)

    def _normalized_momentum(
            self, r: np.ndarray,
            realized_sigma: float,
            window: int) -> float:
        recent = r[-window:] if len(r) >= window else r
        if len(recent) < 10 or realized_sigma < 1e-10:
            return 0.0
        mu_r = float(np.mean(recent))
        se   = realized_sigma / math.sqrt(len(recent))
        return float(np.clip(mu_r / max(se, 1e-12), -5.0, 5.0))

    def _ewma_momentum(
            self, r: np.ndarray,
            window: int) -> float:
        if len(r) < 20:
            return 0.0
        recent     = r[-window:]
        fast_alpha = 2.0 / (min(20, len(recent)) + 1)
        slow_alpha = 2.0 / (len(recent) + 1)
        fast_ewma  = float(recent[0])
        slow_ewma  = float(recent[0])
        for v in recent[1:]:
            fast_ewma = fast_alpha*v + (1-fast_alpha)*fast_ewma
            slow_ewma = slow_alpha*v + (1-slow_alpha)*slow_ewma
        std = float(np.std(recent, ddof=1))
        if std < 1e-12:
            return 0.0
        return float(np.clip((fast_ewma-slow_ewma)/std, -1.0, 1.0))

    def analyse(
            self,
            sym: str,
            lr: np.ndarray,
            S0: float,
            horizon: int,
            profile: AssetProfile,
            tod_mult: float = 1.0,
            window_override: Optional[int] = None,   # FIX v4.3 PROBLEM 3
            prices: Optional[np.ndarray]   = None,   # FIX v4.3 PROBLEM 1
    ) -> VolRegimeResult:

        r = lr[np.isfinite(lr)]
        if len(r) < MIN_TICKS:
            return self._neutral(sym, S0, horizon, profile, tod_mult)

        gp             = self.vm.fit(sym, r)
        realized_sigma = gp.sigma_ewma
        target_sigma   = self._target_sigma_per_tick(sym)

        if target_sigma < 1e-10:
            target_sigma = realized_sigma

        vol_deviation = (
            (realized_sigma - target_sigma)
            / max(target_sigma, 1e-10))

        ratio = realized_sigma / max(target_sigma, 1e-10)
        if ratio < 0.80:
            regime = "COMPRESSED"
        elif ratio > 1.20:
            regime = "EXPANDED"
        else:
            regime = "NORMAL"

        dev_factor = float(np.clip(
            (ratio - 0.70) / 0.80, 0.0, 1.0))
        k_adaptive = float(np.clip(
            profile.vol_k_min
            + dev_factor * (profile.vol_k_max - profile.vol_k_min),
            profile.vol_k_min,
            profile.vol_k_max))

        # FIX v4.3 PROBLEM 3: Use horizon-scaled window if provided
        # Makes momentum calculation timeframe-aware:
        # short horizon → small window → short-term momentum only
        # long horizon  → large window → macro momentum
        if window_override is not None:
            window = max(
                min(window_override, len(r) // 2),
                30)
        else:
            window = min(
                profile.drift_window,
                len(r) // 2, 300)
        short_window = max(window // 3, 20)

        norm_mom_long  = self._normalized_momentum(r, realized_sigma, window)
        norm_mom_short = self._normalized_momentum(r, realized_sigma, short_window)
        ewma_mom       = self._ewma_momentum(r, window)

        # FIX v4.3: Blend short (65%) and long (35%) momentum
        norm_mom_blended = float(np.clip(
            norm_mom_short * 0.65 + norm_mom_long * 0.35,
            -5.0, 5.0))

        # Use short for signal decision (responsive)
        norm_mom = norm_mom_short

        sigma_h = realized_sigma * math.sqrt(max(horizon, 1))

        stop_factor = {
            "COMPRESSED": 0.55,
            "NORMAL":     0.70,
            "EXPANDED":   0.85,
        }.get(regime, 0.70)

        target_up   = S0 * math.exp( k_adaptive * sigma_h)
        target_down = S0 * math.exp(-k_adaptive * sigma_h)
        stop_up     = S0 * math.exp( stop_factor * sigma_h)
        stop_down   = S0 * math.exp(-stop_factor * sigma_h)

        # FIX v4.3 PROBLEM 1: Empirical cap on targets
        # Prevents 15%+ targets from sigma explosion on
        # high-vol assets. Self-calibrates per asset per regime.
        # Uses actual observed historical moves not model assumptions.
        if prices is not None and len(prices) >= horizon * 2:
            emp_cap = _empirical_move_cap(prices, horizon)
            if emp_cap is not None:
                raw_up_move   = (target_up - S0) / S0
                raw_down_move = (S0 - target_down) / S0

                if raw_up_move > emp_cap:
                    # Clamp target, adjust stop proportionally
                    # to preserve RR ratio
                    rr_ratio_up  = raw_up_move / max(
                        (S0 - stop_down) / S0, 1e-10)
                    target_up    = S0 * (1.0 + emp_cap)
                    clamped_stop = emp_cap / max(rr_ratio_up, 1e-10)
                    stop_down    = S0 * (
                        1.0 - min(clamped_stop, emp_cap * 0.8))
                    log.debug(
                        f"EmpCap UP [{sym}]: "
                        f"{raw_up_move*100:.2f}% → "
                        f"{emp_cap*100:.2f}% (h={horizon})")

                if raw_down_move > emp_cap:
                    rr_ratio_down = raw_down_move / max(
                        (stop_up - S0) / S0, 1e-10)
                    target_down   = S0 * (1.0 - emp_cap)
                    clamped_stop  = emp_cap / max(rr_ratio_down, 1e-10)
                    stop_up       = S0 * (
                        1.0 + min(clamped_stop, emp_cap * 0.8))
                    log.debug(
                        f"EmpCap DN [{sym}]: "
                        f"{raw_down_move*100:.2f}% → "
                        f"{emp_cap*100:.2f}% (h={horizon})")

        # FIX v4.3: Relaxed thresholds (preserved from v4.1)
        t_thresh_eff  = max(profile.drift_tstat_min * 0.60, 1.2)
        pvalue        = float(2 * (1 - norm.cdf(abs(norm_mom))))
        p_thresh      = 0.25

        is_bullish = (norm_mom > t_thresh_eff and ewma_mom > 0.02)
        is_bearish = (norm_mom < -t_thresh_eff and ewma_mom < -0.02)

        long_agrees = (
            (norm_mom_long > 0 and norm_mom_short > 0)
            or (norm_mom_long < 0 and norm_mom_short < 0))

        p_significant = pvalue < p_thresh

        mom_score  = float(np.clip(
            abs(norm_mom) / max(t_thresh_eff * 1.5, 1),
            0.0, 1.0))
        ewma_score = float(np.clip(abs(ewma_mom), 0.0, 1.0))
        regime_bonus = {
            "EXPANDED":   1.35,
            "NORMAL":     1.00,
            "COMPRESSED": 0.80,
        }.get(regime, 1.0)
        long_bonus = 1.15 if long_agrees else 0.90

        raw_edge   = (mom_score*0.65 + ewma_score*0.35) * regime_bonus * long_bonus
        raw_edge  *= tod_mult
        edge_score = float(np.clip(raw_edge, 0.0, 1.0))

        edge_ok       = edge_score >= profile.signal_edge_min
        strong_enough = edge_score >= 0.18

        if (edge_ok
                and (p_significant or strong_enough)
                and (is_bullish or is_bearish)):
            direction = "bullish" if is_bullish else "bearish"
        else:
            direction = "neutral"

        if (edge_score >= profile.strong_edge
                and abs(norm_mom) >= t_thresh_eff * 2.0
                and pvalue < 0.10):
            strength = "STRONG"
        elif (edge_score >= profile.moderate_edge
              and abs(norm_mom) >= t_thresh_eff
              and pvalue < 0.25):
            strength = "MODERATE"
        else:
            strength = "WEAK"

        return VolRegimeResult(
            regime=regime,
            realized_sigma=realized_sigma,
            target_sigma=target_sigma,
            vol_deviation=vol_deviation,
            normalized_momentum=norm_mom_blended,  # blended for display
            momentum_direction=direction,
            k_adaptive=k_adaptive,
            target_up=target_up,
            target_down=target_down,
            stop_up=stop_up,
            stop_down=stop_down,
            edge_score=edge_score,
            signal=direction,
            signal_strength=strength,
            tod_multiplier=tod_mult,
        )

    def _neutral(
            self, sym: str, S0: float,
            horizon: int, profile: AssetProfile,
            tod_mult: float) -> VolRegimeResult:
        ts = self._target_sigma_per_tick(sym)
        sh = ts * math.sqrt(max(horizon, 1))
        k  = profile.vol_k_min
        return VolRegimeResult(
            regime="NORMAL",
            realized_sigma=ts,
            target_sigma=ts,
            vol_deviation=0.0,
            normalized_momentum=0.0,
            momentum_direction="neutral",
            k_adaptive=k,
            target_up=S0 * math.exp(k * sh),
            target_down=S0 * math.exp(-k * sh),
            stop_up=S0 * math.exp(0.70 * sh),
            stop_down=S0 * math.exp(-0.70 * sh),
            edge_score=0.0,
            signal="neutral",
            signal_strength="WEAK",
            tod_multiplier=tod_mult,
        )

# ─────────────────────────────────────────────────────────────────────────────
# DRIFT ENGINE v4.3
# FIX v4.3 PROBLEM 3: window_override parameter
# ─────────────────────────────────────────────────────────────────────────────
class DriftEngine:
    def _ewma_momentum(self, r: np.ndarray, window: int) -> float:
        if len(r) < 20:
            return 0.0
        recent     = r[-window:]
        fast_alpha = 2.0 / (min(20, len(recent)) + 1)
        slow_alpha = 2.0 / (len(recent) + 1)
        fast_ewma  = float(recent[0])
        slow_ewma  = float(recent[0])
        for v in recent[1:]:
            fast_ewma = fast_alpha*v + (1-fast_alpha)*fast_ewma
            slow_ewma = slow_alpha*v + (1-slow_alpha)*slow_ewma
        std = float(np.std(recent, ddof=1))
        if std < 1e-12:
            return 0.0
        return float(np.clip((fast_ewma-slow_ewma)/std, -1.0, 1.0))

    def _rolling_tstat(
            self, r: np.ndarray,
            window: int) -> Tuple[float, float]:
        recent = r[-window:] if len(r) >= window else r
        if len(recent) < 20:
            return 0.0, 1.0
        mu  = float(np.mean(recent))
        std = float(np.std(recent, ddof=1))
        if std < 1e-12:
            return 0.0, 1.0
        tstat  = mu / (std / math.sqrt(len(recent)))
        pvalue = float(2 * (1 - norm.cdf(abs(tstat))))
        return tstat, pvalue

    def _multi_window_agree(self, r: np.ndarray, window: int) -> bool:
        if len(r) < window * 2:
            return False
        short_w      = max(window // 3, 20)
        ts_short, _  = self._rolling_tstat(r, short_w)
        ts_long, _   = self._rolling_tstat(r, window)
        return (ts_short * ts_long) > 0

    def analyse(
            self,
            sym: str,
            lr: np.ndarray,
            S0: float,
            profile: AssetProfile,
            tod_mult: float = 1.0,
            window_override: Optional[int] = None,  # FIX v4.3 PROBLEM 3
    ) -> DriftSignal:
        r = lr[np.isfinite(lr)]

        # FIX v4.3 PROBLEM 3: Use horizon-scaled window if provided
        # Makes drift signals timeframe-aware for JUMP and STEP assets
        if window_override is not None:
            window = max(min(window_override, len(r) // 2), 30)
        else:
            window = profile.drift_window

        if len(r) < 30:
            return DriftSignal(
                direction="neutral",
                tstat=0.0, pvalue=1.0,
                drift_per_tick=0.0, drift_pct=0.0,
                regime="NORMAL", ewma_momentum=0.0,
                window_used=window,
                edge_score=0.0, signal_strength="WEAK")

        tstat, pvalue = self._rolling_tstat(r, window)
        momentum      = self._ewma_momentum(r, window)
        vm            = VolatilityModel()
        regime        = vm.detect_regime(r)
        agrees        = self._multi_window_agree(r, window)

        recent         = r[-window:] if len(r) >= window else r
        drift_per_tick = float(np.mean(recent))
        drift_pct      = abs(drift_per_tick) / max(S0, 1e-10) * 100

        tstat_score    = float(np.clip(
            abs(tstat) / max(profile.drift_tstat_min*2, 1),
            0.0, 1.0))
        momentum_score = float(np.clip(abs(momentum), 0.0, 1.0))
        regime_bonus   = {
            "TRENDING":  1.40, "VOLATILE":  1.15,
            "NORMAL":    1.00, "RANGING":   0.65,
        }.get(regime, 1.0)
        agree_bonus = 1.20 if agrees else 0.80

        raw_edge   = (tstat_score*0.55 + momentum_score*0.45) * regime_bonus * agree_bonus
        raw_edge  *= tod_mult
        edge_score = float(np.clip(raw_edge, 0.0, 1.0))

        t_threshold   = profile.drift_tstat_min
        is_bullish    = (tstat > 0 and momentum > 0.05)
        is_bearish    = (tstat < 0 and momentum < -0.05)
        t_significant = abs(tstat) >= t_threshold
        p_significant = pvalue < 0.15

        if (t_significant and p_significant
                and edge_score >= profile.signal_edge_min
                and (is_bullish or is_bearish)):
            direction = "bullish" if is_bullish else "bearish"
        else:
            direction = "neutral"

        if (edge_score >= profile.strong_edge
                and abs(tstat) >= t_threshold*1.5
                and agrees and pvalue < 0.05):
            strength = "STRONG"
        elif (edge_score >= profile.moderate_edge
              and abs(tstat) >= t_threshold
              and pvalue < 0.15):
            strength = "MODERATE"
        else:
            strength = "WEAK"

        return DriftSignal(
            direction=direction,
            tstat=tstat, pvalue=pvalue,
            drift_per_tick=drift_per_tick,
            drift_pct=drift_pct,
            regime=regime,
            ewma_momentum=momentum,
            window_used=window,
            edge_score=edge_score,
            signal_strength=strength,
        )

# ─────────────────────────────────────────────────────────────────────────────
# JUMP-DIFFUSION MODEL v4.3 (Bayesian + counter-drift)
# ─────────────────────────────────────────────────────────────────────────────
class JumpDiffusionModel:
    def detect(self, lr: np.ndarray, k: float = JUMP_THRESHOLD):
        med = np.median(lr)
        mad = np.median(np.abs(lr - med)) * 1.4826
        if mad < 1e-12:
            mad = float(np.std(lr, ddof=1)) * 0.6745
        if mad < 1e-12:
            mad = 1e-10
        is_j = np.abs(lr - med) > k * mad
        return is_j, lr[is_j]

    def _recent_cluster(self, lr: np.ndarray, window: int = 200) -> float:
        recent = lr[-window:] if len(lr) >= window else lr
        if len(recent) == 0:
            return 0.0
        is_j, _ = self.detect(recent)
        return float(np.sum(is_j)) / len(recent)

    def _inter_arrival_cv(self, is_jump: np.ndarray) -> float:
        indices = np.where(is_jump)[0]
        if len(indices) < 3:
            return 1.0
        ia = np.diff(indices).astype(float)
        if np.mean(ia) < 1e-10:
            return 1.0
        return float(np.std(ia) / np.mean(ia))

    def _hazard_intensity_bayesian(
            self, lr: np.ndarray,
            lam_posterior: float,
            expected_lam: float,
            tod_mult: float = 1.0) -> float:
        if len(lr) < 20 or lam_posterior <= 0:
            return 0.0
        is_j, _ = self.detect(lr)
        jump_idx    = np.where(is_j)[0]
        ticks_since = (len(lr) if len(jump_idx) == 0
                       else len(lr) - int(jump_idx[-1]))
        raw_hazard  = 1.0 - math.exp(-lam_posterior * ticks_since)
        if expected_lam > 0 and lam_posterior < expected_lam * 0.7:
            raw_hazard = min(raw_hazard * 1.5, 1.0)
        return float(np.clip(raw_hazard * tod_mult, 0.0, 1.0))

    def _bayesian_lambda(
            self, n_jumps: int, n_obs: int,
            expected_lam: float,
            prior_strength: float = 10.0) -> float:
        if expected_lam <= 0:
            return max(n_jumps / max(n_obs, 1), 1e-6)
        alpha = prior_strength * expected_lam
        beta  = prior_strength
        return (alpha + n_jumps) / (beta + n_obs)

    def _spike_magnitude_percentiles(
            self, jumps: np.ndarray) -> Tuple[float, float]:
        abj = np.abs(jumps)
        if len(abj) < 3:
            return 0.02, 0.03
        p60 = float(np.percentile(abj, 60))
        p75 = float(np.percentile(abj, 75))
        return p60, p75

    def _counter_drift(
            self, lr: np.ndarray,
            is_jump: np.ndarray) -> float:
        diff_r = lr[~is_jump]
        if len(diff_r) < 20:
            return 0.0
        try:
            x = np.arange(len(diff_r), dtype=float)
            slope, _, _, _, _ = linregress(x, diff_r)
            return float(slope)
        except Exception:
            return 0.0

    def fit_unbiased(self, sym: str, lr: np.ndarray) -> JumpParams:
        r = lr[np.isfinite(lr)]
        if len(r) < MIN_TICKS:
            return JumpParams(jump_sign=1.0)
        is_j, jumps = self.detect(r)
        cat = _cat(sym)
        if cat == "boom":
            forced_sign = 1.0
        elif cat == "crash":
            forced_sign = -1.0
        else:
            forced_sign = None
        if len(jumps) == 0:
            sign = forced_sign if forced_sign is not None else 1.0
            return self._fit_core(sym, r, is_j, jumps, sign)
        mean_jump = float(np.mean(jumps))
        data_sign = 1.0 if mean_jump >= 0 else -1.0
        sign      = forced_sign if forced_sign is not None else data_sign
        return self._fit_core(sym, r, is_j, jumps, sign)

    def _fit_core(self, sym, r, is_j, jumps, sign) -> JumpParams:
        diff = r[~is_j]
        mu   = float(np.mean(diff)) if len(diff) > 1 else 0.0
        sig  = max(float(np.std(diff, ddof=1)), 1e-10) if len(diff) > 2 else 0.01

        n_jumps  = int(np.sum(is_j))
        n_obs    = len(r)
        lam_mle  = n_jumps / max(n_obs, 1)

        exp_f    = EXPECTED_JUMP_FREQ.get(sym, 0)
        t_lam    = (1.0/exp_f) if exp_f > 0 else lam_mle
        lam_post = self._bayesian_lambda(n_jumps, n_obs, t_lam)

        lam_dev  = (
            (lam_mle - t_lam) / (t_lam + 1e-12) * 100
            if t_lam > 0 else 0.0)
        rc  = self._recent_cluster(r)
        cv  = self._inter_arrival_cv(is_j)
        hz  = self._hazard_intensity_bayesian(r, lam_post, t_lam)
        cd  = self._counter_drift(r, is_j)

        abj      = np.abs(jumps)
        jm       = float(np.mean(abj)) if len(abj) > 0 else 0.02
        js       = float(np.std(abj, ddof=1)) if len(abj) > 1 else 0.01
        p60, p75 = self._spike_magnitude_percentiles(jumps)

        return JumpParams(
            mu=mu, sigma=sig,
            lam=lam_mle,
            lam_posterior=lam_post,
            jump_mean=jm, jump_std=js,
            jump_sign=sign,
            n_jumps=n_jumps, n_obs=n_obs,
            expected_lam=t_lam,
            lam_deviation=lam_dev,
            recent_cluster=rc,
            inter_arrival_cv=cv,
            hazard_intensity=hz,
            counter_drift=cd,
            spike_mag_p60=p60,
            spike_mag_p75=p75,
        )

    def prob_in_n(self, p: JumpParams, n: int) -> float:
        lam = max(p.lam_posterior, p.lam, 1e-10)
        if lam <= 0:
            return 0.0
        return 1.0 - math.exp(-lam * n)

    def ticks_to_next_range(
            self, p: JumpParams,
            tick_rate: float) -> Tuple[int, int, str, str, str]:
        lam = max(p.lam_posterior, p.lam, 1e-10)
        if lam <= 0:
            return 300, 600, "~5 min", "~10 min", "~7 min"
        mean_inter   = 1.0 / lam
        hz           = p.hazard_intensity
        accel        = 1.0 - (hz * 0.60)
        center_ticks = max(int(mean_inter * accel), 5)
        lo_ticks     = max(int(center_ticks * 0.40), 1)
        hi_ticks     = int(center_ticks * 1.60)

        def _fmt(ticks):
            secs = ticks / max(tick_rate, 0.1)
            if secs < 60:
                return f"~{secs:.0f}s"
            elif secs < 3600:
                return f"~{secs/60:.1f}min"
            else:
                return f"~{secs/3600:.1f}hr"

        return lo_ticks, hi_ticks, _fmt(lo_ticks), _fmt(hi_ticks), _fmt(center_ticks)

    def jump_magnitude_range(self, p: JumpParams) -> Tuple[float, float]:
        lo = max(p.jump_mean - 2*p.jump_std, 0) * 100
        hi = (p.jump_mean + 2*p.jump_std) * 100
        return lo, hi

# ─────────────────────────────────────────────────────────────────────────────
# SPIKE ENGINE v4.3 (Bayesian + ToD — unchanged)
# ─────────────────────────────────────────────────────────────────────────────
class SpikeEngine:
    def __init__(self, jm: JumpDiffusionModel):
        self.jm = jm

    def assess(
            self, sym: str,
            jp: JumpParams,
            horizon: int,
            tick_rate: float,
            tod_mult: float = 1.0,
    ) -> Optional[SpikeAlert]:
        profile = effective_profile(sym)
        if not profile.spike_enabled:
            return None
        cat   = _cat(sym)
        fname = _friendly(sym)
        if cat == "boom":
            direction = "up"
        elif cat == "crash":
            direction = "down"
        else:
            direction = "up" if jp.jump_sign > 0 else "down"

        poisson_prob     = self.jm.prob_in_n(jp, horizon)
        poisson_prob_eff = float(np.clip(poisson_prob * tod_mult, 0.0, 1.0))
        jlo, jhi         = self.jm.jump_magnitude_range(jp)
        mag_target       = jp.spike_mag_p60 * 100

        tlo, thi, slo, shi, scenter = self.jm.ticks_to_next_range(jp, tick_rate)

        c1 = poisson_prob_eff >= profile.spike_min_prob
        c2 = mag_target        >= profile.spike_min_magnitude
        c3 = (
            abs(jp.lam_deviation)  >= profile.spike_lam_dev_min
            or jp.recent_cluster    > jp.expected_lam * 1.2
            or jp.inter_arrival_cv  > 1.15
            or jp.hazard_intensity  >= profile.spike_hazard_high)
        is_imminent = c1 and c2 and c3

        p1 = min(poisson_prob_eff / max(profile.spike_min_prob, 0.01), 1.0) * 0.30
        p2 = min(mag_target / max(profile.spike_min_magnitude, 0.001), 1.0) * 0.25
        p3 = min(jp.hazard_intensity, 1.0) * 0.25
        p4 = min(abs(jp.lam_deviation) / 30.0, 1.0) * 0.10
        p5 = min(tod_mult - 1.0, 0.12) / 0.12 * 0.10
        confidence = float(np.clip(p1+p2+p3+p4+p5, 0.0, 1.0))

        reasons = []
        reasons.append(
            f"Poisson P={poisson_prob:.1%} (posterior λ), "
            f"ToD adj={poisson_prob_eff:.1%} in {horizon} ticks"
            + (" ✓" if c1 else f" (need>={profile.spike_min_prob:.0%})"))
        reasons.append(
            f"Magnitude P60={jp.spike_mag_p60*100:.3f}%/spike"
            + (" ✓" if c2 else f" (need>={profile.spike_min_magnitude:.2f}%)"))
        if abs(jp.lam_deviation) >= profile.spike_lam_dev_min:
            d = "SUPPRESSED" if jp.lam_deviation < 0 else "ELEVATED"
            reasons.append(
                f"Rate {d} {abs(jp.lam_deviation):.0f}% vs theory"
                + (" — spike overdue!" if jp.lam_deviation < 0
                   else " — frequent regime"))
        if jp.hazard_intensity >= profile.spike_hazard_high:
            reasons.append(
                f"Hazard={jp.hazard_intensity:.0%} HIGH (Bayesian nonlinear)")
        if jp.inter_arrival_cv > 1.15:
            reasons.append(f"Clustering CV={jp.inter_arrival_cv:.2f}")
        if jp.counter_drift != 0.0:
            cd_dir = "UP" if jp.counter_drift > 0 else "DOWN"
            reasons.append(
                f"Counter-drift: {cd_dir} slope={jp.counter_drift:.8f}")
        if tod_mult > 1.02:
            reasons.append(f"ToD boost: x{tod_mult:.3f}")
        if not c3:
            reasons.append("No regime anomaly detected yet")
        if not is_imminent:
            return None

        return SpikeAlert(
            symbol=sym, fname=fname,
            direction=direction,
            confidence=confidence,
            poisson_prob=poisson_prob_eff,
            hazard_intensity=jp.hazard_intensity,
            ticks_lo=tlo, ticks_hi=thi,
            time_lo_str=slo, time_hi_str=shi,
            time_center_str=scenter,
            magnitude_lo=jlo, magnitude_hi=jhi,
            magnitude_p60=jp.spike_mag_p60*100,
            magnitude_p75=jp.spike_mag_p75*100,
            reasons=reasons,
            is_imminent=True,
            counter_drift=jp.counter_drift,
            tod_multiplier=tod_mult,
        )

    def format_standalone(self, sa: SpikeAlert) -> str:
        dir_icon  = "📈" if sa.direction == "up" else "📉"
        conf_bar  = "█"*int(sa.confidence*10) + "░"*(10-int(sa.confidence*10))
        cat       = _cat(sa.symbol)
        if cat == "boom":
            dir_word = "BOOM SPIKE UP"
        elif cat == "crash":
            dir_word = "CRASH SPIKE DOWN"
        elif sa.direction == "up":
            dir_word = "JUMP SPIKE UP"
        else:
            dir_word = "JUMP SPIKE DOWN"

        lines = [
            f"⚡ *SPIKE INCOMING — {sa.fname}* {dir_icon}\n",
            f"*{dir_word}* [v4.3 Bayesian]\n",
            f"Confidence: [{conf_bar}] {sa.confidence:.0%}",
            f"Window: *{sa.time_lo_str} — {sa.time_hi_str}*",
            f"Best estimate: *{sa.time_center_str}*",
            f"Approx ticks: *{sa.ticks_lo} — {sa.ticks_hi} ticks*\n",
            f"Target size (P60): *{sa.magnitude_p60:.3f}%*",
            f"Target size (P75): *{sa.magnitude_p75:.3f}%*",
            f"Range: *{sa.magnitude_lo:.2f}% — {sa.magnitude_hi:.2f}%*",
            f"P(spike/ToD): *{sa.poisson_prob:.1%}*",
            f"Hazard (Bayesian): *{sa.hazard_intensity:.0%}*",
            f"ToD multiplier: *x{sa.tod_multiplier:.3f}*\n",
        ]
        if sa.counter_drift != 0.0:
            cd_dir = "📈 UP" if sa.counter_drift > 0 else "📉 DOWN"
            lines.append(
                f"Counter-drift: *{cd_dir}* (slope={sa.counter_drift:.8f})\n")
        lines.append("*Why this alert:*")
        for r in sa.reasons:
            lines.append(f"  - {r}")
        lines += [
            f"\n*What to do:*",
            f"  {dir_word} expected in window.",
            f"  Best size: {sa.magnitude_p60:.3f}%–{sa.magnitude_p75:.3f}%",
            f"  Position {dir_icon} BEFORE spike.",
            f"  Tight stop below/above price.",
            f"  Max 1-2% account risk.",
            f"\n⚠️ RNG-driven — exact tick unpredictable.",
        ]
        return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# TRADE SETUP BUILDER v4.3
# FIX v4.3 PROBLEM 2: Dynamic RR from MC distribution
# FIX v4.3 PROBLEM 4: Z-score neutral gate
# ─────────────────────────────────────────────────────────────────────────────
class TradeSetupBuilder:

    @staticmethod
    def _signal_strength(
            edge: float, prob: float,
            rr: float, profile: AssetProfile) -> str:
        if (edge >= profile.strong_edge
                and prob >= profile.strong_prob
                and rr  >= profile.strong_rr):
            return "STRONG"
        if (edge >= profile.moderate_edge
                and prob >= profile.moderate_prob
                and rr  >= profile.moderate_rr):
            return "MODERATE"
        return "WEAK"

    @staticmethod
    def build(
            mc: 'MCResult',
            timeframe_label: str,
            horizon_ticks: int,
            regime: str = "NORMAL",
            profile: Optional[AssetProfile] = None,
            drift: Optional[DriftSignal]    = None,
            vol_regime: Optional[VolRegimeResult] = None,
            tod_mult: float = 1.0,
    ) -> Optional[TradeSetup]:

        if profile is None:
            profile = effective_profile(mc.symbol)

        S0 = mc.S0
        if S0 <= 0:
            return None

        finals        = mc.paths[:, -1]
        signal_source = "MC"

        # ── Direction priority ──────────────────────────────────────────────
        if profile.direction_bias == "BUY":
            is_bull       = True
            signal_source = "PROFILE"
        elif profile.direction_bias == "SELL":
            is_bull       = False
            signal_source = "PROFILE"
        elif (vol_regime is not None
              and profile.vol_regime_enabled
              and vol_regime.signal != "neutral"):
            is_bull       = (vol_regime.signal == "bullish")
            signal_source = "VOL_REGIME"
        # FIX v4.3 PROBLEM 4: When vol regime is explicitly neutral
        # return None immediately — prevents raw MC fallback from
        # bypassing the z-score gate entirely
        elif (vol_regime is not None
              and profile.vol_regime_enabled
              and vol_regime.signal == "neutral"):
            return None
        elif (drift is not None
              and profile.drift_signal_enabled
              and drift.direction != "neutral"):
            is_bull       = (drift.direction == "bullish")
            signal_source = "DRIFT"
        else:
            is_bull       = (mc.prob_hit_up > mc.prob_hit_down)
            signal_source = "MC"

        direction = "BUY" if is_bull else "SELL"

        # ── Target and stop — regime engine ────────────────────────────────
        if (vol_regime is not None
                and profile.vol_regime_enabled
                and vol_regime.signal != "neutral"):
            if is_bull:
                target = vol_regime.target_up
                inv    = vol_regime.stop_down
            else:
                target = vol_regime.target_down
                inv    = vol_regime.stop_up
        else:
            if is_bull:
                winners = finals[finals > S0]
                losers  = finals[finals <= S0]
                if len(winners) < 10 or len(losers) < 10:
                    return None
                target = float(np.percentile(winners, profile.target_percentile))
                inv    = float(np.percentile(losers,  profile.stop_percentile))
            else:
                winners = finals[finals < S0]
                losers  = finals[finals >= S0]
                if len(winners) < 10 or len(losers) < 10:
                    return None
                target = float(np.percentile(winners, 100 - profile.target_percentile))
                inv    = float(np.percentile(losers,  100 - profile.stop_percentile))

        # FIX v4.3 PROBLEM 2: Dynamic RR from MC distribution structure
        # Compute market-structure target and stop from actual MC percentiles
        # Use whichever (regime or MC-structure) gives better RR
        # This produces genuinely variable RR: 1:1.6, 1:2.1, 1:2.8 etc
        try:
            if is_bull:
                mc_struct_target = float(np.percentile(finals, 75))
                mc_struct_stop   = float(np.percentile(finals, 15))
            else:
                mc_struct_target = float(np.percentile(finals, 25))
                mc_struct_stop   = float(np.percentile(finals, 85))

            mc_target_pct = abs(mc_struct_target - S0) / S0 * 100
            mc_stop_pct   = abs(mc_struct_stop   - S0) / S0 * 100

            if (mc_stop_pct > 1e-6
                    and mc_target_pct >= MIN_TARGET_MOVE * 100):
                mc_rr = mc_target_pct / max(mc_stop_pct, 1e-10)
                regime_target_pct = abs(target - S0) / S0 * 100
                regime_stop_pct   = abs(inv    - S0) / S0 * 100
                regime_rr         = regime_target_pct / max(regime_stop_pct, 1e-10)

                # FIX v4.3: Pick setup with better RR — dynamic not fixed
                if mc_rr > regime_rr:
                    target = mc_struct_target
                    inv    = mc_struct_stop
                    log.debug(
                        f"DynRR [{mc.symbol}]: "
                        f"MC {mc_rr:.2f} > Regime {regime_rr:.2f} "
                        f"→ using MC structure")
        except Exception as _dyn_rr_err:
            log.debug(f"DynRR fallback: {_dyn_rr_err}")

        target_pct = abs(target - S0) / S0 * 100
        stop_pct   = abs(inv    - S0) / S0 * 100

        # Enforce minimum stop width — stop >= 30% of target move
        min_stop_pct = max(target_pct * 0.30, profile.min_stop_pct * 100)
        if stop_pct < min_stop_pct:
            stop_pct = min_stop_pct
            if is_bull:
                inv = S0 * (1 - stop_pct / 100)
            else:
                inv = S0 * (1 + stop_pct / 100)

        if target_pct < max(profile.min_target_pct * 100, MIN_TARGET_MOVE * 100):
            return None
        if stop_pct < 1e-6:
            return None

        rr_ratio = target_pct / max(stop_pct, 1e-10)

        # FIX v4.3: Hard cap RR at 4.0
        # Floor is dynamic — driven by actual MC distribution above
        rr_ratio = min(rr_ratio, 4.0)

        min_rr_eff = max(profile.min_rr, MIN_RR_GLOBAL)
        if rr_ratio < min_rr_eff:
            return None

        # ── Probability from MC ─────────────────────────────────────────────
        prob_mc = mc.prob_hit_up  if is_bull else mc.prob_hit_down
        prob_s  = mc.prob_hit_down if is_bull else mc.prob_hit_up

        # ── Combined edge + calibrated probability ──────────────────────────
        if (vol_regime is not None
                and profile.vol_regime_enabled
                and vol_regime.signal != "neutral"):
            w_v           = profile.drift_edge_weight
            w_m           = 1.0 - w_v
            combined_edge = float(np.clip(
                vol_regime.edge_score * w_v + mc.edge_score * w_m,
                0.0, 1.0))
            prob_boost = vol_regime.edge_score * 0.12
            prob_t     = float(np.clip(
                max(prob_mc + prob_boost, 0.54), 0.50, 0.72))
            if signal_source == "MC":
                signal_source = "VOL_REGIME"

        elif (drift is not None
              and profile.drift_signal_enabled
              and drift.direction != "neutral"):
            w_d           = profile.drift_edge_weight
            w_m           = 1.0 - w_d
            combined_edge = float(np.clip(
                drift.edge_score * w_d + mc.edge_score * w_m,
                0.0, 1.0))
            prob_boost = drift.edge_score * 0.08
            prob_t     = float(np.clip(
                max(prob_mc + prob_boost, 0.53), 0.50, 0.70))
            signal_source = "COMBINED" if signal_source == "MC" else signal_source
        else:
            combined_edge = mc.edge_score
            prob_t        = float(np.clip(prob_mc, 0.50, 0.68))

        # ToD as small bounded adjustment — max +/-3%
        tod_adj = float(np.clip((tod_mult - 1.0) * 0.05, -0.03, 0.03))
        prob_t  = float(np.clip(prob_t + tod_adj, MIN_PROB_TARGET, 0.75))

        if prob_t < MIN_PROB_TARGET:
            return None

        ev = prob_t * target_pct - (1 - prob_t) * stop_pct
        if ev <= 0:
            return None

        ev_min = {
            "VOLATILITY": 0.15,
            "BOOM":       0.30,
            "CRASH":      0.30,
            "JUMP":       0.20,
            "STEP":       0.05,
        }.get(profile.name, 0.15)
        if ev < ev_min:
            return None

        edge_pct = (prob_t - 0.5) * 200
        if edge_pct <= 0:
            return None

        strength = TradeSetupBuilder._signal_strength(
            combined_edge, prob_t, rr_ratio, profile)

        return TradeSetup(
            direction=direction,
            entry=S0,
            target=target,
            invalidation=inv,
            target_pct=target_pct,
            stop_pct=stop_pct,
            rr_ratio=rr_ratio,
            prob_target=prob_t,
            prob_stop=prob_s,
            edge_pct=edge_pct,
            expected_value=ev,
            signal_strength=strength,
            timeframe_label=timeframe_label,
            horizon_ticks=horizon_ticks,
            regime=regime,
            profile_name=profile.name,
            signal_source=signal_source,
            tod_multiplier=tod_mult,
        )

    @staticmethod
    def format_setup(
            ts: TradeSetup,
            fname: str,
            drift: Optional[DriftSignal]      = None,
            vol_regime: Optional[VolRegimeResult] = None,
    ) -> str:
        dir_icon = "📈" if ts.direction == "BUY" else "📉"
        s_icon   = {
            "STRONG": "🟢", "MODERATE": "🟡", "WEAK": "🔴",
        }.get(ts.signal_strength, "⬜")
        rr_str = f"1 : {ts.rr_ratio:.2f}"
        sg     = '+' if ts.direction == 'BUY' else '-'
        sgs    = '-' if ts.direction == 'BUY' else '+'

        lines = [
            f"*Trade Setup — {fname}* {dir_icon} *{ts.direction}*\n",
            f"{s_icon} Strength: *{ts.signal_strength}* | "
            f"Profile: *{ts.profile_name}* | "
            f"Source: *{ts.signal_source}* | "
            f"Regime: *{ts.regime}*",
            f"Timeframe: *{ts.timeframe_label}* "
            f"({ts.horizon_ticks} ticks) | "
            f"ToD: x{ts.tod_multiplier:.3f}\n",
            f"*Entry:*          `{ts.entry:.6f}`",
            f"*Target:*         `{ts.target:.6f}` ({sg}{ts.target_pct:.3f}%)",
            f"*Invalidation:*   `{ts.invalidation:.6f}` ({sgs}{ts.stop_pct:.3f}%)",
            f"\n*Risk/Reward:* `{rr_str}`",
            f"*P(hit target):* `{ts.prob_target:.2%}`",
            f"*P(hit stop):*   `{ts.prob_stop:.2%}`",
            f"*Edge:* `{ts.edge_pct:+.2f}%`",
            f"*EV/trade:* `{ts.expected_value:+.4f}%`\n",
        ]

        if vol_regime and vol_regime.signal != "neutral":
            lines += [
                f"*Vol Regime Engine:*",
                f"  Regime:      `{vol_regime.regime}`",
                f"  Mom z-score: `{vol_regime.normalized_momentum:+.3f}`",
                f"  k_adaptive:  `{vol_regime.k_adaptive:.3f}`",
                f"  Vol dev:     `{vol_regime.vol_deviation:+.3f}`",
                f"  V-Edge:      `{vol_regime.edge_score:.3f}`\n",
            ]
        elif drift and drift.direction != "neutral":
            lines += [
                f"*Drift Confirmation:*",
                f"  Direction: `{drift.direction}`",
                f"  t-stat: `{drift.tstat:+.3f}` | p: `{drift.pvalue:.4f}`",
                f"  Momentum: `{drift.ewma_momentum:+.3f}`",
                f"  D-Edge: `{drift.edge_score:.3f}`\n",
            ]

        if ts.signal_strength == "STRONG":
            lines.append(
                f"*Interpretation:* Strong {ts.signal_source} signal. "
                f"Clear {ts.direction} bias. R/R {rr_str}. Use 1-2% risk.")
        elif ts.signal_strength == "MODERATE":
            lines.append(
                f"*Interpretation:* Moderate edge. Consider 0.5-1% risk.")
        else:
            lines.append(
                f"*Interpretation:* Weak setup — reduce size or wait.")
        return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# CONFLICT RESOLVER (preserved)
# ─────────────────────────────────────────────────────────────────────────────
class ConflictResolver:
    @staticmethod
    def resolve(
            sym: str,
            mc: 'MCResult',
            spike: Optional[SpikeAlert],
            trade: Optional[TradeSetup],
    ) -> Tuple[Optional[TradeSetup], bool, str]:
        profile = effective_profile(sym)
        if spike is None:
            return trade, True, "no_spike"
        if trade is None:
            return None, False, "no_trade"
        spike_is_bull = spike.direction == "up"
        trade_is_bull = trade.direction == "BUY"
        if spike_is_bull == trade_is_bull:
            return trade, True, "aligned"
        else:
            if profile.spike_is_primary:
                log.info(
                    f"ConflictResolver [{profile.name}]: "
                    f"spike={spike.direction} vs trade={trade.direction} "
                    f"— trade SUPPRESSED")
                return None, False, "spike_primary_conflict"
            else:
                return trade, True, "both_independent"

# ─────────────────────────────────────────────────────────────────────────────
# MC ENGINE v4.3
# FIX v4.3 PROBLEM 3: horizon_scaled_window passed to vre/drift
# FIX v4.3 PROBLEM 1: prices passed to vre.analyse
# FIX v4.3 PROBLEM 5: ensemble agreement gate inside run()
# FIX v4.3: HybridTargetRefiner wired in
# ─────────────────────────────────────────────────────────────────────────────
class MCEngine:
    def __init__(self, tod: TimeOfDayProfile):
        self.vm   = VolatilityModel()
        self.jm   = JumpDiffusionModel()
        self.se   = SpikeEngine(self.jm)
        self.tsb  = TradeSetupBuilder()
        self.cr   = ConflictResolver()
        self.de   = DriftEngine()
        self.vre  = VolatilityRegimeEngine(self.vm)
        self.tod  = tod
        self._rng = np.random.default_rng(42)
        # FIX v4.3: Hybrid target refiner
        self.htr  = HybridTargetRefiner()

    def _is_jump_sym(self, sym: str) -> bool:
        f = REVERSE_MAP.get(sym, "")
        return any(x in f for x in ("Boom", "Crash", "Jump"))

    def _adaptive_targets(
            self, S0: float,
            sigma_tick: float,
            horizon: int,
            profile: AssetProfile):
        base = profile.sigma_multiplier_base
        if horizon <= 50:
            mult = base
        elif horizon <= 100:
            mult = base * 1.15
        elif horizon <= 300:
            mult = base * 1.35
        elif horizon <= 600:
            mult = base * 1.55
        else:
            mult = base * 1.75
        sigma_h = sigma_tick * math.sqrt(max(horizon, 1))
        tup     = S0 * math.exp( mult * sigma_h)
        tdn     = S0 * math.exp(-mult * sigma_h)
        pct     = (tup - S0) / S0 * 100
        lbl     = f"+-{mult:.1f}s ({pct:.3f}%)"
        return tup, tdn, sigma_h, lbl

    @staticmethod
    def _first_passage(paths, target_up, target_down, n_paths):
        hit_u = np.any(paths >= target_up,  axis=1)
        hit_d = np.any(paths <= target_down, axis=1)
        both  = hit_u & hit_d
        fw    = np.zeros(n_paths, dtype=bool)
        for i in np.where(both)[0]:
            iu  = int(np.argmax(paths[i] >= target_up))
            id_ = int(np.argmax(paths[i] <= target_down))
            fw[i] = iu < id_
        p_up = float(
            np.sum(hit_u & ~hit_d) + np.sum(fw)
        ) / max(n_paths, 1)
        p_dn = float(
            np.sum(hit_d & ~hit_u) + np.sum(both & ~fw)
        ) / max(n_paths, 1)
        return p_up, p_dn

    def _edge_score(
            self,
            prob_hit_up: float,
            prob_hit_dn: float,
            fit_quality: float,
            lam_deviation: float,
            is_jump: bool,
            ci_width_pct: float,
            wr_weight: float,
            profile: AssetProfile,
            drift: Optional[DriftSignal]      = None,
            vol_regime: Optional[VolRegimeResult] = None,
            tod_mult: float = 1.0,
    ) -> Tuple[float, str]:

        regime_strong = (
            (vol_regime is not None
             and vol_regime.signal != "neutral"
             and vol_regime.edge_score >= 0.15)
            or (drift is not None
                and drift.direction != "neutral"
                and drift.edge_score >= 0.15)
        )
        if ci_width_pct < profile.ci_width_min and not regime_strong:
            return 0.0, "neutral"

        max_p = max(prob_hit_up, prob_hit_dn)
        min_p = min(prob_hit_up, prob_hit_dn)
        gap   = max_p - min_p
        skew  = gap / (max_p + min_p + 1e-9)

        regime_bonus = 1.0
        if is_jump and abs(lam_deviation) >= profile.spike_lam_dev_min:
            regime_bonus = 1.0 + min(abs(lam_deviation) / 80.0, 0.20)

        mc_score = float(np.clip(
            skew * fit_quality * regime_bonus * wr_weight,
            0.0, 1.0))

        tod_adj  = (tod_mult - 1.0) * 0.05
        mc_score = float(np.clip(
            mc_score + tod_adj * mc_score, 0.0, 1.0))

        if (vol_regime is not None
                and profile.vol_regime_enabled
                and vol_regime.signal != "neutral"):
            w_v   = profile.drift_edge_weight
            w_m   = 1.0 - w_v
            score = float(np.clip(
                vol_regime.edge_score * w_v + mc_score * w_m,
                0.0, 1.0))
            sig = (vol_regime.signal
                   if score >= profile.signal_edge_min
                   else "neutral")
            return score, sig

        if (drift is not None
                and profile.drift_signal_enabled
                and drift.direction != "neutral"):
            w_d   = profile.drift_edge_weight
            w_m   = 1.0 - w_d
            score = float(np.clip(
                drift.edge_score * w_d + mc_score * w_m,
                0.0, 1.0))
            sig = (drift.direction
                   if score >= profile.signal_edge_min
                   else "neutral")
            return score, sig

        score = mc_score
        if (gap >= profile.direction_gap
                and score >= profile.signal_edge_min):
            sig = "bullish" if prob_hit_up > prob_hit_dn else "bearish"
        elif is_jump and score >= profile.signal_edge_min * 0.75:
            sig = "jump_imminent"
        else:
            sig = "neutral"

        return score, sig

    def run(
            self,
            sym: str,
            dm: 'DataManager',
            horizon: int,
            horizon_label: str,
            pm: 'PatternMemory',
            n: Optional[int]       = None,
            timeframe_key: Optional[str] = None,
    ) -> Optional[MCResult]:

        profile = effective_profile(sym)
        n_paths = n or profile.mc_paths

        lr = dm.log_returns(sym)
        S0 = dm.last(sym)
        if S0 is None or len(lr) < MIN_TICKS:
            return None

        ns     = min(horizon, MC_STEPS_MAX)
        is_j   = self._is_jump_sym(sym)
        regime = "NORMAL"
        cat    = _cat(sym)

        tod_mult = self.tod.multiplier(sym) if profile.tod_enabled else 1.0

        try:
            drift      = None
            vol_regime = None

            # FIX v4.3 PROBLEM 3: Compute horizon-scaled window
            # Shorter horizons → smaller window → short-term momentum only
            # Longer horizons  → larger window → macro momentum
            horizon_scaled_window = max(
                min(ns * 2, profile.drift_window), 30)

            if is_j:
                jp = self.jm.fit_unbiased(sym, lr)
                Z  = self._rng.standard_normal((n_paths, ns)).astype(np.float64)
                U  = self._rng.uniform(0, 1, (n_paths, ns)).astype(np.float64)
                Zj = self._rng.standard_normal((n_paths, ns)).astype(np.float64)
                paths = _jd_kernel(
                    float(S0), float(jp.mu), float(jp.sigma),
                    float(jp.lam_posterior),
                    float(jp.jump_mean), float(jp.jump_std),
                    float(jp.jump_sign),
                    1.0, n_paths, ns, Z, U, Zj)
                pj       = self.jm.prob_in_n(jp, ns)
                ej       = self.jm.ticks_to_next_range(
                    jp, dm.observed_tick_rate(sym))[0]
                jlo, jhi = self.jm.jump_magnitude_range(jp)
                sigma_tick = max(jp.sigma, 1e-10)
                fq         = 0.80
                lam_dev    = jp.lam_deviation

                # FIX v4.3 PROBLEM 3: horizon-aware drift for JUMP
                if profile.drift_signal_enabled:
                    drift = self.de.analyse(
                        sym, lr, S0, profile, tod_mult,
                        window_override=horizon_scaled_window)
                    regime = drift.regime

            else:
                gp = self.vm.fit(sym, lr)
                jp = None
                Z  = self._rng.standard_normal((n_paths, ns)).astype(np.float64)
                paths = _gbm_kernel(
                    float(S0), float(gp.mu), float(gp.sigma_ewma),
                    1.0, n_paths, ns, Z)
                pj = ej = jlo = jhi = None
                sigma_tick = max(gp.sigma_ewma, 1e-10)
                fq      = gp.fit_quality
                lam_dev = 0.0

                # FIX v4.3 PROBLEM 1+3: pass prices and horizon window to vre
                if profile.vol_regime_enabled:
                    prices_arr = dm.prices(sym)
                    vol_regime = self.vre.analyse(
                        sym, lr, S0, ns,
                        profile, tod_mult,
                        window_override=horizon_scaled_window,
                        prices=prices_arr,
                    )
                    regime = vol_regime.regime
                elif profile.drift_signal_enabled:
                    drift = self.de.analyse(
                        sym, lr, S0, profile, tod_mult,
                        window_override=horizon_scaled_window)
                    regime = drift.regime

            if not np.all(np.isfinite(paths)):
                paths = np.where(np.isfinite(paths), paths, S0)

            # FIX v4.3: For VOL use regime targets so MC first-passage
            # is consistent with trade setup targets
            if (vol_regime is not None
                    and profile.vol_regime_enabled
                    and vol_regime.signal != "neutral"):
                tup     = vol_regime.target_up
                tdn     = vol_regime.target_down
                sigma_h = vol_regime.realized_sigma * math.sqrt(max(ns, 1))
                k       = vol_regime.k_adaptive
                pct     = (tup - S0) / S0 * 100
                tlabel  = f"VReg k={k:.2f} ({pct:.3f}%)"
            else:
                tup, tdn, sigma_h, tlabel = self._adaptive_targets(
                    S0, sigma_tick, ns, profile)

            finals    = paths[:, -1]
            prob_up_h = float(np.mean(finals > S0))
            prob_hit_up, prob_hit_dn = self._first_passage(
                paths, tup, tdn, n_paths)

            p5_end  = float(np.percentile(finals, 5))
            p95_end = float(np.percentile(finals, 95))
            ci_width_pct = (p95_end - p5_end) / max(S0, 1e-10)

            wr_weight = pm.win_rate_weight(sym, ns)

            score, sig = self._edge_score(
                prob_hit_up, prob_hit_dn,
                fq, lam_dev, is_j,
                ci_width_pct, wr_weight,
                profile, drift, vol_regime, tod_mult)

            # FIX v4.3 PROBLEM 5: Ensemble agreement gate inside MCEngine.run
            # Run two half-path sub-ensembles and measure directional agreement
            # If below floor → degrade score and force neutral
            ens_agreement = 1.0
            if ENABLE_ENSEMBLE:
                try:
                    half     = max(n_paths // 2, 100)
                    finals_a = paths[:half, -1]
                    finals_b = paths[half:half*2, -1]

                    p_up_a = float(np.mean(finals_a > S0))
                    p_up_b = float(np.mean(finals_b > S0))

                    dir_a = "bull" if p_up_a > 0.5 else "bear"
                    dir_b = "bull" if p_up_b > 0.5 else "bear"

                    hard_agree  = 1.0 if dir_a == dir_b else 0.0
                    soft_agree  = 1.0 - abs(p_up_a - p_up_b)
                    ens_agreement = float(np.clip(
                        hard_agree * 0.5 + soft_agree * 0.5,
                        0.0, 1.0))

                    if ens_agreement < ENSEMBLE_AGREEMENT_FLOOR:
                        log.debug(
                            f"Ensemble gate [{sym}]: "
                            f"agree={ens_agreement:.3f} "
                            f"< {ENSEMBLE_AGREEMENT_FLOOR} "
                            f"→ forcing neutral")
                        # Degrade score proportionally and force neutral
                        score = score * ens_agreement
                        sig   = "neutral"

                except Exception as _ens_err:
                    log.debug(f"Ensemble gate err: {_ens_err}")
                    ens_agreement = 1.0

            implied_edge = (max(prob_hit_up, prob_hit_dn) - 0.5) * 200

            mc = MCResult(
                symbol=sym,
                S0=float(S0),
                horizon=ns,
                horizon_label=horizon_label,
                paths=paths,
                p5=np.percentile(paths,  5, axis=0),
                p25=np.percentile(paths, 25, axis=0),
                p50=np.percentile(paths, 50, axis=0),
                p75=np.percentile(paths, 75, axis=0),
                p95=np.percentile(paths, 95, axis=0),
                target_up=tup,
                target_down=tdn,
                target_label=tlabel,
                prob_up_horizon=prob_up_h,
                prob_hit_up=prob_hit_up,
                prob_hit_down=prob_hit_dn,
                sigma_horizon=sigma_h,
                ci_width_pct=float(ci_width_pct),
                prob_jump=pj,
                exp_jump_tick=ej,
                jump_magnitude_lo=jlo,
                jump_magnitude_hi=jhi,
                edge_score=score,
                signal=sig,
                implied_edge_pct=implied_edge,
                timeframe_key=timeframe_key,
                regime=(vol_regime.regime if vol_regime
                        else (drift.regime if drift else regime)),
                profile_name=profile.name,
                drift_signal=drift,
                vol_regime=vol_regime,
                tod_multiplier=tod_mult,
                ens_agreement=ens_agreement,
            )

            ts = self.tsb.build(
                mc, horizon_label, ns,
                mc.regime, profile,
                drift, vol_regime, tod_mult)

            # FIX v4.3: Apply HybridTargetRefiner
            # Uses empirical cold path when pattern memory insufficient
            if ts is not None:
                try:
                    prices_for_refiner = dm.prices(sym)
                    is_bull_ts         = (ts.direction == "BUY")
                    ref_target, ref_stop, ref_method = self.htr.refine(
                        sym=sym,
                        S0=float(S0),
                        is_bull=is_bull_ts,
                        mc_target=ts.target,
                        mc_stop=ts.invalidation,
                        finals=finals,
                        prices=prices_for_refiner,
                        horizon=ns,
                        records=pm.records,
                    )
                    ref_target_pct = abs(ref_target - S0) / S0 * 100
                    ref_stop_pct   = abs(ref_stop   - S0) / S0 * 100
                    if (ref_target_pct >= MIN_TARGET_MOVE * 100
                            and ref_stop_pct > 1e-6):
                        ref_rr = min(
                            ref_target_pct / max(ref_stop_pct, 1e-10),
                            4.0)
                        if ref_rr >= MIN_RR_GLOBAL:
                            ts.target       = ref_target
                            ts.invalidation = ref_stop
                            ts.target_pct   = ref_target_pct
                            ts.stop_pct     = ref_stop_pct
                            ts.rr_ratio     = ref_rr
                            ts.signal_source = (
                                ts.signal_source + f"+{ref_method}")
                            log.debug(
                                f"HybridRefiner [{sym}]: "
                                f"{ref_method} "
                                f"tgt={ref_target:.6f} "
                                f"rr={ref_rr:.2f}")
                except Exception as _htr_err:
                    log.debug(f"HybridRefiner wire: {_htr_err}")

            mc.trade_setup = ts
            return mc

        except Exception as e:
            log.error(f"MCEngine.run error ({sym}): {e}")
            return None

# ─────────────────────────────────────────────────────────────────────────────
# NARRATIVE ENGINE v4.3
# ─────────────────────────────────────────────────────────────────────────────
class NarrativeEngine:
    def __init__(self, vm: VolatilityModel, jm: JumpDiffusionModel):
        self.vm = vm
        self.jm = jm

    def _regime_desc(self, regime: str) -> str:
        return {
            "VOLATILE":   "Volatility ELEVATED — recent moves larger than usual.",
            "RANGING":    "Market RANGING — volatility compressed.",
            "TRENDING":   "Market TRENDING — consistent directional drift.",
            "NORMAL":     "Market behaving NORMALLY.",
            "COMPRESSED": "Vol COMPRESSED — realized below advertised target.",
            "EXPANDED":   "Vol EXPANDED — realized above advertised target.",
        }.get(regime, "Normal regime.")

    def _vol_regime_block(self, vr: VolRegimeResult) -> str:
        dir_icon = ("📈" if vr.signal == "bullish"
                    else ("📉" if vr.signal == "bearish" else "⬜"))
        sig_icon = "✅" if vr.signal != "neutral" else "❌"
        abs_mom  = abs(vr.normalized_momentum)
        filled   = int(min(abs_mom / 5 * 10, 10))
        mom_bar  = "█" * filled + "░" * (10 - filled)
        lines = [
            f"\n*Vol Regime Engine [v4.3]:*",
            f"  Regime:       *{vr.regime}*",
            f"  Direction:    *{vr.signal.upper()}* {dir_icon} {sig_icon}",
            f"  Mom z-score:  [{mom_bar}] `{vr.normalized_momentum:+.4f}`",
            f"  k_adaptive:   `{vr.k_adaptive:.3f}`",
            f"  Vol deviation:`{vr.vol_deviation:+.3f}` "
            f"({'above' if vr.vol_deviation > 0 else 'below'} target)",
            f"  V-Edge:       `{vr.edge_score:.4f}`",
            f"  Strength:     *{vr.signal_strength}*",
            f"  ToD mult:     `x{vr.tod_multiplier:.3f}`",
        ]
        return "\n".join(lines)

    def _drift_block(self, drift: Optional[DriftSignal]) -> str:
        if drift is None:
            return ""
        dir_icon = ("📈" if drift.direction == "bullish"
                    else ("📉" if drift.direction == "bearish" else "⬜"))
        filled   = int(abs(drift.ewma_momentum) * 10)
        mom_bar  = "█" * filled + "░" * (10 - filled)
        sig_icon = "✅" if drift.direction != "neutral" else "❌"
        lines = [
            f"\n*Drift Analysis [v4.3]:*",
            f"  Direction:   *{drift.direction.upper()}* {dir_icon} {sig_icon}",
            f"  t-statistic: `{drift.tstat:+.4f}` (thresh={drift.window_used}t)",
            f"  p-value:     `{drift.pvalue:.4f}`"
            + (" ✓ significant" if drift.pvalue < 0.15 else " ✗ not significant"),
            f"  EWMA mom:   [{mom_bar}] `{drift.ewma_momentum:+.4f}`",
            f"  Regime:      *{drift.regime}*",
            f"  D-Edge:      `{drift.edge_score:.4f}`",
            f"  Strength:    *{drift.signal_strength}*",
        ]
        return "\n".join(lines)

    def vol_context(
            self, sym: str, gp: GBMParams,
            mc: MCResult, tf_label: str) -> str:
        profile     = effective_profile(sym)
        ann         = self.vm.ann_vol(gp) * 100
        dev         = self.vm.deviation(gp)
        regime_desc = self._regime_desc(mc.regime)

        lines = [
            f"*Market Context — {_friendly(sym)} ({tf_label})*\n",
            f"Profile: *{profile.name}* | Engine: *VOL_REGIME+MC* | "
            f"Regime: *{mc.regime}* | ToD: x{mc.tod_multiplier:.3f}",
            f"{regime_desc}",
        ]
        if gp.advertised_vol > 0:
            lines.append(
                f"\nDesigned vol: *{gp.advertised_vol*100:.0f}%* ann. "
                f"Live EWMA: *{ann:.2f}%* "
                f"({'above' if dev > 0 else 'below'} by {abs(dev):.1f}%).")
        lines.append(
            f"Expected move ({tf_label}): "
            f"*+-{mc.sigma_horizon*100:.4f}%* (1-sigma).")

        if mc.vol_regime:
            lines.append(self._vol_regime_block(mc.vol_regime))
        if mc.drift_signal:
            lines.append(self._drift_block(mc.drift_signal))

        # Ensemble agreement display
        if mc.ens_agreement < 1.0:
            ens_icon = "✅" if mc.ens_agreement >= ENSEMBLE_AGREEMENT_FLOOR else "⚠️"
            lines.append(
                f"\nEnsemble agreement: "
                f"{ens_icon} `{mc.ens_agreement:.3f}`"
                + (" — signal degraded" if mc.ens_agreement < ENSEMBLE_AGREEMENT_FLOOR
                   else ""))

        if mc.signal == "bullish":
            lines.append(
                f"\n*Signal — BULLISH* 🟢\n"
                f"Vol regime + momentum confirm UP. "
                f"Edge: {mc.edge_score:.3f}")
        elif mc.signal == "bearish":
            lines.append(
                f"\n*Signal — BEARISH* 🔴\n"
                f"Vol regime + momentum confirm DOWN. "
                f"Edge: {mc.edge_score:.3f}")
        else:
            lines.append(
                "\n*Signal — NEUTRAL* ⬜\n"
                "Vol regime not statistically significant. "
                "No trade setup.")

        if gp.ks_pvalue < 0.05:
            lines.append(
                "\n*Fat tails detected* (KS p<0.05) — use wider stops.")

        if mc.trade_setup:
            ts = mc.trade_setup
            lines.append(
                f"\n*Trade setup:* {ts.direction} | "
                f"R/R *1:{ts.rr_ratio:.2f}* | "
                f"P(tgt) *{ts.prob_target:.2%}* | "
                f"EV `{ts.expected_value:+.4f}%` | "
                f"[{ts.signal_source}]")
        else:
            lines.append(
                f"\n*No valid trade setup.* "
                f"Gates: P>={MIN_PROB_TARGET:.0%} "
                f"EV>0 RR>={MIN_RR_GLOBAL:.1f} "
                f"Move>={MIN_TARGET_MOVE*100:.2f}%.")
        return "\n".join(lines)

    def jump_context(
            self, sym: str, jp: JumpParams,
            mc: MCResult, tf_label: str) -> str:
        profile = effective_profile(sym)
        cat     = _cat(sym)
        fname   = _friendly(sym)
        freq    = EXPECTED_JUMP_FREQ.get(sym, 0)
        jlo     = mc.jump_magnitude_lo or 0
        jhi     = mc.jump_magnitude_hi or 0

        lines = [
            f"*Market Context — {fname} ({tf_label})*\n",
            f"Profile: *{profile.name}* | "
            f"Regime: *{mc.regime}* | "
            f"ToD: x{mc.tod_multiplier:.3f}\n",
        ]
        if cat == "jump":
            lines.append(
                f"*Jump Index:* Bidirectional jumps every *~{freq} ticks*. "
                f"Dominant: *{'UP' if jp.jump_sign > 0 else 'DOWN'}*.")
        elif cat == "boom":
            lines.append(
                f"*Boom Index:* Upward spike every *~{freq} ticks*. "
                f"Fitted: 1/~{1/max(jp.lam_posterior,1e-10):.0f}t (Bayesian λ).")
        elif cat == "crash":
            lines.append(
                f"*Crash Index:* Downward spike every *~{freq} ticks*. "
                f"Fitted: 1/~{1/max(jp.lam_posterior,1e-10):.0f}t (Bayesian λ).")

        lines.append(
            f"Spike P60: *{jp.spike_mag_p60*100:.3f}%* | "
            f"P75: *{jp.spike_mag_p75*100:.3f}%* | "
            f"Hazard: *{jp.hazard_intensity:.0%}*")

        if jp.counter_drift != 0.0:
            cd_dir = "📈 UP" if jp.counter_drift > 0 else "📉 DOWN"
            lines.append(
                f"Counter-drift: *{cd_dir}* slope={jp.counter_drift:.8f}")

        if mc.prob_jump is not None:
            lines.append(
                f"P(spike/{mc.horizon}t): *{mc.prob_jump:.1%}* (posterior λ)")

        if abs(jp.lam_deviation) > profile.spike_lam_dev_min:
            d = "SUPPRESSED" if jp.lam_deviation < 0 else "ELEVATED"
            lines.append(
                f"\nRate *{d}* by *{abs(jp.lam_deviation):.0f}%* vs theoretical"
                + ("— spike may be overdue."
                   if jp.lam_deviation < 0
                   else "— frequent spike regime."))

        if mc.drift_signal:
            lines.append(self._drift_block(mc.drift_signal))

        if mc.ens_agreement < 1.0:
            ens_icon = "✅" if mc.ens_agreement >= ENSEMBLE_AGREEMENT_FLOOR else "⚠️"
            lines.append(
                f"\nEnsemble: {ens_icon} `{mc.ens_agreement:.3f}`"
                + (" — degraded" if mc.ens_agreement < ENSEMBLE_AGREEMENT_FLOOR
                   else ""))

        if mc.signal == "bullish":
            lines.append(f"\n*Signal — BULLISH* 🟢 Edge={mc.edge_score:.3f}")
        elif mc.signal == "bearish":
            lines.append(f"\n*Signal — BEARISH* 🔴 Edge={mc.edge_score:.3f}")
        elif mc.signal == "jump_imminent":
            lines.append(f"\n*Signal — JUMP IMMINENT* ⚡ Edge={mc.edge_score:.3f}")
        else:
            lines.append("\n*Signal — NEUTRAL* ⬜ No confirmed edge.")
        return "\n".join(lines)

    def step_context(
            self, sym: str, gp: GBMParams,
            mc: MCResult, tf_label: str) -> str:
        lines = [
            f"*Step Index ({tf_label})*\n",
            f"Fixed +-0.1 per tick. Random direction.\n",
        ]
        if mc.drift_signal:
            lines.append(self._drift_block(mc.drift_signal))
        lines += [
            f"\nEWMA sigma: *{gp.sigma_ewma:.8f}*/tick",
            f"90% CI: *{mc.ci_width_pct*100:.4f}%*",
            f"ToD: x{mc.tod_multiplier:.3f}",
        ]
        if mc.ens_agreement < 1.0:
            lines.append(
                f"Ensemble: `{mc.ens_agreement:.3f}`"
                + (" ⚠️ degraded"
                   if mc.ens_agreement < ENSEMBLE_AGREEMENT_FLOOR else ""))
        if mc.trade_setup:
            ts = mc.trade_setup
            lines.append(
                f"\n*Range Signal:* {ts.direction} via drift. "
                f"R/R *1:{ts.rr_ratio:.2f}* | "
                f"EV `{ts.expected_value:+.4f}%`.")
        else:
            lines.append(
                "\n*No signal.* Step is near-random. "
                "Drift must be statistically significant "
                "with confirmed momentum.")
        return "\n".join(lines)

    def risk_reward_block(self, mc: MCResult, risk_pct: float) -> str:
        if mc.trade_setup:
            ts = mc.trade_setup
            return (
                f"\n*Risk/Reward ({risk_pct:.0f}% risk):*\n"
                f"Direction:    *{ts.direction}*\n"
                f"Entry:        `{ts.entry:.6f}`\n"
                f"Target:       `{ts.target:.6f}` "
                f"({'+' if ts.direction=='BUY' else '-'}{ts.target_pct:.3f}%)\n"
                f"Invalidation: `{ts.invalidation:.6f}` "
                f"({'-' if ts.direction=='BUY' else '+'}{ts.stop_pct:.3f}%)\n"
                f"R/R Ratio:    *1 : {ts.rr_ratio:.2f}*\n"
                f"P(target):    `{ts.prob_target:.2%}`\n"
                f"EV/trade:     `{ts.expected_value:+.4f}%`\n"
                f"Edge:         `{ts.edge_pct:+.2f}%`\n"
                f"Source:       *{ts.signal_source}*\n"
                f"ToD:          x{ts.tod_multiplier:.3f}\n"
                f"Ensemble:     `{mc.ens_agreement:.3f}`"
            )
        tup  = mc.target_up
        tdn  = mc.target_down
        r_up = (tup - mc.S0) / mc.S0 * 100
        r_dn = (mc.S0 - tdn) / mc.S0 * 100
        prob = max(mc.prob_hit_up, mc.prob_hit_down)
        ev   = prob * r_up - (1 - prob) * r_dn
        return (
            f"\n*R/R ({risk_pct:.0f}% risk):*\n"
            f"Target: `{tup:.5f}` (+{r_up:.3f}%)\n"
            f"Stop:   `{tdn:.5f}` (-{r_dn:.3f}%)\n"
            f"EV:     `{ev:+.4f}%`\n"
            f"Note:   Trade setup did not meet quality gates "
            f"(P>={MIN_PROB_TARGET:.0%} EV>0 "
            f"RR>={MIN_RR_GLOBAL:.1f} "
            f"Move>={MIN_TARGET_MOVE*100:.2f}%)."
        )

# ─────────────────────────────────────────────────────────────────────────────
# CHART GENERATOR v4.3
# ─────────────────────────────────────────────────────────────────────────────
class ChartGen:
    def __init__(self):
        plt.rcParams.update({
            "figure.facecolor":  BG,
            "axes.facecolor":    AX,
            "axes.edgecolor":    GY,
            "axes.labelcolor":   WH,
            "xtick.color":       GY,
            "ytick.color":       GY,
            "text.color":        WH,
            "grid.color":        DG,
            "grid.alpha":        0.8,
            "font.family":       "monospace",
            "axes.spines.top":   False,
            "axes.spines.right": False,
        })

    def make(
            self, fname: str, sym: str,
            dm: 'DataManager',
            mc: Optional[MCResult],
            gp: Optional[GBMParams],
            jp: Optional[JumpParams],
            vm: VolatilityModel,
            note: str,
            tod: TimeOfDayProfile) -> BytesIO:

        profile = effective_profile(sym)
        prices  = dm.prices(sym)
        recent  = prices[-500:] if len(prices) >= 500 else prices
        is_j    = jp is not None

        fig = plt.figure(figsize=(20, 14), facecolor=BG)
        gs  = gridspec.GridSpec(
            3, 4, figure=fig,
            height_ratios=[3.8, 1.6, 0.85],
            width_ratios=[2.5, 1.0, 1.0, 1.0],
            hspace=0.50, wspace=0.32,
        )

        # ── Panel 1: Price + MC paths ─────────────────────────────────────
        a1 = fig.add_subplot(gs[0, :])
        a1.set_facecolor(AX)
        xh = np.arange(-len(recent), 0)
        a1.plot(xh, recent, color=BL, lw=1.6,
                label="Live Price", zorder=5, alpha=0.95)
        a1.axvspan(xh[0], 0, alpha=0.05, color=BL)

        if mc is not None:
            ns  = mc.horizon
            xf  = np.arange(0, ns + 1)
            idx = np.random.choice(
                mc.paths.shape[0],
                min(40, mc.paths.shape[0]),
                replace=False)
            for i in idx:
                a1.plot(xf, mc.paths[i],
                        color=PU, alpha=0.030, lw=0.5)
            a1.fill_between(xf, mc.p5, mc.p95,
                            alpha=0.12, color=PU,
                            label="90% CI", zorder=2)
            a1.fill_between(xf, mc.p25, mc.p75,
                            alpha=0.25, color=BL,
                            label="50% CI", zorder=3)
            a1.plot(xf, mc.p50, color=GR, lw=2.2,
                    ls="--", label="Median", zorder=6)
            a1.axhline(mc.S0, color=YL, lw=0.8,
                       ls=":", alpha=0.7,
                       label="Entry", zorder=4)

            if mc.trade_setup:
                ts  = mc.trade_setup
                tc  = GR2 if ts.direction == "BUY" else RD2
                ic  = RD2 if ts.direction == "BUY" else GR2
                a1.axhline(ts.target, color=tc, lw=2.0,
                           ls="-", alpha=0.90, label="TARGET")
                a1.axhline(ts.invalidation, color=ic, lw=1.8,
                           ls="--", alpha=0.85, label="STOP")
                xmid = ns * 0.55
                sg   = '+' if ts.direction == 'BUY' else '-'
                a1.annotate(
                    f"TARGET {sg}{ts.target_pct:.2f}%\n"
                    f"RR 1:{ts.rr_ratio:.2f} "
                    f"EV:{ts.expected_value:+.3f}%\n"
                    f"P={ts.prob_target:.1%} "
                    f"Ens:{mc.ens_agreement:.2f}",
                    xy=(xmid, ts.target),
                    fontsize=8, color=tc,
                    fontweight="bold",
                    va="bottom" if ts.direction == "BUY" else "top",
                    bbox=dict(boxstyle="round,pad=0.2",
                              fc=BG, ec=tc, alpha=0.85))
                sg2 = '-' if ts.direction == 'BUY' else '+'
                a1.annotate(
                    f"STOP {sg2}{ts.stop_pct:.2f}%",
                    xy=(xmid, ts.invalidation),
                    fontsize=8, color=ic,
                    fontweight="bold",
                    va="top" if ts.direction == "BUY" else "bottom",
                    bbox=dict(boxstyle="round,pad=0.2",
                              fc=BG, ec=ic, alpha=0.85))
            else:
                a1.axhline(mc.target_up, color=GR2, lw=1.4,
                           ls="-.", alpha=0.8, label="Upper")
                a1.axhline(mc.target_down, color=RD2, lw=1.4,
                           ls="-.", alpha=0.8, label="Lower")

            a1.axvline(0, color=OR, lw=2.0, ls="--",
                       alpha=0.9, label="NOW", zorder=7)

            if (mc.vol_regime
                    and mc.vol_regime.signal != "neutral"):
                vr  = mc.vol_regime
                arc = GR2 if vr.signal == "bullish" else RD2
                a1.annotate(
                    f"VOL_REGIME "
                    f"{'▲' if vr.signal=='bullish' else '▼'}"
                    f" k={vr.k_adaptive:.2f} "
                    f"dev={vr.vol_deviation:+.3f} "
                    f"Ens:{mc.ens_agreement:.2f}",
                    xy=(ns * 0.05, mc.S0),
                    fontsize=8.5, color=arc,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3",
                              fc=BG, ec=arc, alpha=0.90))
            elif (mc.drift_signal
                  and mc.drift_signal.direction != "neutral"):
                ds  = mc.drift_signal
                arc = GR2 if ds.direction == "bullish" else RD2
                a1.annotate(
                    f"DRIFT "
                    f"{'▲' if ds.direction=='bullish' else '▼'}"
                    f" t={ds.tstat:+.2f} "
                    f"p={ds.pvalue:.3f} "
                    f"win={ds.window_used}",
                    xy=(ns * 0.05, mc.S0),
                    fontsize=8.5, color=arc,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3",
                              fc=BG, ec=arc, alpha=0.90))

            self._prob_box(a1, mc, profile)

        regime_str = (f" | {mc.regime}"
                      if mc and mc.regime != "NORMAL" else "")
        a1.set_title(
            f"  SQE v4.3  ·  {fname}  ·  "
            f"[{profile.name}]  ·  "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            fontsize=12, color=WH,
            loc="left", pad=12, fontweight="bold")
        if mc:
            src    = mc.trade_setup.signal_source if mc.trade_setup else "—"
            ts_str = ""
            if mc.trade_setup:
                ts = mc.trade_setup
                ts_str = (
                    f"{ts.direction} "
                    f"RR1:{ts.rr_ratio:.2f} "
                    f"P={ts.prob_target:.1%} "
                    f"EV:{ts.expected_value:+.3f}% | ")
            a1.set_title(
                f"{mc.horizon_label}{regime_str} | "
                f"{ts_str}"
                f"{mc.signal.upper()} | "
                f"Edge {mc.edge_score:.3f} | "
                f"src:{src} | "
                f"Ens:{mc.ens_agreement:.2f} | "
                f"ToD:x{mc.tod_multiplier:.3f}",
                fontsize=9, color=OR,
                loc="right", pad=12)
        a1.set_xlabel("Ticks (history | forecast)",
                      fontsize=9, color=GY)
        a1.set_ylabel("Price", fontsize=9, color=GY)
        a1.legend(fontsize=7.5, framealpha=0.2,
                  facecolor=BG, edgecolor=GY,
                  ncol=7, loc="upper left")
        a1.grid(True, lw=0.35, color=DG)

        # ── Panel 2: Return distribution ──────────────────────────────────
        a2 = fig.add_subplot(gs[1, 0])
        a2.set_facecolor(AX)
        lr = dm.log_returns(sym)
        if len(lr) > 50:
            lo  = np.percentile(lr, 0.5)
            hi  = np.percentile(lr, 99.5)
            lc  = lr[(lr >= lo) & (lr <= hi)]
            n_b = min(100, max(30, len(lc) // 20))
            a2.hist(lc, bins=n_b, color=BL, alpha=0.65,
                    density=True, label="Returns",
                    edgecolor="none")
            if gp and gp.sigma > 1e-10:
                x = np.linspace(lc.min(), lc.max(), 500)
                a2.plot(x, norm.pdf(x, gp.mu, gp.sigma),
                        color=GR, lw=2.0, label="GBM fit")
                thr = JUMP_THRESHOLD * gp.sigma
                a2.axvline(gp.mu + thr, color=RD,
                           lw=0.9, ls="--", alpha=0.7,
                           label="Jump thr")
                a2.axvline(gp.mu - thr, color=RD,
                           lw=0.9, ls="--", alpha=0.7)
                if gp.mu != 0:
                    a2.axvline(gp.mu, color=YL, lw=1.5,
                               ls="-", alpha=0.9,
                               label=f"drift={gp.mu:.6f}")
            if is_j and jp:
                ij, _ = JumpDiffusionModel().detect(lc)
                jv    = lc[ij]
                if len(jv) > 0:
                    a2.scatter(jv, np.zeros_like(jv),
                               color=OR, s=20, zorder=5,
                               label="Jumps")
        a2.set_title("Return Distribution",
                     fontsize=9, color=GY, pad=5)
        a2.legend(fontsize=7, framealpha=0.15, facecolor=BG)
        a2.grid(True, lw=0.25)
        a2.set_xlabel("Log-return", fontsize=8)

        # ── Panel 3: Vol Regime / Drift panel ────────────────────────────
        a3 = fig.add_subplot(gs[1, 1])
        a3.set_facecolor(AX)
        if mc and mc.vol_regime:
            self._vol_regime_panel(a3, lr, mc, profile)
            a3.set_title("Vol Regime Analysis",
                         fontsize=9, color=GY, pad=5)
        else:
            self._drift_panel(a3, lr, mc, profile)
            a3.set_title("Drift t-stat Analysis",
                         fontsize=9, color=GY, pad=5)
        a3.grid(True, lw=0.25)

        # ── Panel 4: Probability gauges ───────────────────────────────────
        a4 = fig.add_subplot(gs[1, 2])
        a4.set_facecolor(AX)
        if mc:
            self._prob_gauges(a4, mc, profile)
        a4.set_title("Probability Gauges",
                     fontsize=9, color=GY, pad=5)
        a4.grid(True, axis="x", lw=0.25)

        # ── Panel 5: ToD heatmap ──────────────────────────────────────────
        a5 = fig.add_subplot(gs[1, 3])
        a5.set_facecolor(AX)
        self._tod_panel(a5, sym, tod)
        a5.set_title("Time-of-Day Profile",
                     fontsize=9, color=GY, pad=5)
        a5.grid(True, lw=0.25)

        # ── Panel 6: Summary bar ──────────────────────────────────────────
        a6 = fig.add_subplot(gs[2, :])
        a6.set_facecolor(BG)
        a6.axis("off")
        self._param_box(a6, sym, dm, mc, gp, jp, vm, note, profile, tod)

        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=115,
                    bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        buf.seek(0)
        return buf

    @staticmethod
    def _vol_regime_panel(
            ax, lr: np.ndarray,
            mc: Optional[MCResult],
            profile: AssetProfile):
        if len(lr) < 60:
            ax.text(0.5, 0.5, "Not enough data",
                    ha="center", va="center",
                    color=GY, transform=ax.transAxes)
            return

        window  = min(profile.drift_window, len(lr) // 3, 200)
        step    = max(1, window // 10)
        zscores = []
        indices = list(range(window, len(lr), step))

        for i in indices:
            seg      = lr[i-window:i]
            std_s    = float(np.std(seg, ddof=1))
            sig_ewma = max(std_s, 1e-10)
            mu_s     = float(np.mean(seg))
            se       = sig_ewma / math.sqrt(len(seg))
            zscores.append(float(np.clip(
                mu_s / max(se, 1e-12), -5, 5)))

        if not zscores:
            return

        zscores = np.array(zscores)
        xvals   = np.array(indices)
        w       = max(1, step)
        thr     = profile.drift_tstat_min * 0.60

        pos_sig = zscores >= thr
        neg_sig = zscores <= -thr
        pos_ns  = (zscores > 0) & ~pos_sig
        neg_ns  = (zscores < 0) & ~neg_sig

        if np.any(pos_sig):
            ax.bar(xvals[pos_sig], zscores[pos_sig],
                   color=GR2, alpha=0.9, width=w)
        if np.any(neg_sig):
            ax.bar(xvals[neg_sig], zscores[neg_sig],
                   color=RD2, alpha=0.9, width=w)
        if np.any(pos_ns):
            ax.bar(xvals[pos_ns], zscores[pos_ns],
                   color=GR2, alpha=0.3, width=w)
        if np.any(neg_ns):
            ax.bar(xvals[neg_ns], zscores[neg_ns],
                   color=RD2, alpha=0.3, width=w)

        ax.axhline(thr, color=YL, lw=1.2, ls="--",
                   alpha=0.8, label=f"+{thr:.1f}")
        ax.axhline(-thr, color=YL, lw=1.2, ls="--",
                   alpha=0.8, label=f"-{thr:.1f}")
        ax.axhline(0, color=GY, lw=0.8, alpha=0.5)

        if mc and mc.vol_regime:
            vr  = mc.vol_regime
            col = (GR2 if vr.signal == "bullish"
                   else (RD2 if vr.signal == "bearish" else GY))
            ax.set_title(
                f"[{vr.signal.upper()}] "
                f"z={vr.normalized_momentum:+.2f} "
                f"k={vr.k_adaptive:.2f}",
                fontsize=8, color=col, pad=3)

        ax.legend(fontsize=7, framealpha=0.15, facecolor=BG)
        ax.set_ylabel("Mom z-score", fontsize=8)
        ax.set_xlabel("Tick index", fontsize=8)

    @staticmethod
    def _drift_panel(
            ax, lr: np.ndarray,
            mc: Optional[MCResult],
            profile: AssetProfile):
        if len(lr) < 60:
            ax.text(0.5, 0.5, "Not enough data",
                    ha="center", va="center",
                    color=GY, transform=ax.transAxes)
            return

        window  = min(profile.drift_window, len(lr) // 3, 200)
        step    = max(1, window // 10)
        tstats  = []
        pvalues = []
        indices = list(range(window, len(lr), step))

        for i in indices:
            seg = lr[i-window:i]
            mu  = float(np.mean(seg))
            std = float(np.std(seg, ddof=1))
            if std < 1e-12:
                tstats.append(0.0)
                pvalues.append(1.0)
            else:
                t = mu / (std / math.sqrt(len(seg)))
                p = float(2 * (1 - norm.cdf(abs(t))))
                tstats.append(float(t))
                pvalues.append(p)

        if not tstats:
            return

        tstats  = np.array(tstats)
        pvalues = np.array(pvalues)
        xvals   = np.array(indices)
        w       = max(1, step)

        sig_mask = pvalues < 0.15
        pos_sig  = (tstats >= 0) & sig_mask
        neg_sig  = (tstats < 0)  & sig_mask
        pos_ns   = (tstats >= 0) & ~sig_mask
        neg_ns   = (tstats < 0)  & ~sig_mask

        if np.any(pos_sig):
            ax.bar(xvals[pos_sig], tstats[pos_sig],
                   color=GR2, alpha=0.9, width=w)
        if np.any(neg_sig):
            ax.bar(xvals[neg_sig], tstats[neg_sig],
                   color=RD2, alpha=0.9, width=w)
        if np.any(pos_ns):
            ax.bar(xvals[pos_ns], tstats[pos_ns],
                   color=GR2, alpha=0.3, width=w)
        if np.any(neg_ns):
            ax.bar(xvals[neg_ns], tstats[neg_ns],
                   color=RD2, alpha=0.3, width=w)

        thr = profile.drift_tstat_min
        ax.axhline(thr, color=YL, lw=1.2, ls="--",
                   alpha=0.8, label=f"+{thr:.1f}")
        ax.axhline(-thr, color=YL, lw=1.2, ls="--",
                   alpha=0.8, label=f"-{thr:.1f}")
        ax.axhline(0, color=GY, lw=0.8, alpha=0.5)

        if mc and mc.drift_signal:
            ds      = mc.drift_signal
            col     = (GR2 if ds.direction == "bullish"
                       else (RD2 if ds.direction == "bearish" else GY))
            sig_str = " ✓ SIG" if ds.pvalue < 0.15 else " ✗ NS"
            ax.set_title(
                f"[{ds.direction.upper()}] "
                f"t={ds.tstat:+.2f}{sig_str} "
                f"win={ds.window_used}",
                fontsize=8, color=col, pad=3)

        ax.legend(fontsize=7, framealpha=0.15, facecolor=BG)
        ax.set_ylabel("t-stat", fontsize=8)
        ax.set_xlabel("Tick index", fontsize=8)

    @staticmethod
    def _tod_panel(ax, sym: str, tod: TimeOfDayProfile):
        tod._ensure(sym)
        buckets = tod._buckets[sym]
        mults   = [b.multiplier for b in buckets]
        hours   = [i / 2 for i in range(TOD_BUCKETS)]

        colors = []
        for m in mults:
            if m >= 1.05:
                colors.append(GR2)
            elif m <= 0.95:
                colors.append(RD2)
            else:
                colors.append(GY)

        ax.bar(hours, mults, width=0.45,
               color=colors, alpha=0.75)
        ax.axhline(1.0, color=YL, lw=1.0,
                   ls="--", alpha=0.7, label="Neutral")
        ax.axhline(TOD_MULT_MAX, color=GR2,
                   lw=0.7, ls=":", alpha=0.5)
        ax.axhline(TOD_MULT_MIN, color=RD2,
                   lw=0.7, ls=":", alpha=0.5)

        cur_h = _tod_bucket() / 2
        ax.axvline(cur_h, color=OR, lw=1.5,
                   ls="-", alpha=0.9, label="NOW")
        ax.set_xlim(0, 24)
        ax.set_ylim(TOD_MULT_MIN - 0.02, TOD_MULT_MAX + 0.02)
        ax.set_xlabel("UTC Hour", fontsize=8)
        ax.set_ylabel("Multiplier", fontsize=8)
        ax.legend(fontsize=7, framealpha=0.15, facecolor=BG)

        b = buckets[_tod_bucket()]
        ax.set_title(
            f"Now: x{b.multiplier:.3f} "
            f"wr={b.win_rate:.0%} n={b.count:.0f}",
            fontsize=8, color=OR, pad=3)

    @staticmethod
    def _prob_box(ax, mc: MCResult, profile: AssetProfile):
        sig_c = {
            "bullish":       GR,
            "bearish":       RD,
            "jump_imminent": OR,
            "neutral":       GY,
        }.get(mc.signal, GY)

        ts_line = ""
        if mc.trade_setup:
            ts = mc.trade_setup
            ts_line = (
                f"\n{ts.direction} "
                f"RR1:{ts.rr_ratio:.2f} "
                f"P={ts.prob_target:.1%} "
                f"EV={ts.expected_value:+.3f}% "
                f"[{ts.signal_strength}] "
                f"src:{ts.signal_source}")
        else:
            ts_line = "\nNo valid setup (quality gates)"

        vr_line = ""
        if mc.vol_regime:
            vr = mc.vol_regime
            vr_line = (
                f"\nVolReg: {vr.signal.upper()} "
                f"z={vr.normalized_momentum:+.2f} "
                f"k={vr.k_adaptive:.2f}")

        drift_line = ""
        if mc.drift_signal:
            ds       = mc.drift_signal
            sig_icon = "✓" if ds.pvalue < 0.15 else "✗"
            drift_line = (
                f"\nDrift: {ds.direction.upper()} "
                f"t={ds.tstat:+.2f} "
                f"p={ds.pvalue:.3f} {sig_icon} "
                f"win={ds.window_used}")

        pj_line = ""
        if mc.prob_jump is not None:
            pj_line = (
                f"\nP(spike {mc.horizon}t): "
                f"{mc.prob_jump:.1%} (Bayesian λ)")

        ens_line = (
            f"\nEnsemble: {mc.ens_agreement:.3f}"
            + (" ⚠️" if mc.ens_agreement < ENSEMBLE_AGREEMENT_FLOOR else " ✓"))

        txt = (
            f"Signal: {mc.signal.upper()} [{profile.name}]\n"
            f"Edge: {mc.edge_score:.3f} | "
            f"CI: {mc.ci_width_pct*100:.4f}%\n"
            f"P(up): {mc.prob_hit_up:.1%}  "
            f"P(dn): {mc.prob_hit_down:.1%}"
            f"{vr_line}"
            f"{drift_line}"
            f"{pj_line}"
            f"{ens_line}"
            f"{ts_line}"
            f"\nToD: x{mc.tod_multiplier:.3f}"
        )
        ax.text(
            0.01, 0.97, txt,
            transform=ax.transAxes,
            fontsize=7.8, va="top", color=WH,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.45",
                      fc="#0d1b2a", ec=sig_c, alpha=0.92))

    @staticmethod
    def _prob_gauges(ax, mc: MCResult, profile: AssetProfile):
        labels = ["P(up\nhorizon)", "P(hit\ntarget)", "P(hit\nstop)"]
        vals   = [mc.prob_up_horizon, mc.prob_hit_up, mc.prob_hit_down]
        colors = [
            BL,
            (GR if mc.prob_hit_up > 0.55
             else (RD if mc.prob_hit_up < 0.50 else YL)),
            (RD2 if mc.prob_hit_down > 0.55
             else (GR if mc.prob_hit_down < 0.48 else YL)),
        ]

        if mc.vol_regime:
            vr = mc.vol_regime
            labels.append("VReg\nedge")
            vals.append(vr.edge_score)
            colors.append(
                GR if vr.signal == "bullish"
                else (RD if vr.signal == "bearish" else GY))
        elif mc.drift_signal:
            ds = mc.drift_signal
            labels.append("Drift\nedge")
            vals.append(ds.edge_score)
            colors.append(
                GR if ds.direction == "bullish"
                else (RD if ds.direction == "bearish" else GY))

        if mc.prob_jump is not None:
            labels.append(f"P(spike\n{mc.horizon}t)")
            vals.append(mc.prob_jump)
            colors.append(
                OR if mc.prob_jump > profile.spike_min_prob else GY)

        if mc.trade_setup:
            labels.append("R/R\nrating")
            rr_norm = min(mc.trade_setup.rr_ratio / 4.0, 1.0)
            vals.append(rr_norm)
            colors.append(
                GR if rr_norm > 0.5 else (YL if rr_norm > 0.3 else RD))

            labels.append("EV\n(norm)")
            ev_norm = float(np.clip(
                mc.trade_setup.expected_value / 0.5, 0.0, 1.0))
            vals.append(ev_norm)
            colors.append(
                GR if ev_norm > 0.3 else (YL if ev_norm > 0.1 else RD))

            labels.append("ToD\nmult")
            tod_norm = float(np.clip(
                (mc.tod_multiplier - TOD_MULT_MIN)
                / (TOD_MULT_MAX - TOD_MULT_MIN),
                0.0, 1.0))
            vals.append(tod_norm)
            colors.append(
                GR if mc.tod_multiplier > 1.05
                else (RD if mc.tod_multiplier < 0.95 else GY))

        # FIX v4.3: Ensemble agreement gauge
        labels.append("Ensemble\nagree")
        vals.append(mc.ens_agreement)
        colors.append(
            GR if mc.ens_agreement >= 0.80
            else (YL if mc.ens_agreement >= ENSEMBLE_AGREEMENT_FLOOR
                  else RD))

        bars = ax.barh(labels, vals, color=colors,
                       alpha=0.85, height=0.55)
        ax.set_xlim(0, 1)
        ax.axvline(0.5, color=GY, lw=0.9, ls="--", alpha=0.6)
        for bar, v, lbl in zip(bars, vals, labels):
            if "R/R" in lbl and mc.trade_setup:
                disp = f"1:{mc.trade_setup.rr_ratio:.1f}"
            elif "EV" in lbl and mc.trade_setup:
                disp = f"{mc.trade_setup.expected_value:+.3f}%"
            elif "ToD" in lbl:
                disp = f"x{mc.tod_multiplier:.3f}"
            elif "Ensemble" in lbl:
                disp = f"{v:.3f}"
            else:
                disp = f"{v:.1%}"
            ax.text(
                min(v + 0.03, 0.90),
                bar.get_y() + bar.get_height() / 2,
                disp, va="center",
                fontsize=8.5, color=WH, fontweight="bold")

    def _param_box(
            self, ax, sym, dm, mc, gp, jp,
            vm, note, profile: AssetProfile,
            tod: TimeOfDayProfile):
        S0v  = dm.last(sym)
        s0s  = f"{S0v:.6f}" if S0v else "N/A"
        rate = dm.observed_tick_rate(sym)
        b    = tod._buckets.get(
            sym, [TodBucket()] * TOD_BUCKETS)[_tod_bucket()]
        info = [
            f"  | {REVERSE_MAP.get(sym,sym)} "
            f"| PRICE={s0s} "
            f"| TICKS={dm.n(sym):,} "
            f"| RATE={rate:.2f}t/s "
            f"| PROFILE={profile.name} "
            f"| ENGINE="
            f"{'VOL_REGIME+MC' if profile.vol_regime_enabled else ('DRIFT+MC' if profile.drift_signal_enabled else 'MC+SPIKE')}"
            f" | {'LIVE' if dm._connected else 'RECON'}"
            f" | SQEv4.3"
            f" | ToD_mult={b.multiplier:.3f}"
            f" | ToD_wr={b.win_rate:.0%}"
            + (f" | Ens={mc.ens_agreement:.3f}" if mc else "")
        ]
        if gp and gp.n_obs > 0:
            ann = vm.ann_vol(gp) * 100
            dev = vm.deviation(gp)
            info.append(
                f"  | GBM mu={gp.mu:.8f} "
                f"sig={gp.sigma:.8f} "
                f"ewma={gp.sigma_ewma:.8f} "
                f"ann={ann:.3f}% "
                f"KS={gp.ks_pvalue:.4f} "
                f"FQ={gp.fit_quality:.3f}"
                + (f" Adv={gp.advertised_vol*100:.0f}%"
                   f" Dev={dev:+.1f}%"
                   if gp.advertised_vol else ""))
        if jp and jp.n_obs > 0:
            et = (1/jp.lam_posterior
                  if jp.lam_posterior > 1e-10
                  else float("inf"))
            info.append(
                f"  | JUMP "
                f"lam_mle={jp.lam:.6f} "
                f"lam_post={jp.lam_posterior:.6f}"
                f"(~1/{et:.0f}t) "
                f"nj={jp.n_jumps} "
                f"dev={jp.lam_deviation:+.1f}% "
                f"hz={jp.hazard_intensity:.0%} "
                f"sign={'UP' if jp.jump_sign > 0 else 'DN'}"
                f" p60={jp.spike_mag_p60*100:.3f}%"
                f" p75={jp.spike_mag_p75*100:.3f}%"
                f" cd={jp.counter_drift:.8f}")
        if mc:
            rng  = (mc.p95[-1] - mc.p5[-1]) / max(mc.S0, 1e-10) * 100
            ts_s = ""
            if mc.trade_setup:
                ts = mc.trade_setup
                ts_s = (
                    f" | {ts.direction} "
                    f"RR1:{ts.rr_ratio:.2f} "
                    f"P={ts.prob_target:.2%} "
                    f"EV={ts.expected_value:+.4f}%"
                    f" [{ts.signal_strength}]"
                    f" src={ts.signal_source}"
                    f" tod=x{ts.tod_multiplier:.3f}")
            vr_s = ""
            if mc.vol_regime:
                vr = mc.vol_regime
                vr_s = (
                    f" | vr={vr.signal}"
                    f" z={vr.normalized_momentum:+.3f}"
                    f" k={vr.k_adaptive:.3f}"
                    f" dev={vr.vol_deviation:+.3f}")
            drift_s = ""
            if mc.drift_signal:
                ds = mc.drift_signal
                drift_s = (
                    f" | drift={ds.direction}"
                    f" t={ds.tstat:+.3f}"
                    f" p={ds.pvalue:.4f}"
                    f" mom={ds.ewma_momentum:+.3f}"
                    f" win={ds.window_used}")
            info.append(
                f"  | MC({mc.horizon}t) "
                f"med={mc.p50[-1]:.6f} "
                f"CI={mc.ci_width_pct*100:.4f}% "
                f"90w={rng:.4f}% "
                f"edge={mc.edge_score:.4f} "
                f"sig={mc.signal} "
                f"ens={mc.ens_agreement:.3f}"
                f"{ts_s}{vr_s}{drift_s}")
        info += [
            f"  | {note}",
            f"  | Gates v4.3: "
            f"P>={MIN_PROB_TARGET:.0%} EV>0 "
            f"RR>={MIN_RR_GLOBAL:.1f} "
            f"Move>={MIN_TARGET_MOVE*100:.2f}% "
            f"Ens>={ENSEMBLE_AGREEMENT_FLOOR:.2f} "
            f"STRONG/MODERATE | "
            f"EmpCap p{EMPIRICAL_CAP_PERCENTILE}x{EMPIRICAL_CAP_MULTIPLIER} | "
            f"ToD [{TOD_MULT_MIN},{TOD_MULT_MAX}]",
            "  | No model guarantees profit. Max 1-2% risk.",
        ]
        ax.text(
            0.004, 0.95, "\n".join(info),
            transform=ax.transAxes,
            fontsize=6.8, va="top", color=WH,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.45",
                      fc=AX, ec=GY, alpha=0.93))

# ─────────────────────────────────────────────────────────────────────────────
# ALERT ENGINE v4.3
# ─────────────────────────────────────────────────────────────────────────────
class AlertEngine:
    def __init__(self, mce: MCEngine, pm: PatternMemory, pe: PersistenceEngine):
        self.mce  = mce
        self.pm   = pm
        self.pe   = pe
        self._last: Dict[str, float] = pe.get_alert_last()

    def _save_last(self):
        self.pe.save_alert_last(self._last)

    def check_trade(
            self, sym: str,
            dm: 'DataManager',
            horizon: int,
            horizon_label: str,
    ) -> Optional[Tuple[float, str, MCResult]]:
        profile = effective_profile(sym)
        now     = time.time()

        if now - self._last.get(sym, 0) < profile.alert_cooldown:
            return None
        if dm.n(sym) < profile.alert_min_ticks:
            return None

        r = self.mce.run(sym, dm, horizon, horizon_label, self.pm, n=5000)
        if r is None:
            return None

        # VOL: vol regime must be non-neutral
        if profile.vol_regime_enabled:
            if r.vol_regime is None:
                log.debug(f"Alert blocked {sym}: no vol_regime computed")
                return None
            if r.vol_regime.signal == "neutral":
                log.debug(
                    f"Alert blocked {sym}: vol_regime NEUTRAL "
                    f"(z={r.vol_regime.normalized_momentum:+.3f} "
                    f"edge={r.vol_regime.edge_score:.3f} "
                    f"strength={r.vol_regime.signal_strength})")
                return None

        # JUMP/STEP: drift must be non-neutral
        elif profile.drift_signal_enabled:
            if r.drift_signal is None or r.drift_signal.direction == "neutral":
                log.debug(f"Alert blocked {sym}: drift NEUTRAL")
                return None

        if r.signal in ("neutral",):
            log.debug(f"Alert blocked {sym}: signal neutral")
            return None

        # FIX v4.3 PROBLEM 5: Ensemble gate in alert engine
        if r.ens_agreement < ENSEMBLE_AGREEMENT_FLOOR:
            log.debug(
                f"Alert blocked {sym}: "
                f"ensemble={r.ens_agreement:.3f} "
                f"< {ENSEMBLE_AGREEMENT_FLOOR}")
            return None

        if r.trade_setup is None:
            log.debug(f"Alert blocked {sym}: no trade setup")
            return None

        ts = r.trade_setup

        if ts.signal_strength not in profile.alert_strength:
            log.debug(
                f"Alert blocked {sym}: "
                f"strength={ts.signal_strength}")
            return None

        if ts.rr_ratio < max(profile.alert_min_rr, MIN_RR_GLOBAL):
            log.debug(f"Alert blocked {sym}: RR={ts.rr_ratio:.2f}")
            return None

        if r.edge_score < profile.alert_edge_min:
            log.debug(f"Alert blocked {sym}: edge={r.edge_score:.3f}")
            return None

        if ts.prob_target < MIN_PROB_TARGET:
            log.debug(f"Alert blocked {sym}: P={ts.prob_target:.2%}")
            return None

        if ts.expected_value <= 0:
            log.debug(f"Alert blocked {sym}: EV={ts.expected_value:+.4f}%")
            return None

        if ts.edge_pct <= MIN_EDGE_PCT:
            log.debug(f"Alert blocked {sym}: edge_pct={ts.edge_pct:+.2f}%")
            return None

        if ts.target_pct < MIN_TARGET_MOVE * 100:
            log.debug(f"Alert blocked {sym}: target={ts.target_pct:.3f}%")
            return None

        self._last[sym] = now
        self._save_last()
        log.info(
            f"Alert PASSED {sym}: {ts.direction} "
            f"[{ts.signal_strength}] [{ts.signal_source}] "
            f"P={ts.prob_target:.2%} "
            f"EV={ts.expected_value:+.4f}% "
            f"RR=1:{ts.rr_ratio:.2f} "
            f"Move={ts.target_pct:.3f}% "
            f"Ens={r.ens_agreement:.3f} "
            f"ToD=x{ts.tod_multiplier:.3f}")
        return r.edge_score, r.signal, r

    def check_spike(
            self, sym: str,
            dm: 'DataManager',
            jm: JumpDiffusionModel,
            se: SpikeEngine,
    ) -> Optional[SpikeAlert]:
        profile   = effective_profile(sym)
        now       = time.time()
        spike_key = f"spike_{sym}"

        if not profile.spike_enabled:
            return None
        if now - self._last.get(spike_key, 0) < profile.alert_cooldown:
            return None
        if dm.n(sym) < MIN_TICKS:
            return None

        lr       = dm.log_returns(sym)
        jp       = jm.fit_unbiased(sym, lr)
        tr       = dm.observed_tick_rate(sym)
        tod_mult = self.mce.tod.multiplier(sym)
        sa       = se.assess(sym, jp, 300, tr, tod_mult)
        if sa is not None:
            self._last[spike_key] = now
            self._save_last()
        return sa

# ─────────────────────────────────────────────────────────────────────────────
# KEYBOARD HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _tf_kb(sym: str, prefix: str = "tf") -> List[List[InlineKeyboardButton]]:
    return [[
        InlineKeyboardButton("1m",  callback_data=f"{prefix}:{sym}:1m"),
        InlineKeyboardButton("5m",  callback_data=f"{prefix}:{sym}:5m"),
        InlineKeyboardButton("15m", callback_data=f"{prefix}:{sym}:15m"),
        InlineKeyboardButton("30m", callback_data=f"{prefix}:{sym}:30m"),
        InlineKeyboardButton("1h",  callback_data=f"{prefix}:{sym}:1h"),
    ]]

def _tick_kb(sym: str, prefix: str = "tk") -> List[List[InlineKeyboardButton]]:
    return [[
        InlineKeyboardButton("50t",  callback_data=f"{prefix}:{sym}:50t"),
        InlineKeyboardButton("100t", callback_data=f"{prefix}:{sym}:100t"),
        InlineKeyboardButton("300t", callback_data=f"{prefix}:{sym}:300t"),
        InlineKeyboardButton("600t", callback_data=f"{prefix}:{sym}:600t"),
    ]]

def _action_kb(
        sym: str, tf_key: str, cat: str,
) -> List[List[InlineKeyboardButton]]:
    rows = [
        _tf_kb(sym, "tf")[0],
        _tick_kb(sym, "tk")[0],
        [
            InlineKeyboardButton(
                "🔄 Refresh",
                callback_data=f"tf:{sym}:{tf_key}"),
            InlineKeyboardButton(
                "🎲 Prob Table",
                callback_data=f"prob_tf:{sym}:{tf_key}"),
            InlineKeyboardButton(
                "⭐ Watch",
                callback_data=f"addwatch:{sym}"),
        ],
    ]
    if cat in ("boom", "crash"):
        rows.append([
            InlineKeyboardButton(
                "⚡ Spike Check",
                callback_data=f"spike:{sym}"),
            InlineKeyboardButton(
                "📈 Results",
                callback_data=f"results:{sym}"),
        ])
    else:
        rows.append([
            InlineKeyboardButton(
                "📊 Drift/Regime",
                callback_data=f"drift:{sym}"),
            InlineKeyboardButton(
                "📈 Results",
                callback_data=f"results:{sym}"),
            InlineKeyboardButton(
                "🕐 ToD",
                callback_data=f"tod:{sym}"),
        ])
    return rows

# ─────────────────────────────────────────────────────────────────────────────
# TELEGRAM BOT v4.3
# ─────────────────────────────────────────────────────────────────────────────
class SQEBotV43:
    def __init__(self):
        self.pe  = PersistenceEngine()
        self.tod = TimeOfDayProfile()
        tod_data = self.pe.get_tod_data()
        if tod_data:
            try:
                self.tod.from_dict(tod_data)
                log.info("ToD profiles loaded from persistence.")
            except Exception as e:
                log.warning(f"ToD load: {e}")

        self.dm   = DataManager()
        self.vm   = VolatilityModel()
        self.jm   = JumpDiffusionModel()
        self.mce  = MCEngine(self.tod)
        self.cg   = ChartGen()
        self.pm   = PatternMemory(tod_profile=self.tod)
        self.ae   = AlertEngine(self.mce, self.pm, self.pe)
        self.narr = NarrativeEngine(self.vm, self.jm)
        self.se   = SpikeEngine(self.jm)
        self.de   = DriftEngine()
        self.vre  = VolatilityRegimeEngine(self.vm)
        self.tsb  = TradeSetupBuilder()
        self.cr   = ConflictResolver()

        self._chats: Set[int]          = self.pe.get_chats()
        self._states: Dict[int, UserState] = self.pe.get_user_states()
        self._ws_task = self._al_task  = None
        self.dm.add_tick_cb(self._on_tick)
        self._bot_ref = None

    def _state(self, chat_id: int) -> UserState:
        if chat_id not in self._states:
            self._states[chat_id] = UserState()
        return self._states[chat_id]

    def _save_states(self):
        self.pe.save_user_states(self._states)
        self.pe.save_chats(self._chats)
        self.pe.save_tod_data(self.tod.to_dict())

    def _track(self, update: Update):
        if update.effective_chat:
            cid = update.effective_chat.id
            self._chats.add(cid)
            self.pe.save_chats(self._chats)

    async def _on_tick(self, sym: str, price: float, ts: float):
        self.pm.resolve(sym, price,
                        bot=self._bot_ref,
                        chats=self._chats)
        rate = self.dm.observed_tick_rate(sym)
        self.tod.update(sym, True, 0.0, 0.0, 0.0, rate)

    async def cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        self._track(update)
        cid   = update.effective_chat.id
        state = self._state(cid)
        conn  = self.dm._connected
        total = sum(self.dm.n(s) for s in ALL_SYMBOLS.values())
        kb = [
            [InlineKeyboardButton("📈 Volatility", callback_data="menu:vol"),
             InlineKeyboardButton("💥 Boom",       callback_data="menu:boom"),
             InlineKeyboardButton("📉 Crash",      callback_data="menu:crash")],
            [InlineKeyboardButton("🦘 Jump",       callback_data="menu:jump"),
             InlineKeyboardButton("📊 Step",       callback_data="menu:step"),
             InlineKeyboardButton("⭐ Watchlist",  callback_data="menu:watchlist")],
            [InlineKeyboardButton("⚡ Quick Analyze",  callback_data="menu:quick"),
             InlineKeyboardButton("🚨 Spike Scanner",  callback_data="menu:spike_scan"),
             InlineKeyboardButton("⚙️ Settings",       callback_data="menu:settings")],
            [InlineKeyboardButton("📡 Status",     callback_data="menu:status"),
             InlineKeyboardButton("📊 Patterns",   callback_data="menu:patterns"),
             InlineKeyboardButton("🕐 ToD Summary",callback_data="menu:tod")],
            [InlineKeyboardButton(
                "🔔 Alerts ON" if state.alerts_enabled else "🔕 Alerts OFF",
                callback_data="toggle:alerts")],
        ]
        st_icon = "🟢" if conn else "🔴"
        await update.message.reply_text(
            "*SYNTHETIC QUANT ELITE v4.3*\n"
            "Calibrated Precision Edition\n\n"
            f"{st_icon} `{'Live' if conn else 'Reconnecting'}`\n"
            f"📊 `{total:,} ticks | {len(ALL_SYMBOLS)} instruments`\n\n"
            "*Signal Quality Gates (v4.3):*\n"
            f"  ✅ P(target) >= {MIN_PROB_TARGET:.0%} (cap 72%)\n"
            "  ✅ EV > 0 (per-asset minimum)\n"
            f"  ✅ RR >= {MIN_RR_GLOBAL:.1f} (dynamic, cap 4.0)\n"
            f"  ✅ Move >= {MIN_TARGET_MOVE*100:.2f}%\n"
            f"  ✅ Ensemble >= {ENSEMBLE_AGREEMENT_FLOOR:.2f}\n"
            "  ✅ STRONG or MODERATE only\n"
            "  ✅ Vol Regime confirmed (VOL)\n"
            "  ✅ Drift confirmed (JUMP/STEP)\n"
            f"  ✅ Empirical target cap p{EMPIRICAL_CAP_PERCENTILE}x{EMPIRICAL_CAP_MULTIPLIER}\n\n"
            "*Engines (v4.3):*\n"
            "  VOL:   Vol Regime+MC [horizon-aware]\n"
            "  BOOM/CRASH: Bayesian Spike PRIMARY\n"
            "  JUMP:  Drift+JD [horizon-aware]\n"
            "  STEP:  Drift+MC [horizon-aware]\n"
            "  ALL:   ToD ±3% | Ensemble gate\n\n"
            f"*Alerts:* `{'ON' if state.alerts_enabled else 'OFF'}`\n"
            "Select a category:",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup(kb))

    async def cmd_predict(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        self._track(update)
        if not ctx.args:
            await update.message.reply_text(
                "Usage: `/predict <symbol>`\ne.g. `/predict V75`",
                parse_mode=ParseMode.MARKDOWN)
            return
        sym = _resolve(" ".join(ctx.args))
        if not sym:
            await update.message.reply_text("Unknown symbol.")
            return
        cid     = update.effective_chat.id
        profile = effective_profile(sym)
        self._state(cid).pending_sym = sym
        cat = _cat(sym)
        kb  = _tf_kb(sym, "tf") + _tick_kb(sym, "tk")
        if cat in ("boom", "crash"):
            kb.append([InlineKeyboardButton(
                "⚡ Spike Check",
                callback_data=f"spike:{sym}")])
        engine_str = (
            "VOL_REGIME+MC" if profile.vol_regime_enabled
            else ("DRIFT+MC" if profile.drift_signal_enabled
                  else "MC+SPIKE"))
        tod_m = self.tod.multiplier(sym)
        await update.message.reply_text(
            f"*{_friendly(sym)}* [{profile.name}] ({cat.upper()})\n"
            f"Engine: *{engine_str}* | ToD: x{tod_m:.3f}\n"
            f"Select timeframe:",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup(kb))

    async def cmd_analyze(self, u, c):
        await self.cmd_predict(u, c)

    async def cmd_spike(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        self._track(update)
        if not ctx.args:
            await update.message.reply_text(
                "Usage: `/spike <symbol>`",
                parse_mode=ParseMode.MARKDOWN)
            return
        sym = _resolve(" ".join(ctx.args))
        if not sym:
            await update.message.reply_text("Unknown symbol.")
            return
        cat = _cat(sym)
        if cat not in ("boom", "crash"):
            await update.message.reply_text(
                f"Spike detection: Boom/Crash only.\n"
                f"`{_friendly(sym)}` is {cat.upper()}.\n"
                f"Use `/predict` for drift/regime signals.",
                parse_mode=ParseMode.MARKDOWN)
            return
        await self._do_spike_check(update.message.chat_id, ctx, sym)

    async def cmd_drift(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        self._track(update)
        if not ctx.args:
            await update.message.reply_text(
                "Usage: `/drift <symbol>`\ne.g. `/drift V75`",
                parse_mode=ParseMode.MARKDOWN)
            return
        sym = _resolve(" ".join(ctx.args))
        if not sym:
            await update.message.reply_text("Unknown symbol.")
            return
        await self._do_drift_report(update.message.chat_id, ctx, sym)

    async def cmd_tod(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        self._track(update)
        if not ctx.args:
            await update.message.reply_text(
                "Usage: `/tod <symbol>`\ne.g. `/tod V75`",
                parse_mode=ParseMode.MARKDOWN)
            return
        sym = _resolve(" ".join(ctx.args))
        if not sym:
            await update.message.reply_text("Unknown symbol.")
            return
        await self._do_tod_report(update.message.chat_id, ctx, sym)

    async def cmd_prob(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        self._track(update)
        if not ctx.args:
            await update.message.reply_text(
                "Usage: `/prob <symbol>`",
                parse_mode=ParseMode.MARKDOWN)
            return
        sym = _resolve(" ".join(ctx.args))
        if not sym:
            await update.message.reply_text("Unknown symbol.")
            return
        cid = update.effective_chat.id
        self._state(cid).pending_sym = sym
        kb  = _tf_kb(sym, "prob_tf") + _tick_kb(sym, "prob_tk")
        await update.message.reply_text(
            f"*{_friendly(sym)}* — Select timeframe for prob table:",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup(kb))

    async def cmd_parameters(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        self._track(update)
        if not ctx.args:
            await update.message.reply_text(
                "Usage: `/parameters <symbol>`",
                parse_mode=ParseMode.MARKDOWN)
            return
        sym = _resolve(" ".join(ctx.args))
        if not sym:
            await update.message.reply_text("Unknown symbol.")
            return
        await self._send_parameters(update.message.chat_id, ctx, sym)

    async def cmd_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        self._track(update)
        lines = [
            "*Data Feed — SQE v4.3*\n",
            f"WS: {'🟢 Live' if self.dm._connected else '🔴 Reconnecting'}\n",
            f"{'Symbol':<14} {'Ticks':>7} {'Rate':>7} {'Profile':>8} {'Engine':>12}",
            "─" * 58,
        ]
        for fn, ds in ALL_SYMBOLS.items():
            n       = self.dm.n(ds)
            rate    = self.dm.observed_tick_rate(ds)
            profile = effective_profile(ds)
            engine  = (
                "VOL_REGIME" if profile.vol_regime_enabled
                else ("DRIFT+MC" if profile.drift_signal_enabled
                      else "MC+SPIKE"))
            tod_m = self.tod.multiplier(ds)
            ok    = "✅" if n >= MIN_TICKS else f"({n})"
            lines.append(
                f"{fn:<14} {n:>7,} {rate:>6.2f}t/s "
                f"{profile.name:>8} {engine:>12} "
                f"x{tod_m:.2f} {ok}")
        await update.message.reply_text(
            "\n".join(lines),
            parse_mode=ParseMode.MARKDOWN)

    async def cmd_results(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        self._track(update)
        if not ctx.args:
            await update.message.reply_text(
                "Usage: `/results <symbol>`",
                parse_mode=ParseMode.MARKDOWN)
            return
        sym = _resolve(" ".join(ctx.args))
        if not sym:
            await update.message.reply_text("Unknown symbol.")
            return
        await self._send_results(update.message.chat_id, ctx, sym)

    async def cmd_watch(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        self._track(update)
        cid = update.effective_chat.id
        if not ctx.args:
            await update.message.reply_text(
                "Usage: `/watch <symbol>`",
                parse_mode=ParseMode.MARKDOWN)
            return
        sym = _resolve(" ".join(ctx.args))
        if not sym:
            await update.message.reply_text("Unknown symbol.")
            return
        st = self._state(cid)
        if sym not in st.watchlist:
            st.watchlist.append(sym)
        self._save_states()
        await update.message.reply_text(
            f"⭐ *{_friendly(sym)}* added.\n"
            f"Watchlist: {', '.join(_friendly(s) for s in st.watchlist)}",
            parse_mode=ParseMode.MARKDOWN)

    async def cmd_unwatch(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        self._track(update)
        cid = update.effective_chat.id
        if not ctx.args:
            await update.message.reply_text(
                "Usage: `/unwatch <symbol>`",
                parse_mode=ParseMode.MARKDOWN)
            return
        sym = _resolve(" ".join(ctx.args))
        st  = self._state(cid)
        if sym in st.watchlist:
            st.watchlist.remove(sym)
        self._save_states()
        await update.message.reply_text(
            f"Removed. Watchlist: "
            f"{', '.join(_friendly(s) for s in st.watchlist) or 'empty'}",
            parse_mode=ParseMode.MARKDOWN)

    async def cmd_quick_sym(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        self._track(update)
        cid = update.effective_chat.id
        cmd = update.message.text.lstrip("/").split("@")[0].upper()
        sym_map = {
            "V10": "R_10", "V25": "R_25", "V50": "R_50",
            "V75": "R_75", "V100": "R_100", "V250": "R_250",
            "BOOM300": "BOOM300N",  "BOOM500": "BOOM500",
            "BOOM600": "BOOM600N",  "BOOM900": "BOOM900",
            "BOOM1000": "BOOM1000",
            "CRASH300": "CRASH300N", "CRASH500": "CRASH500",
            "CRASH600": "CRASH600N", "CRASH900": "CRASH900",
            "CRASH1000": "CRASH1000",
            "STEP": "stpRNG", "STEPINDEX": "stpRNG",
            "JUMP10": "JD10", "JUMP25": "JD25", "JUMP50": "JD50",
            "JUMP75": "JD75", "JUMP100": "JD100",
        }
        sym = sym_map.get(cmd)
        if not sym:
            await update.message.reply_text("Unknown quick command.")
            return
        self._state(cid).pending_sym = sym
        cat     = _cat(sym)
        profile = effective_profile(sym)
        kb      = _tf_kb(sym, "tf") + _tick_kb(sym, "tk")
        if cat in ("boom", "crash"):
            kb.append([InlineKeyboardButton(
                "⚡ Spike Check",
                callback_data=f"spike:{sym}")])
        engine_str = (
            "VOL_REGIME+MC" if profile.vol_regime_enabled
            else ("DRIFT+MC" if profile.drift_signal_enabled
                  else "MC+SPIKE"))
        tod_m = self.tod.multiplier(sym)
        await update.message.reply_text(
            f"*{_friendly(sym)}* [{profile.name}] "
            f"Engine: *{engine_str}* | ToD: x{tod_m:.3f}\n"
            f"Select timeframe:",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup(kb))

    async def cb_query(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        q   = update.callback_query
        await q.answer()
        cid = q.message.chat_id
        self._chats.add(cid)
        self.pe.save_chats(self._chats)
        d = q.data
        try:
            if d.startswith("menu:"):
                await self._handle_menu(q, d[5:], cid, ctx)
            elif d.startswith("sym:"):
                sym     = d[4:]
                cat     = _cat(sym)
                profile = effective_profile(sym)
                self._state(cid).pending_sym = sym
                kb = _tf_kb(sym, "tf") + _tick_kb(sym, "tk")
                if cat in ("boom", "crash"):
                    kb.append([InlineKeyboardButton(
                        "⚡ Spike Check",
                        callback_data=f"spike:{sym}")])
                engine_str = (
                    "VOL_REGIME+MC" if profile.vol_regime_enabled
                    else ("DRIFT+MC" if profile.drift_signal_enabled
                          else "MC+SPIKE"))
                tod_m = self.tod.multiplier(sym)
                await q.message.reply_text(
                    f"*{_friendly(sym)}* [{profile.name}] "
                    f"Engine: *{engine_str}* | ToD: x{tod_m:.3f}\n"
                    f"Select timeframe:",
                    parse_mode=ParseMode.MARKDOWN,
                    reply_markup=InlineKeyboardMarkup(kb))
            elif d.startswith("tf:"):
                _, sym, tf_key = d.split(":")
                h, hl = self.dm.ticks_for_timeframe(sym, tf_key)
                await q.message.reply_text(
                    f"Analysing *{_friendly(sym)}* — *{hl}* ({h}t) ...",
                    parse_mode=ParseMode.MARKDOWN)
                await self._do_analysis(cid, ctx, sym, h, hl, tf_key)
            elif d.startswith("tk:"):
                parts     = d.split(":")
                sym, hkey = parts[1], parts[2]
                h  = HORIZONS.get(hkey, 300)
                hl = HORIZON_LABELS.get(hkey, "300 Ticks")
                await q.message.reply_text(
                    f"Analysing *{_friendly(sym)}* — *{hl}* ...",
                    parse_mode=ParseMode.MARKDOWN)
                await self._do_analysis(cid, ctx, sym, h, hl)
            elif d.startswith("prob_tf:"):
                _, sym, tf_key = d.split(":")
                h, hl = self.dm.ticks_for_timeframe(sym, tf_key)
                await q.message.reply_text(
                    f"Building prob table *{_friendly(sym)}* ...",
                    parse_mode=ParseMode.MARKDOWN)
                await self._send_prob_table(cid, ctx, sym, h, hl)
            elif d.startswith("prob_tk:"):
                parts     = d.split(":")
                sym, hkey = parts[1], parts[2]
                h  = HORIZONS.get(hkey, 300)
                hl = HORIZON_LABELS.get(hkey, "300 Ticks")
                await q.message.reply_text(
                    f"Building prob table *{_friendly(sym)}* ...",
                    parse_mode=ParseMode.MARKDOWN)
                await self._send_prob_table(cid, ctx, sym, h, hl)
            elif d.startswith("spike:"):
                sym = d[6:]
                await self._do_spike_check(cid, ctx, sym)
            elif d.startswith("drift:"):
                sym = d[6:]
                await self._do_drift_report(cid, ctx, sym)
            elif d.startswith("tod:"):
                sym = d[4:]
                await self._do_tod_report(cid, ctx, sym)
            elif d.startswith("results:"):
                sym = d[8:]
                await self._send_results(cid, ctx, sym)
            elif d.startswith("addwatch:"):
                sym = d[9:]
                st  = self._state(cid)
                if sym not in st.watchlist:
                    st.watchlist.append(sym)
                self._save_states()
                await q.message.reply_text(
                    f"⭐ *{_friendly(sym)}* added!",
                    parse_mode=ParseMode.MARKDOWN)
            elif d.startswith("rmwatch:"):
                sym = d[8:]
                st  = self._state(cid)
                if sym in st.watchlist:
                    st.watchlist.remove(sym)
                self._save_states()
                await q.message.reply_text(
                    f"Removed *{_friendly(sym)}*.",
                    parse_mode=ParseMode.MARKDOWN)
            elif d.startswith("risk:"):
                pct = float(d[5:])
                self._state(cid).risk_pct = pct
                self._save_states()
                await q.message.reply_text(
                    f"Risk: *{pct:.1f}%*",
                    parse_mode=ParseMode.MARKDOWN)
            elif d == "toggle:alerts":
                st = self._state(cid)
                st.alerts_enabled = not st.alerts_enabled
                self._save_states()
                s = "ENABLED 🔔" if st.alerts_enabled else "DISABLED 🔕"
                await q.message.reply_text(
                    f"Alerts *{s}*.\n"
                    + (f"Gates v4.3:\n"
                       f"  P>={MIN_PROB_TARGET:.0%} EV>0\n"
                       f"  RR>={MIN_RR_GLOBAL:.1f} (dynamic)\n"
                       f"  Ensemble>={ENSEMBLE_AGREEMENT_FLOOR:.2f}\n"
                       f"  EmpCap p{EMPIRICAL_CAP_PERCENTILE}x{EMPIRICAL_CAP_MULTIPLIER}\n"
                       f"  STRONG/MODERATE only"
                       if st.alerts_enabled else "Tap to re-enable."),
                    parse_mode=ParseMode.MARKDOWN)
            elif d == "menu:spike_scan":
                await self._do_spike_scan(cid, ctx)
        except Exception as e:
            log.error(f"cb_query error: {e}")
            try:
                await q.message.reply_text(f"Error: {str(e)[:100]}")
            except Exception:
                pass

    async def _handle_menu(self, q, cat: str, cid: int, ctx):
        menus = {
            "vol":   (VOLATILITY_SYMBOLS, "📈 Volatility Indices"),
            "boom":  (BOOM_SYMBOLS,       "💥 Boom Indices"),
            "crash": (CRASH_SYMBOLS,      "📉 Crash Indices"),
            "jump":  (JUMP_SYMBOLS,       "🦘 Jump Indices"),
            "step":  (STEP_SYMBOLS,       "📊 Step Index"),
        }
        if cat in menus:
            syms, title = menus[cat]
            rows = []
            for fn, ds in syms.items():
                n     = self.dm.n(ds)
                ok    = "✅" if n >= MIN_TICKS else f"({n})"
                tod_m = self.tod.multiplier(ds)
                c     = _cat(ds)
                row   = [
                    InlineKeyboardButton(
                        f"{fn} {ok} ToD:{tod_m:.2f}",
                        callback_data=f"sym:{ds}"),
                    InlineKeyboardButton("⭐", callback_data=f"addwatch:{ds}"),
                ]
                if c in ("boom", "crash"):
                    row.append(InlineKeyboardButton(
                        "⚡", callback_data=f"spike:{ds}"))
                else:
                    row.append(InlineKeyboardButton(
                        "📊", callback_data=f"drift:{ds}"))
                rows.append(row)
            await q.message.reply_text(
                f"*{title}*\n"
                f"Tap analyse | ⭐ watch | "
                f"{'⚡ spike' if cat in ('boom','crash') else '📊 regime/drift'}",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=InlineKeyboardMarkup(rows))
        elif cat == "watchlist":
            await self._show_watchlist(q, cid, ctx)
        elif cat == "quick":
            await self._show_quick(q)
        elif cat == "spike_scan":
            await self._do_spike_scan(cid, ctx)
        elif cat == "tod":
            await self._show_tod_summary(q)
        elif cat == "status":
            lines = ["*Feed Status — SQE v4.3*\n"]
            for fn, ds in ALL_SYMBOLS.items():
                n  = self.dm.n(ds)
                ok = "✅" if n >= MIN_TICKS else f"({n})"
                p  = effective_profile(ds)
                tm = self.tod.multiplier(ds)
                lines.append(
                    f"{fn:<14} {n:>7,} [{p.name}] ToD:x{tm:.2f} {ok}")
            await q.message.reply_text(
                "\n".join(lines),
                parse_mode=ParseMode.MARKDOWN)
        elif cat == "patterns":
            await self._show_patterns(q)
        elif cat == "settings":
            await self._show_settings(q, cid)

    async def _show_tod_summary(self, q):
        lines = [
            "*🕐 Time-of-Day — SQE v4.3*\n",
            f"Current bucket: UTC "
            f"{_tod_bucket()//2:02d}:"
            f"{'30' if _tod_bucket()%2 else '00'}\n"
        ]
        for fn, ds in list(ALL_SYMBOLS.items())[:15]:
            b    = self.tod._buckets.get(
                ds, [TodBucket()]*TOD_BUCKETS)[_tod_bucket()]
            icon = ("🟢" if b.multiplier >= 1.05
                    else ("🔴" if b.multiplier <= 0.95 else "🟡"))
            lines.append(
                f"{icon} `{fn:<14}` "
                f"x{b.multiplier:.3f} "
                f"wr={b.win_rate:.0%} "
                f"n={b.count:.0f}")
        await q.message.reply_text(
            "\n".join(lines),
            parse_mode=ParseMode.MARKDOWN)

    async def _show_watchlist(self, q, cid, ctx):
        st = self._state(cid)
        if not st.watchlist:
            await q.message.reply_text(
                "Watchlist empty.",
                parse_mode=ParseMode.MARKDOWN)
            return
        rows = []
        for ds in st.watchlist:
            fn    = _friendly(ds)
            n     = self.dm.n(ds)
            ok    = "✅" if n >= MIN_TICKS else f"({n})"
            cat   = _cat(ds)
            tod_m = self.tod.multiplier(ds)
            row   = [
                InlineKeyboardButton(
                    f"{fn} {ok} ToD:{tod_m:.2f}",
                    callback_data=f"sym:{ds}"),
                InlineKeyboardButton("🗑", callback_data=f"rmwatch:{ds}"),
            ]
            if cat in ("boom", "crash"):
                row.append(InlineKeyboardButton(
                    "⚡", callback_data=f"spike:{ds}"))
            else:
                row.append(InlineKeyboardButton(
                    "📊", callback_data=f"drift:{ds}"))
            rows.append(row)
        await q.message.reply_text(
            "⭐ *My Watchlist* (with ToD multipliers)",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup(rows))

    async def _show_quick(self, q):
        best = sorted(
            ALL_SYMBOLS.items(),
            key=lambda kv: self.dm.n(kv[1]),
            reverse=True)[:10]
        rows = [[InlineKeyboardButton(
            f"{k} ({self.dm.n(v):,}t) ToD:x{self.tod.multiplier(v):.2f}",
            callback_data=f"sym:{v}")]
            for k, v in best]
        await q.message.reply_text(
            "⚡ *Quick Analyze* — Most data:",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup(rows))

    async def _show_patterns(self, q):
        lines = ["📊 *Pattern Memory v4.3*\n"]
        for fn, ds in ALL_SYMBOLS.items():
            n, wr, ae = self.pm.stats(ds)
            if n > 0:
                e      = ("🟢" if wr >= 0.55 else ("🔴" if wr < 0.45 else "🟡"))
                wins   = sum(1 for r in self.pm.records
                             if r.symbol == ds and r.resolved and r.correct)
                losses = n - wins
                tod_m  = self.tod.multiplier(ds)
                lines.append(
                    f"{e} `{fn}`: {wr:.0%} | "
                    f"W:{wins} L:{losses} | "
                    f"edge {ae:.2f} | ToD:x{tod_m:.2f}")
        if len(lines) == 1:
            lines.append("Building...")
        await q.message.reply_text(
            "\n".join(lines),
            parse_mode=ParseMode.MARKDOWN)

    async def _show_settings(self, q, cid: int):
        st = self._state(cid)
        kb = [
            [InlineKeyboardButton("Risk 0.5%", callback_data="risk:0.5"),
             InlineKeyboardButton("Risk 1%",   callback_data="risk:1.0"),
             InlineKeyboardButton("Risk 2%",   callback_data="risk:2.0")],
            [InlineKeyboardButton(
                "🔕 Disable Alerts" if st.alerts_enabled else "🔔 Enable Alerts",
                callback_data="toggle:alerts")],
        ]
        await q.message.reply_text(
            f"⚙️ *Settings — SQE v4.3*\n\n"
            f"Risk: *{st.risk_pct:.1f}%*\n"
            f"Alerts: *{'ON 🔔' if st.alerts_enabled else 'OFF 🔕'}*\n\n"
            f"*Signal Quality Gates v4.3:*\n"
            f"  P(target) >= {MIN_PROB_TARGET:.0%} (cap 72%)\n"
            f"  EV > 0 (per-asset minimum)\n"
            f"  RR >= {MIN_RR_GLOBAL:.1f} (dynamic, cap 4.0)\n"
            f"  Move >= {MIN_TARGET_MOVE*100:.2f}%\n"
            f"  Ensemble >= {ENSEMBLE_AGREEMENT_FLOOR:.2f}\n"
            f"  STRONG or MODERATE only\n"
            f"  Vol Regime confirmed (VOL)\n"
            f"  Drift confirmed (JUMP/STEP)\n"
            f"  Empirical cap p{EMPIRICAL_CAP_PERCENTILE}x{EMPIRICAL_CAP_MULTIPLIER}\n\n"
            f"*Engines (v4.3):*\n"
            f"  VOL:   VOL_REGIME+MC [horizon-aware]\n"
            f"  BOOM:  BAYESIAN SPIKE PRIMARY\n"
            f"  CRASH: BAYESIAN SPIKE PRIMARY\n"
            f"  JUMP:  DRIFT+JD [horizon-aware]\n"
            f"  STEP:  DRIFT+MC [horizon-aware]\n\n"
            f"Persist: `{PERSIST_FILE}`\n"
            f"Patterns: `{PATTERN_FILE}`",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup(kb))

    async def _do_analysis(
            self, cid: int, ctx,
            sym: str, horizon: int,
            hlabel: str, tf_key: Optional[str] = None):
        fname   = _friendly(sym)
        n       = self.dm.n(sym)
        S0      = self.dm.last(sym)
        profile = effective_profile(sym)
        cat     = _cat(sym)

        if n < MIN_TICKS or S0 is None:
            await ctx.bot.send_message(
                cid,
                f"*{fname}* [{profile.name}]: "
                f"{n} ticks (need {MIN_TICKS}+). Retry in ~30s.",
                parse_mode=ParseMode.MARKDOWN)
            return

        lr = self.dm.log_returns(sym)
        gp = jp = None

        if cat in ("vol", "step"):
            gp = self.vm.fit(sym, lr)
        else:
            jp = self.jm.fit_unbiased(sym, lr)
            gp = GBMParams(
                mu=jp.mu, sigma=jp.sigma,
                sigma_ewma=jp.sigma,
                n_obs=jp.n_obs, fit_quality=0.80)

        loop = asyncio.get_event_loop()
        mc   = await loop.run_in_executor(
            None, self.mce.run,
            sym, self.dm, horizon, hlabel, self.pm, None, tf_key)

        if mc is None:
            await ctx.bot.send_message(
                cid,
                f"Analysis failed for {fname}. Retry.",
                parse_mode=ParseMode.MARKDOWN)
            return

        spike_alert = None
        if profile.spike_enabled and jp:
            tr          = self.dm.observed_tick_rate(sym)
            tod_mult    = self.tod.multiplier(sym)
            spike_alert = self.se.assess(sym, jp, horizon, tr, tod_mult)

        validated_trade, trade_valid, reason = self.cr.resolve(
            sym, mc, spike_alert, mc.trade_setup)
        mc.trade_setup = validated_trade if trade_valid else None

        if mc.signal not in ("neutral",):
            tup = mc.target_up
            tdn = mc.target_down
            if mc.trade_setup:
                tup = mc.trade_setup.target
                tdn = mc.trade_setup.invalidation
            self.pm.record(
                sym, mc.horizon, mc.signal,
                max(mc.prob_hit_up, mc.prob_hit_down),
                mc.edge_score, S0, tup, tdn)

        note = self.pm.note(sym, mc.horizon)
        st   = self._state(cid)

        img = await loop.run_in_executor(
            None, self.cg.make,
            fname, sym, self.dm,
            mc, gp, jp, self.vm, note, self.tod)

        cap    = self._caption(fname, sym, mc, gp, jp, note, st.risk_pct)
        used_tf = tf_key or "5m"
        kb      = _action_kb(sym, used_tf, cat)

        await _safe_send_photo(
            ctx.bot, cid, photo=img, caption=cap,
            reply_markup=InlineKeyboardMarkup(kb))

        full_text = self._full_analysis(
            fname, sym, mc, gp, jp, note, st.risk_pct, hlabel)
        if full_text:
            await _safe_send_message(ctx.bot, cid, full_text)

        if spike_alert:
            await _safe_send_message(
                ctx.bot, cid,
                self.se.format_standalone(spike_alert))

        if not trade_valid and reason == "spike_primary_conflict":
            await _safe_send_message(
                ctx.bot, cid,
                f"*Note:* Trade suppressed — "
                f"Bayesian spike direction takes priority.")

        self.pe.save_tod_data(self.tod.to_dict())

    async def _do_tod_report(self, cid: int, ctx, sym: str):
        fname   = _friendly(sym)
        profile = effective_profile(sym)
        self.tod._ensure(sym)
        buckets = self.tod._buckets[sym]
        cur_b   = _tod_bucket()

        lines = [
            f"*Time-of-Day — {fname}* [{profile.name}]\n",
            f"Current: UTC {cur_b//2:02d}:{'30' if cur_b%2 else '00'} "
            f"(bucket {cur_b})\n",
            f"*Current bucket:*",
            self.tod.summary(sym), "",
            f"*Best 5 (by multiplier):*",
        ]
        sorted_b = sorted(
            enumerate(buckets),
            key=lambda x: x[1].multiplier,
            reverse=True)[:5]
        for idx, b in sorted_b:
            h    = idx // 2
            mins = "30" if idx % 2 else "00"
            icon = "🟢" if b.multiplier >= 1.05 else "🟡"
            lines.append(
                f"{icon} {h:02d}:{mins} UTC | "
                f"x{b.multiplier:.3f} | "
                f"wr={b.win_rate:.0%} | "
                f"n={b.count:.0f}")

        lines += ["", "*Worst 3:*"]
        worst_b = sorted(
            enumerate(buckets),
            key=lambda x: x[1].multiplier)[:3]
        for idx, b in worst_b:
            h    = idx // 2
            mins = "30" if idx % 2 else "00"
            lines.append(
                f"🔴 {h:02d}:{mins} UTC | "
                f"x{b.multiplier:.3f} | "
                f"wr={b.win_rate:.0%} | "
                f"n={b.count:.0f}")

        lines += [
            "", "*How ToD works (v4.3):*",
            "  48 half-hour UTC buckets.",
            "  EWM updated on each resolved signal.",
            f"  Multiplier range: x{TOD_MULT_MIN:.2f}–x{TOD_MULT_MAX:.2f}.",
            "  Applied as ±3% max prob adjustment.",
            "  Neutral (x1.000) until 3+ signals.",
        ]

        kb = [[
            InlineKeyboardButton("📊 Full Analysis", callback_data=f"sym:{sym}"),
            InlineKeyboardButton("🔄 Refresh",       callback_data=f"tod:{sym}"),
        ]]
        await _safe_send_message(
            ctx.bot, cid, "\n".join(lines),
            reply_markup=InlineKeyboardMarkup(kb))

    async def _do_drift_report(self, cid: int, ctx, sym: str):
        fname   = _friendly(sym)
        n       = self.dm.n(sym)
        profile = effective_profile(sym)

        if n < MIN_TICKS:
            await ctx.bot.send_message(
                cid,
                f"{fname}: {n} ticks — need {MIN_TICKS}+.",
                parse_mode=ParseMode.MARKDOWN)
            return

        lr       = self.dm.log_returns(sym)
        S0       = self.dm.last(sym) or 1.0
        tod_mult = self.tod.multiplier(sym)

        # FIX v4.3: Pass horizon-scaled window to drift report
        default_horizon       = 300
        horizon_scaled_window = max(
            min(default_horizon * 2, profile.drift_window), 30)

        if profile.vol_regime_enabled:
            prices_arr = self.dm.prices(sym)
            vr = self.vre.analyse(
                sym, lr, S0, default_horizon,
                profile, tod_mult,
                window_override=horizon_scaled_window,
                prices=prices_arr)
            dir_icon = ("📈" if vr.signal == "bullish"
                        else ("📉" if vr.signal == "bearish" else "⬜"))
            sig_icon = ("✅ CONFIRMED" if vr.signal != "neutral"
                        else "❌ NOT CONFIRMED")
            lines = [
                f"*Vol Regime Report — {fname}* [{profile.name}]\n",
                f"Engine: *VOL_REGIME+MC* | ToD: x{tod_mult:.3f} | "
                f"Window: {horizon_scaled_window}t\n",
                f"Signal:    *{vr.signal.upper()}* {dir_icon} {sig_icon}",
                f"Strength:  *{vr.signal_strength}*",
                f"Regime:    *{vr.regime}*\n",
                f"*Realized vs Target:*",
                f"  Realized σ: `{vr.realized_sigma:.8f}`/tick",
                f"  Target σ:   `{vr.target_sigma:.8f}`/tick",
                f"  Deviation:  `{vr.vol_deviation:+.4f}` "
                f"({'above' if vr.vol_deviation > 0 else 'below'} target)",
                f"  k_adaptive: `{vr.k_adaptive:.4f}`\n",
                f"*Momentum (blended z-score):*",
                f"  z-score: `{vr.normalized_momentum:+.4f}`"
                + (" ✅" if abs(vr.normalized_momentum) >= profile.drift_tstat_min * 0.60
                   else " ❌"),
                f"  V-Edge:  `{vr.edge_score:.4f}`",
                f"  ToD adj: ±{abs(self.tod.prob_adjustment(sym))*100:.2f}%\n",
                f"*Trade Levels (k={vr.k_adaptive:.2f}):*",
                f"  Target UP:   `{vr.target_up:.6f}` "
                f"(+{(vr.target_up-S0)/S0*100:.3f}%)",
                f"  Target DOWN: `{vr.target_down:.6f}` "
                f"(-{(S0-vr.target_down)/S0*100:.3f}%)",
                f"  Stop UP:     `{vr.stop_up:.6f}`",
                f"  Stop DOWN:   `{vr.stop_down:.6f}`",
            ]
            if vr.signal != "neutral":
                rr = min(
                    abs(vr.target_up - S0) / max(abs(S0 - vr.stop_down), 1e-10),
                    4.0)
                lines += [
                    f"\n*Signal CONFIRMED* {dir_icon}",
                    f"Direction: {vr.signal.upper()}",
                    f"Approx R/R: 1:{rr:.2f} (capped at 4.0)",
                    f"Run `/predict {fname}` for full MC-calibrated setup.",
                ]
            else:
                lines += [
                    "\n*Signal NOT YET CONFIRMED*",
                    "Momentum not significant or regime unfavorable.",
                    "Monitor and check again later.",
                ]
        else:
            ds = self.de.analyse(
                sym, lr, S0, profile, tod_mult,
                window_override=horizon_scaled_window)
            dir_icon = ("📈" if ds.direction == "bullish"
                        else ("📉" if ds.direction == "bearish" else "⬜"))
            sig_icon = ("✅ CONFIRMED" if ds.direction != "neutral"
                        else "❌ NOT CONFIRMED")
            filled   = int(abs(ds.ewma_momentum) * 10)
            mom_bar  = "█" * filled + "░" * (10 - filled)
            lines = [
                f"*Drift Report — {fname}* [{profile.name}]\n",
                f"Engine: *DRIFT+MC* | ToD: x{tod_mult:.3f} | "
                f"Window: {horizon_scaled_window}t\n",
                f"Direction: *{ds.direction.upper()}* {dir_icon} {sig_icon}",
                f"Strength:  *{ds.signal_strength}*\n",
                f"*Statistics:*",
                f"  t-stat:  `{ds.tstat:+.6f}` "
                f"(thresh={profile.drift_tstat_min:.1f})"
                + (" ✓" if abs(ds.tstat) >= profile.drift_tstat_min else " ✗"),
                f"  p-value: `{ds.pvalue:.6f}`"
                + (" ✓" if ds.pvalue < 0.15 else " ✗ (need < 0.15)"),
                f"  EWMA mom:[{mom_bar}] `{ds.ewma_momentum:+.4f}`"
                + (" ✓" if abs(ds.ewma_momentum) > 0.05 else " ✗"),
                f"  Regime:  *{ds.regime}*",
                f"  D-Edge:  `{ds.edge_score:.4f}`",
                f"  Window:  `{ds.window_used}t`",
                f"  ToD adj: ±{abs(self.tod.prob_adjustment(sym))*100:.2f}%\n",
            ]
            if ds.direction != "neutral":
                lines += [
                    f"\n*Signal CONFIRMED* {dir_icon}",
                    f"Direction: {ds.direction.upper()}",
                    f"Run `/predict {fname}` for full trade setup.",
                ]
            else:
                lines += [
                    "\n*Signal NOT YET CONFIRMED*",
                    "Not all conditions met.",
                    "Monitor and check again later.",
                ]

        lines.append(RISK_MSG)
        kb = [[
            InlineKeyboardButton("📊 Full Analysis", callback_data=f"sym:{sym}"),
            InlineKeyboardButton("🔄 Refresh",       callback_data=f"drift:{sym}"),
            InlineKeyboardButton("🕐 ToD",           callback_data=f"tod:{sym}"),
        ]]
        await _safe_send_message(
            ctx.bot, cid, "\n".join(lines),
            reply_markup=InlineKeyboardMarkup(kb))

    async def _do_spike_check(self, cid: int, ctx, sym: str):
        fname   = _friendly(sym)
        n       = self.dm.n(sym)
        profile = effective_profile(sym)

        if n < MIN_TICKS:
            await ctx.bot.send_message(
                cid,
                f"{fname}: {n} ticks — need {MIN_TICKS}+.",
                parse_mode=ParseMode.MARKDOWN)
            return

        lr       = self.dm.log_returns(sym)
        jp       = self.jm.fit_unbiased(sym, lr)
        tr       = self.dm.observed_tick_rate(sym)
        tod_mult = self.tod.multiplier(sym)
        sa       = self.se.assess(sym, jp, 300, tr, tod_mult)

        if sa:
            report = self.se.format_standalone(sa)
            S0     = self.dm.last(sym) or 0.0
            self.pm.record(
                sym, 300, "spike_imminent",
                sa.confidence, sa.confidence, S0,
                S0*(1+sa.magnitude_p75/100),
                S0*(1-sa.magnitude_p75/100))
        else:
            jlo, jhi = self.jm.jump_magnitude_range(jp)
            pp       = self.jm.prob_in_n(jp, 300)
            tlo, thi, slo, shi, sc = self.jm.ticks_to_next_range(jp, tr)
            freq  = EXPECTED_JUMP_FREQ.get(sym, 0)
            report = (
                f"*Spike Diagnostic — {fname}* [{profile.name}]\n\n"
                f"Status: ⬜ NOT IMMINENT\n\n"
                f"Poisson P(300t): {pp:.2%} Bayesian λ "
                f"(need>={profile.spike_min_prob:.0%})\n"
                f"P60: {jp.spike_mag_p60*100:.3f}%\n"
                f"P75: {jp.spike_mag_p75*100:.3f}%\n"
                f"Hazard (Bayesian): {jp.hazard_intensity:.0%}\n"
                f"Lambda dev: {jp.lam_deviation:+.1f}%\n"
                f"Counter-drift: {jp.counter_drift:.8f}\n"
                f"Design rate: 1/{freq} ticks\n"
                f"Live (posterior): 1/~{1/max(jp.lam_posterior,1e-10):.0f}t\n"
                f"Est next: {slo}–{shi}\n"
                f"ToD: x{tod_mult:.3f}\n\n"
                f"⚠️ Max 1-2% risk per trade."
            )

        kb = [[
            InlineKeyboardButton("📊 Full Analysis", callback_data=f"sym:{sym}"),
            InlineKeyboardButton("🔄 Refresh",       callback_data=f"spike:{sym}"),
        ]]
        await _safe_send_message(
            ctx.bot, cid, report,
            reply_markup=InlineKeyboardMarkup(kb))

    async def _do_spike_scan(self, cid: int, ctx):
        await ctx.bot.send_message(
            cid,
            "🔍 Scanning Boom/Crash (Bayesian v4.3) ...",
            parse_mode=ParseMode.MARKDOWN)
        results  = []
        scan_syms = {**BOOM_SYMBOLS, **CRASH_SYMBOLS}
        for fn, ds in scan_syms.items():
            if self.dm.n(ds) < MIN_TICKS:
                continue
            try:
                profile  = effective_profile(ds)
                lr       = self.dm.log_returns(ds)
                jp       = self.jm.fit_unbiased(ds, lr)
                tr       = self.dm.observed_tick_rate(ds)
                tod_mult = self.tod.multiplier(ds)
                sa       = self.se.assess(ds, jp, 300, tr, tod_mult)
                pp       = self.jm.prob_in_n(jp, 300)
                hz       = jp.hazard_intensity
                conf     = (sa.confidence if sa
                            else float(np.clip((pp*tod_mult+hz)/2, 0, 1)))
                results.append((fn, ds, sa, conf, jp, profile, tod_mult))
            except Exception as e:
                log.debug(f"Spike scan {ds}: {e}")

        results.sort(key=lambda x: x[3], reverse=True)
        lines = ["*🚨 Spike Scanner v4.3 — Boom/Crash (Bayesian)*\n"]
        for fn, ds, sa, conf, jp, profile, tod_m in results[:15]:
            icon     = ("🚨" if sa else ("⚡" if conf > 0.4 else "⬜"))
            dir_icon = "📈" if jp.jump_sign > 0 else "📉"
            win_str  = (f" | {sa.time_lo_str}–{sa.time_hi_str}" if sa else "")
            lines.append(
                f"{icon} `{fn}` {dir_icon} | "
                f"Conf:{conf:.0%} | "
                f"Hz:{jp.hazard_intensity:.0%} | "
                f"P:{self.jm.prob_in_n(jp,300):.1%} | "
                f"ToD:x{tod_m:.2f}"
                f"{win_str}"
                + (" *IMMINENT*" if sa else ""))
        if not results:
            lines.append("No symbols ready yet.")
        kb = [
            [InlineKeyboardButton(fn, callback_data=f"spike:{ds}")]
            for fn, ds, sa, conf, jp, profile, tod_m
            in results[:6] if conf > 0.30
        ]
        markup = InlineKeyboardMarkup(kb) if kb else None
        await _safe_send_message(
            ctx.bot, cid, "\n".join(lines),
            reply_markup=markup)

    async def _send_prob_table(
            self, cid: int, ctx,
            sym: str, horizon: int, hlabel: str):
        fname   = _friendly(sym)
        S0      = self.dm.last(sym)
        n       = self.dm.n(sym)
        profile = effective_profile(sym)
        if not S0 or n < MIN_TICKS:
            await ctx.bot.send_message(cid, f"{fname}: {n} ticks — retry.")
            return
        results: Dict[str, MCResult] = {}
        for tf_key, tf_label in TIMEFRAMES.items():
            try:
                ticks, lbl = self.dm.ticks_for_timeframe(sym, tf_key)
                r = self.mce.run(
                    sym, self.dm, ticks, lbl,
                    self.pm, n=3000, timeframe_key=tf_key)
                if r:
                    results[tf_key] = r
            except Exception as e:
                log.debug(f"prob_table {tf_key}: {e}")

        tod_m = self.tod.multiplier(sym)
        lines = [
            f"*{fname}* [{profile.name}] — Probability Matrix v4.3\n",
            f"Price: `{S0:.6f}` | Ticks: `{n:,}` | ToD: x{tod_m:.3f}\n",
            f"{'TF':<8}|{'P(up)':>7}|{'P(+)':>7}|{'P(-)':>7}"
            f"|{'Edge':>6}|{'RR':>7}|{'EV':>7}|{'Str':>4}|{'Ens':>6}",
            "─" * 72,
        ]
        for tf_key, r in results.items():
            rr_str  = (f"1:{r.trade_setup.rr_ratio:.1f}"
                       if r.trade_setup else "—")
            st_str  = (r.trade_setup.signal_strength[:3]
                       if r.trade_setup else "—")
            ev_str  = (f"{r.trade_setup.expected_value:+.3f}"
                       if r.trade_setup else "—")
            ens_str = f"{r.ens_agreement:.2f}"
            se_i    = _se(r.signal)
            lines.append(
                f"{TIMEFRAMES[tf_key]:<8}|"
                f"{r.prob_up_horizon:>7.1%}|"
                f"{r.prob_hit_up:>7.1%}|"
                f"{r.prob_hit_down:>7.1%}|"
                f"{r.edge_score:>6.3f}|"
                f"{rr_str:>7}|"
                f"{ev_str:>7}|"
                f"{se_i}{st_str}|"
                f"{ens_str:>6}")

        if results:
            best_tf = max(results, key=lambda k: results[k].edge_score)
            br      = results[best_tf]
            lines  += [
                "",
                f"*Best:* {TIMEFRAMES[best_tf]} | "
                f"{_se(br.signal)} {br.signal.upper()} | "
                f"Edge {br.edge_score:.3f} | "
                f"Ens {br.ens_agreement:.3f}",
            ]
            if br.trade_setup:
                ts = br.trade_setup
                lines.append(
                    f"Setup: *{ts.direction}* | "
                    f"RR *1:{ts.rr_ratio:.2f}* | "
                    f"P *{ts.prob_target:.2%}* | "
                    f"EV `{ts.expected_value:+.4f}%` | "
                    f"*{ts.signal_strength}* [{ts.signal_source}] "
                    f"ToD:x{ts.tod_multiplier:.3f}")
            if br.vol_regime:
                vr = br.vol_regime
                lines.append(
                    f"VolReg: {vr.signal.upper()} | "
                    f"z={vr.normalized_momentum:+.3f} | "
                    f"k={vr.k_adaptive:.3f} | "
                    f"dev={vr.vol_deviation:+.3f}")
            elif br.drift_signal:
                ds = br.drift_signal
                lines.append(
                    f"Drift: {ds.direction.upper()} | "
                    f"t={ds.tstat:+.3f} | "
                    f"p={ds.pvalue:.4f} | "
                    f"mom={ds.ewma_momentum:+.3f} | "
                    f"win={ds.window_used}")

        lr  = self.dm.log_returns(sym)
        cat = _cat(sym)
        narr = ""
        try:
            ref = (results.get("5m") or
                   (list(results.values())[0] if results else None))
            if ref:
                if cat in ("vol", "step"):
                    gp = self.vm.fit(sym, lr)
                    narr = (self.narr.step_context(sym, gp, ref, ref.horizon_label)
                            if cat == "step"
                            else self.narr.vol_context(sym, gp, ref, ref.horizon_label))
                else:
                    jp   = self.jm.fit_unbiased(sym, lr)
                    narr = self.narr.jump_context(sym, jp, ref, ref.horizon_label)
        except Exception as e:
            log.debug(f"narrative: {e}")

        lines += ["", narr, "", self.pm.note(sym), RISK_MSG]
        await _safe_send_message(ctx.bot, cid, "\n".join(lines))

    async def _send_parameters(self, cid: int, ctx, sym: str):
        fname   = _friendly(sym)
        lr      = self.dm.log_returns(sym)
        n       = self.dm.n(sym)
        cat     = _cat(sym)
        rate    = self.dm.observed_tick_rate(sym)
        profile = effective_profile(sym)
        S0      = self.dm.last(sym) or 1.0
        tod_m   = self.tod.multiplier(sym)

        horizon_scaled_window = max(
            min(300 * 2, profile.drift_window), 30)

        lines = [
            f"*Parameters: {fname}* [{profile.name}]\n",
            f"Category: `{cat.upper()}` | Ticks: `{n:,}` | "
            f"Rate: `{rate:.2f}t/s` | ToD: x{tod_m:.3f}\n",
            f"*Profile v4.3:*\n"
            f"  Engine: `{'VOL_REGIME+MC' if profile.vol_regime_enabled else ('DRIFT+MC' if profile.drift_signal_enabled else 'MC+SPIKE')}`\n"
            f"  Edge min:    `{profile.edge_min:.3f}`\n"
            f"  Min target:  `{profile.min_target_pct*100:.4f}%`\n"
            f"  Min R/R:     `{profile.min_rr:.2f}` (global {MIN_RR_GLOBAL:.1f} cap 4.0)\n"
            f"  Stop min:    `>=30% target`\n"
            f"  Dir bias:    `{profile.direction_bias}`\n"
            f"  Alert edge:  `{profile.alert_edge_min:.3f}`\n"
            f"  Alert cool:  `{profile.alert_cooldown:.0f}s`\n"
            f"  P(tgt) gate: `>={MIN_PROB_TARGET:.0%}` (cap 72%)\n"
            f"  EV gate:     `>0 per-asset`\n"
            f"  Move gate:   `>={MIN_TARGET_MOVE*100:.2f}%`\n"
            f"  Ensemble:    `>={ENSEMBLE_AGREEMENT_FLOOR:.2f}`\n"
            f"  EmpCap:      `p{EMPIRICAL_CAP_PERCENTILE}x{EMPIRICAL_CAP_MULTIPLIER}`\n"
            f"  HorizWin:    `{horizon_scaled_window}t`\n",
        ]

        if len(lr) < MIN_TICKS:
            lines.append(f"Need {MIN_TICKS}+ ticks. Have {n}.")
        else:
            if cat in ("vol", "step"):
                p  = self.vm.fit(sym, lr)
                a  = self.vm.ann_vol(p) * 100
                d  = self.vm.deviation(p)
                rg = self.vm.detect_regime(lr)
                lines += [
                    f"*GBM+EWMA | Regime:{rg}*",
                    f"mu/tick:   `{p.mu:.9f}`",
                    f"sigma MLE: `{p.sigma:.9f}`",
                    f"EWMA sig:  `{p.sigma_ewma:.9f}`",
                    f"Ann vol:   `{a:.5f}%`",
                    f"KS pval:   `{p.ks_pvalue:.5f}`"
                    + (" normal" if p.ks_pvalue > 0.05 else " fat-tailed"),
                    f"Fit qual:  `{p.fit_quality:.4f}`",
                ]
                if p.advertised_vol:
                    lines.append(
                        f"Adv vol:   `{p.advertised_vol*100:.0f}%`"
                        f" | Dev: `{d:+.2f}%`")

                # FIX v4.3: Show empirical cap
                prices_arr = self.dm.prices(sym)
                emp_cap    = _empirical_move_cap(prices_arr, 300)
                if emp_cap is not None:
                    lines.append(
                        f"EmpCap(300t): `{emp_cap*100:.3f}%` "
                        f"(p{EMPIRICAL_CAP_PERCENTILE}x{EMPIRICAL_CAP_MULTIPLIER})")

                if profile.vol_regime_enabled:
                    prices_arr = self.dm.prices(sym)
                    vr = self.vre.analyse(
                        sym, lr, S0, 300, profile, tod_m,
                        window_override=horizon_scaled_window,
                        prices=prices_arr)
                    lines += [
                        f"\n*Vol Regime Engine v4.3:*",
                        f"  Signal:   `{vr.signal}` [{vr.signal_strength}]",
                        f"  Regime:   `{vr.regime}`",
                        f"  z-score:  `{vr.normalized_momentum:+.6f}` (blended)",
                        f"  k_adapt:  `{vr.k_adaptive:.4f}`",
                        f"  Vol dev:  `{vr.vol_deviation:+.4f}`",
                        f"  V-Edge:   `{vr.edge_score:.4f}`",
                        f"  Window:   `{horizon_scaled_window}t`",
                        f"  ToD adj:  ±{abs(self.tod.prob_adjustment(sym))*100:.2f}%",
                    ]
            else:
                jp  = self.jm.fit_unbiased(sym, lr)
                et  = (1/jp.lam_posterior
                       if jp.lam_posterior > 1e-10 else float("inf"))
                ef  = EXPECTED_JUMP_FREQ.get(sym, 0)
                pp  = self.jm.prob_in_n(jp, 300)
                tlo, thi, slo, shi, sc = self.jm.ticks_to_next_range(jp, rate)
                lines += [
                    "*Merton Jump-Diffusion v4.3 (Bayesian):*",
                    f"mu/tick:       `{jp.mu:.9f}`",
                    f"sigma/tick:    `{jp.sigma:.9f}`",
                    f"lambda MLE:    `{jp.lam:.7f}` (1/{1/max(jp.lam,1e-10):.0f}t)",
                    f"lambda post:   `{jp.lam_posterior:.7f}` (1/{et:.0f}t) [Bayesian]",
                    f"theory lam:    `{jp.expected_lam:.7f}` (1/{ef}t)",
                    f"lambda dev:    `{jp.lam_deviation:+.1f}%`",
                    f"Spike P60:     `{jp.spike_mag_p60*100:.4f}%`",
                    f"Spike P75:     `{jp.spike_mag_p75*100:.4f}%`",
                    f"Counter-drift: `{jp.counter_drift:.8f}`",
                    f"Hazard (Bayesian): `{jp.hazard_intensity:.0%}`",
                    f"P(spike/300t): `{pp:.2%}` (posterior λ)",
                    f"Est next:      `{slo}–{shi}`",
                ]

            if profile.drift_signal_enabled:
                ds = self.de.analyse(
                    sym, lr, S0, profile, tod_m,
                    window_override=horizon_scaled_window)
                lines += [
                    f"\n*Drift Analysis:*",
                    f"  Direction: `{ds.direction}`",
                    f"  t-stat:    `{ds.tstat:+.6f}`",
                    f"  p-value:   `{ds.pvalue:.6f}`",
                    f"  momentum:  `{ds.ewma_momentum:+.4f}`",
                    f"  regime:    `{ds.regime}`",
                    f"  d-edge:    `{ds.edge_score:.4f}`",
                    f"  strength:  `{ds.signal_strength}`",
                    f"  window:    `{ds.window_used}t`",
                ]

        lines += ["", self.pm.note(sym), RISK_MSG]
        await _safe_send_message(ctx.bot, cid, "\n".join(lines))

    async def _send_results(self, cid: int, ctx, sym: str):
        fname   = _friendly(sym)
        profile = effective_profile(sym)
        n, wr, ae = self.pm.stats(sym)
        recent    = self.pm.recent_results(sym)
        tod_m     = self.tod.multiplier(sym)
        if n == 0:
            await ctx.bot.send_message(
                cid,
                f"*Results — {fname}* [{profile.name}]\n\n"
                f"No resolved predictions yet.\n"
                f"v4.3 quality gates: fewer but higher-quality signals.\n"
                f"ToD: x{tod_m:.3f}",
                parse_mode=ParseMode.MARKDOWN)
            return
        wins   = sum(1 for r in self.pm.records
                     if r.symbol == sym and r.resolved and r.correct)
        losses = n - wins
        icon   = ("🟢" if wr >= 0.55 else ("🔴" if wr < 0.45 else "🟡"))
        msg    = (
            f"*Results — {fname}* [{profile.name}]\n\n"
            f"*Accuracy:* {icon} *{wr:.0%}*\n"
            f"W:{wins} L:{losses} Total:{n}\n"
            f"Avg edge: {ae:.3f} | ToD now: x{tod_m:.3f}\n\n"
            f"{recent}"
        )
        await _safe_send_message(ctx.bot, cid, msg)

    def _caption(
            self, fname: str, sym: str,
            mc: MCResult,
            gp: Optional[GBMParams],
            jp: Optional[JumpParams],
            note: str, risk_pct: float) -> str:
        S0      = mc.S0
        n       = self.dm.n(sym)
        se_i    = _se(mc.signal)
        profile = effective_profile(sym)

        L = [
            f"*{fname}* [{profile.name}] — {mc.horizon_label}\n",
            f"Price: `{_safe_md(S0,'.6f')}` | "
            f"Ticks: `{n:,}` | "
            f"ToD: x{mc.tod_multiplier:.3f} | "
            f"Ens: {mc.ens_agreement:.2f}",
        ]

        if mc.trade_setup:
            ts  = mc.trade_setup
            di  = "📈" if ts.direction == "BUY" else "📉"
            si  = {"STRONG": "🟢", "MODERATE": "🟡", "WEAK": "🔴"}.get(
                ts.signal_strength, "⬜")
            sg  = '+' if ts.direction == 'BUY' else '-'
            sgs = '-' if ts.direction == 'BUY' else '+'
            L += [
                f"\n{si} *{ts.direction}* {di} "
                f"[{ts.signal_strength}] [{ts.signal_source}]\n",
                f"Entry:  `{_safe_md(ts.entry,'.6f')}`",
                f"Target: `{_safe_md(ts.target,'.6f')}` ({sg}{ts.target_pct:.3f}%)",
                f"Stop:   `{_safe_md(ts.invalidation,'.6f')}` ({sgs}{ts.stop_pct:.3f}%)",
                f"R/R: *1:{ts.rr_ratio:.2f}* | P(tgt): `{ts.prob_target:.2%}`",
                f"Edge: `{ts.edge_pct:+.2f}%` | EV: `{ts.expected_value:+.4f}%`",
            ]
        else:
            L += [
                f"\n{se_i} *{mc.signal.upper()}*",
                f"Edge:`{_safe_md(mc.edge_score,'.4f')}` "
                f"CI:`{_safe_md(mc.ci_width_pct*100,'.4f')}%`",
                f"No trade setup — quality gates not met.",
            ]

        if mc.vol_regime and mc.vol_regime.signal != "neutral":
            vr = mc.vol_regime
            L.append(
                f"VolReg: `{vr.signal}` "
                f"z=`{vr.normalized_momentum:+.3f}` "
                f"k=`{vr.k_adaptive:.3f}`")
        elif mc.drift_signal:
            ds    = mc.drift_signal
            sig_i = "✅" if ds.direction != "neutral" else "❌"
            L.append(
                f"Drift: `{ds.direction}` "
                f"t=`{ds.tstat:+.3f}` "
                f"p=`{ds.pvalue:.3f}` {sig_i} "
                f"win={ds.window_used}")

        if mc.prob_jump is not None:
            L.append(
                f"P(spike/{mc.horizon}t): "
                f"`{mc.prob_jump:.2%}` (Bayesian λ)")

        L += ["", note, RISK_MSG]
        return "\n".join(line for line in L if line is not None)

    def _full_analysis(
            self, fname: str, sym: str,
            mc: MCResult,
            gp: Optional[GBMParams],
            jp: Optional[JumpParams],
            note: str, risk_pct: float,
            hlabel: str) -> str:
        cat   = _cat(sym)
        parts = []
        if mc.trade_setup:
            parts.append(TradeSetupBuilder.format_setup(
                mc.trade_setup, fname,
                mc.drift_signal, mc.vol_regime))
        try:
            if cat in ("vol", "step") and gp:
                if cat == "step":
                    parts.append(self.narr.step_context(sym, gp, mc, hlabel))
                else:
                    parts.append(self.narr.vol_context(sym, gp, mc, hlabel))
            elif jp:
                parts.append(self.narr.jump_context(sym, jp, mc, hlabel))
        except Exception as e:
            log.debug(f"narrative: {e}")
        parts.append(self.narr.risk_reward_block(mc, risk_pct))
        parts.append(f"\n{note}")
        parts.append(RISK_MSG)
        return "\n\n".join(p for p in parts if p)

    async def _alert_loop(self, bot):
        self._bot_ref = bot
        log.info(f"Alert scanner v4.3: warmup={ALERT_WARMUP}s")
        await asyncio.sleep(ALERT_WARMUP)
        log.info(
            f"Alert scanner v4.3 active. "
            f"Gates: P>={MIN_PROB_TARGET:.0%} EV>0 "
            f"RR>={MIN_RR_GLOBAL:.1f} "
            f"Move>={MIN_TARGET_MOVE*100:.2f}% "
            f"Ens>={ENSEMBLE_AGREEMENT_FLOOR:.2f} "
            f"EmpCap p{EMPIRICAL_CAP_PERCENTILE}x{EMPIRICAL_CAP_MULTIPLIER}")
        scan_count = 0
        syms       = list(ALL_SYMBOLS.items())

        while True:
            try:
                await asyncio.sleep(20)
                if not self.dm._init_done:
                    continue
                enabled = [c for c in self._chats
                           if self._state(c).alerts_enabled]
                if not enabled:
                    continue

                fn, ds  = syms[scan_count % len(syms)]
                scan_count += 1
                cat     = _cat(ds)
                profile = effective_profile(ds)
                tod_m   = self.tod.multiplier(ds)

                if profile.spike_enabled:
                    sa = self.ae.check_spike(ds, self.dm, self.jm, self.se)
                    if sa:
                        msg = self.se.format_standalone(sa)
                        for cid in enabled:
                            try:
                                await _safe_send_message(bot, cid, msg)
                                await asyncio.sleep(0.1)
                            except Exception as e:
                                log.debug(f"Spike alert {cid}: {e}")

                res = self.ae.check_trade(ds, self.dm, 300, "300 Ticks")
                if not res:
                    continue
                score, sig, mc = res

                spike_sa = None
                if profile.spike_enabled:
                    lr = self.dm.log_returns(ds)
                    if len(lr) >= MIN_TICKS:
                        jp       = self.jm.fit_unbiased(ds, lr)
                        tr       = self.dm.observed_tick_rate(ds)
                        spike_sa = self.se.assess(ds, jp, 300, tr, tod_m)

                validated, tv, reason = self.cr.resolve(
                    ds, mc, spike_sa, mc.trade_setup)
                if not tv or validated is None:
                    continue

                ts   = validated
                note = self.pm.note(ds, 300)
                di   = "📈" if ts.direction == "BUY" else "📉"
                si   = {"STRONG": "🟢", "MODERATE": "🟡"}.get(
                    ts.signal_strength, "🟡")
                sg   = '+' if ts.direction == 'BUY' else '-'
                sg2  = '-' if ts.direction == 'BUY' else '+'

                engine_line = ""
                if mc.vol_regime and mc.vol_regime.signal != "neutral":
                    vr = mc.vol_regime
                    engine_line = (
                        f"\nVolReg: *{vr.signal.upper()}* "
                        f"z={vr.normalized_momentum:+.3f} "
                        f"k={vr.k_adaptive:.3f} "
                        f"[{vr.signal_strength}]")
                elif mc.drift_signal:
                    ds_sig = mc.drift_signal
                    engine_line = (
                        f"\nDrift: *{ds_sig.direction.upper()}* "
                        f"t={ds_sig.tstat:+.3f} "
                        f"p={ds_sig.pvalue:.3f} "
                        f"win={ds_sig.window_used} "
                        f"[{ds_sig.signal_strength}]")

                text = (
                    f"*TRADE ALERT v4.3 — {fn}* [{profile.name}] {di} {_se(sig)}\n\n"
                    f"{si} *{ts.direction}* [{ts.signal_strength}] [{ts.signal_source}]\n"
                    f"Entry:  `{ts.entry:.6f}`\n"
                    f"Target: `{ts.target:.6f}` ({sg}{ts.target_pct:.3f}%)\n"
                    f"Stop:   `{ts.invalidation:.6f}` ({sg2}{ts.stop_pct:.3f}%)\n"
                    f"R/R:    *1:{ts.rr_ratio:.2f}*\n"
                    f"P(tgt): `{ts.prob_target:.2%}`\n"
                    f"Edge:   `{ts.edge_pct:+.2f}%`\n"
                    f"EV:     `{ts.expected_value:+.4f}%`\n"
                    f"Move:   `{ts.target_pct:.3f}%`\n"
                    f"Ens:    `{mc.ens_agreement:.3f}`\n"
                    f"ToD:    x{ts.tod_multiplier:.3f}"
                    f"{engine_line}\n\n"
                    f"{note}{RISK_MSG}"
                )
                for cid in enabled:
                    try:
                        await _safe_send_message(bot, cid, text)
                        await asyncio.sleep(0.1)
                    except Exception as e:
                        log.debug(f"Trade alert {cid}: {e}")

            except asyncio.CancelledError:
                log.info("Alert scanner v4.3 stopped.")
                break
            except Exception as e:
                log.warning(f"Alert loop: {e}")
                await asyncio.sleep(15)

    async def _post_init(self, app: Application):
        await app.bot.set_my_commands([
            BotCommand("start",       "Main menu"),
            BotCommand("predict",     "Analyse — /predict V75"),
            BotCommand("analyze",     "Same as predict"),
            BotCommand("spike",       "Spike — /spike Boom1000"),
            BotCommand("drift",       "Drift/Regime — /drift V75"),
            BotCommand("tod",         "Time-of-Day — /tod V75"),
            BotCommand("prob",        "Prob matrix — /prob V100"),
            BotCommand("parameters",  "Params — /parameters V75"),
            BotCommand("results",     "Results — /results V75"),
            BotCommand("watch",       "Watch — /watch V75"),
            BotCommand("unwatch",     "Unwatch"),
            BotCommand("status",      "Data status"),
            BotCommand("v75",         "Quick V75"),
            BotCommand("v100",        "Quick V100"),
            BotCommand("boom1000",    "Quick Boom1000"),
            BotCommand("crash500",    "Quick Crash500"),
            BotCommand("jump50",      "Quick Jump50"),
            BotCommand("step",        "Quick Step"),
        ])
        self._ws_task = asyncio.ensure_future(self.dm.run())
        self._al_task = asyncio.ensure_future(self._alert_loop(app.bot))
        log.info("SQE v4.3 background tasks launched OK")

    async def _post_stop(self, app: Application):
        log.info("Stopping SQE v4.3 ...")
        self.pe.save_user_states(self._states)
        self.pe.save_chats(self._chats)
        self.pe.save_tod_data(self.tod.to_dict())
        for t in (self._ws_task, self._al_task):
            if t and not t.done():
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
        await self.dm.stop()
        log.info("SQE v4.3 shutdown — state saved OK")

    async def _track_msg(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        self._track(update)

    def run(self):
        app = (
            Application.builder()
            .token(TELEGRAM_TOKEN)
            .post_init(self._post_init)
            .post_stop(self._post_stop)
            .build()
        )
        for cmd in [
            "v10", "v25", "v50", "v75", "v100", "v250",
            "boom300", "boom500", "boom600", "boom900", "boom1000",
            "crash300", "crash500", "crash600", "crash900", "crash1000",
            "jump10", "jump25", "jump50", "jump75", "jump100",
            "step", "stepindex",
        ]:
            app.add_handler(CommandHandler(cmd, self.cmd_quick_sym))

        app.add_handler(CommandHandler("start",      self.cmd_start))
        app.add_handler(CommandHandler("predict",    self.cmd_predict))
        app.add_handler(CommandHandler("analyze",    self.cmd_analyze))
        app.add_handler(CommandHandler("spike",      self.cmd_spike))
        app.add_handler(CommandHandler("drift",      self.cmd_drift))
        app.add_handler(CommandHandler("tod",        self.cmd_tod))
        app.add_handler(CommandHandler("prob",       self.cmd_prob))
        app.add_handler(CommandHandler("parameters", self.cmd_parameters))
        app.add_handler(CommandHandler("results",    self.cmd_results))
        app.add_handler(CommandHandler("watch",      self.cmd_watch))
        app.add_handler(CommandHandler("unwatch",    self.cmd_unwatch))
        app.add_handler(CommandHandler("status",     self.cmd_status))
        app.add_handler(CallbackQueryHandler(self.cb_query))
        app.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            self._track_msg))

        log.info("SQE v4.3 — Starting polling ...")
        app.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True)

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("=" * 66)
    log.info("  SYNTHETIC QUANT ELITE v4.3")
    log.info("  Calibrated Precision Edition")
    log.info("  6 Surgical Fixes Applied")
    log.info("=" * 66)
    log.info(f"  Global gates v4.3:")
    log.info(
        f"    P>={MIN_PROB_TARGET:.0%} (cap 72%)  "
        f"EV>0 per-asset  "
        f"RR>={MIN_RR_GLOBAL:.1f} (dynamic cap 4.0)")
    log.info(
        f"    Stop>=30% target  "
        f"Move>={MIN_TARGET_MOVE*100:.2f}%  "
        f"Ens>={ENSEMBLE_AGREEMENT_FLOOR:.2f}")
    log.info(
        f"    EmpCap p{EMPIRICAL_CAP_PERCENTILE}x{EMPIRICAL_CAP_MULTIPLIER}  "
        f"HybridRefiner cold<{REFINER_MIN_PATTERNS}")
    log.info(
        f"  ToD: {TOD_BUCKETS} buckets  "
        f"alpha={TOD_ALPHA}  "
        f"range=[{TOD_MULT_MIN},{TOD_MULT_MAX}]  "
        f"adj=±3% max")
    log.info("=" * 66)

    for name, profile in [
        ("VOL",   PROFILE_VOL),
        ("BOOM",  PROFILE_BOOM),
        ("CRASH", PROFILE_CRASH),
        ("JUMP",  PROFILE_JUMP),
        ("STEP",  PROFILE_STEP),
    ]:
        engine = (
            "VOL_REGIME+MC" if profile.vol_regime_enabled
            else ("DRIFT+MC" if profile.drift_signal_enabled
                  else "MC+SPIKE"))
        log.info(
            f"  [{name:<5}] {engine:<14} "
            f"edge>={profile.alert_edge_min:.2f} "
            f"RR>={profile.alert_min_rr:.1f} "
            f"cool={profile.alert_cooldown:.0f}s "
            f"spike={profile.spike_enabled} "
            f"bias={profile.direction_bias} "
            f"str={profile.alert_strength}")

    log.info("=" * 66)

    if not TELEGRAM_TOKEN:
        log.error("TELEGRAM_TOKEN not set!")
        raise SystemExit(1)

    log.info(f"Telegram: ...{TELEGRAM_TOKEN[-8:]}")
    log.info(f"Deriv:    {DERIV_APP_ID}")
    log.info(f"Persist:  {PERSIST_FILE}")
    log.info(f"Patterns: {PATTERN_FILE}")
    log.info(f"MC paths: {MC_PATHS:,}")
    log.info("=" * 66)

    try:
        bot = SQEBotV43()
        bot.run()
    except KeyboardInterrupt:
        log.info("Interrupted — shutting down.")
    except Exception as e:
        log.critical(f"Fatal: {e}", exc_info=True)
        raise SystemExit(1)
