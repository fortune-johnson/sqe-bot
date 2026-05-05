"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║   SYNTHETIC QUANT ELITE v4.5 — PRODUCTION                                      ║
║   Changes from v4.4:                                                            ║
║   REMOVED: PostSpikeFade (negative EV, unreliable)                             ║
║   REMOVED: FadeSetup, FadeCandidate dataclasses                                ║
║   ADDED:   SpikeSuppressionAnalyser — 5-factor quality scoring                 ║
║             Mirrors VolSignalQuality pattern exactly:                           ║
║             — Alert ALWAYS fires if raw_confidence >= 0.95                     ║
║             — Suppression adds star rating + warnings for user                  ║
║             — User sees quality info and makes own decision                     ║
║             — Stars NEVER block the alert (only inform)                        ║
║   ADDED:   VolSignalQuality — 5-factor directional signal quality              ║
║   ADDED:   Star rating system (1-5) on all alerts                              ║
║   All existing MC, Bayesian hazard, Vol Regime, Drift logic preserved.         ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""
import os
import asyncio
import json
import logging
import math
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

# ============================================================================
# LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("SQEv45_Prod")

# Dummy HTTP server for Render
if os.environ.get("RENDER"):
    from threading import Thread
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"SQE Bot Running")
        def log_message(self, *args):
            pass

    def run_server():
        port = int(os.environ.get("PORT", 10000))
        HTTPServer(("0.0.0.0", port), Handler).serve_forever()

    Thread(target=run_server, daemon=True).start()

# ============================================================================
# CREDENTIALS & CONFIG
# ============================================================================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "8772749656:AAFhykHBZeXHaGI-YaW-P6mC4M0avmINpYY")
DERIV_APP_ID   = os.environ.get("DERIV_APP_ID", "1089")
DERIV_WS_URL   = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"

BUFFER_SIZE    = 10_000
HISTORY_COUNT  = 2_000
MC_PATHS       = 10_000
MC_STEPS_MAX   = 600
JUMP_THRESHOLD = 4.5
MIN_TICKS      = 80
ALERT_WARMUP   = 30
EWMA_LAMBDA    = 0.94

ENABLE_ENSEMBLE          = True
ENSEMBLE_AGREEMENT_FLOOR = 0.60

EMPIRICAL_CAP_PERCENTILE = 80
EMPIRICAL_CAP_MULTIPLIER = 1.5
EMPIRICAL_MIN_WINDOWS    = 20

REFINER_MIN_PATTERNS = 5

MIN_PROB_TARGET  = 0.53
MIN_EDGE_PCT     = 0.0
MIN_EV           = 0.0
MIN_TARGET_MOVE  = 0.001
MIN_RR_GLOBAL    = 1.4

TOD_BUCKETS  = 48
TOD_ALPHA    = 0.12
TOD_MULT_MIN = 0.88
TOD_MULT_MAX = 1.12

PERSIST_FILE = "sqe_v45_state.pkl"
PATTERN_FILE = "pattern_memory_v45.pkl"

# ============================================================================
# SPIKE ALERT THRESHOLD
# Raw confidence gate — unchanged from v4.3
# Suppression analysis runs AFTER this gate is passed
# and adds quality info to the alert message.
# Stars NEVER block the alert — they inform the user.
# ============================================================================
SPIKE_MIN_CONFIDENCE = 0.95

# Bootstrap ensemble
N_BOOTSTRAP_RUNS    = 3
BOOTSTRAP_AGREEMENT = 0.60

# Imbalance filter
IMBALANCE_WINDOW  = 50
IMBALANCE_THRESH  = 0.60
IMBALANCE_BOOST   = 1.10
IMBALANCE_PENALTY = 0.90

# Momentum quality
MOMENTUM_QUALITY_WINDOW   = 15
MOMENTUM_QUALITY_STRONG   = 0.70
MOMENTUM_QUALITY_CONFLICT = 0.35
MOMENTUM_BOOST            = 1.05
MOMENTUM_PENALTY          = 0.90

# Adaptive thresholds
ADAPTIVE_PRIOR_STRENGTH = 50.0
ADAPTIVE_LEARNING_RATE  = 0.02
ADAPTIVE_MIN_PROB       = 0.51
ADAPTIVE_MAX_PROB       = 0.65
ADAPTIVE_MIN_RR         = 1.2
ADAPTIVE_MAX_RR         = 2.2

# ============================================================================
# v4.5 SPIKE SUPPRESSION — INFORMATIONAL ONLY
# These factors adjust a DISPLAY confidence and compute a star rating.
# They do NOT gate or block alerts.
# The only gate remains: raw_confidence >= SPIKE_MIN_CONFIDENCE
# This mirrors how VolSignalQuality works for directional signals.
# ============================================================================
DROUGHT_DISCOUNT_2X  = 0.85
DROUGHT_DISCOUNT_3X  = 0.65
DROUGHT_DISCOUNT_5X  = 0.40

VOL_COMPRESSION_MILD           = -0.20
VOL_COMPRESSION_STRONG         = -0.35
VOL_COMPRESSION_DISCOUNT_MILD  = 0.85
VOL_COMPRESSION_DISCOUNT_STRONG= 0.65

ACTIVITY_RATIO_QUIET       = 0.50
ACTIVITY_RATIO_VERY_QUIET  = 0.20
ACTIVITY_DISCOUNT_QUIET    = 0.85
ACTIVITY_DISCOUNT_VERY_QUIET = 0.60

POST_SPIKE_COOLING_FRACTION = 0.40

RANGE_COMPRESSION_RATIO    = 0.60
RANGE_EXPANSION_RATIO      = 1.40
RANGE_COMPRESSION_DISCOUNT = 0.90
RANGE_EXPANSION_BOOST      = 1.05

# ============================================================================
# v4.5 VOL SIGNAL QUALITY — same informational pattern
# ============================================================================
VOL_QUALITY_WINDOWS       = [50, 150, 300]
PERSISTENCE_WINDOW        = 10
PERSISTENCE_INTERVAL      = 20
PERSISTENCE_MIN_AGREEMENT = 0.70
DRIFT_NOISE_MIN_RATIO     = 0.05

# Minimum stars to include quality block in message
# (below this we still send, just with a clear caution header)
SPIKE_CAUTION_STARS    = 3   # below this: add CAUTION header
DIRECTION_CAUTION_STARS= 3

# ============================================================================
# TIMEFRAMES & SYMBOLS
# ============================================================================
TIMEFRAMES: Dict[str, str] = {
    "1m":  "1 Minute",  "5m":  "5 Minutes",
    "15m": "15 Minutes","30m": "30 Minutes", "1h": "1 Hour",
}
TF_SECONDS: Dict[str, int] = {
    "1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600,
}
HORIZONS: Dict[str, int] = {
    "50t": 50, "100t": 100, "300t": 300, "600t": 600,
}
HORIZON_LABELS: Dict[str, str] = {
    "50t": "50 Ticks",  "100t": "100 Ticks",
    "300t":"300 Ticks", "600t": "600 Ticks",
}

VOLATILITY_SYMBOLS: Dict[str, str] = {
    "V10": "R_10", "V25": "R_25", "V50": "R_50",
    "V75": "R_75", "V100": "R_100", "V10(1s)": "1HZ10V",
    "V25(1s)": "1HZ25V", "V50(1s)": "1HZ50V",
    "V75(1s)": "1HZ75V", "V100(1s)": "1HZ100V",
    "V250": "R_250",
}
BOOM_SYMBOLS: Dict[str, str] = {
    "Boom300": "BOOM300N", "Boom500": "BOOM500",
    "Boom600": "BOOM600N", "Boom900": "BOOM900",
    "Boom1000": "BOOM1000",
}
CRASH_SYMBOLS: Dict[str, str] = {
    "Crash300": "CRASH300N", "Crash500": "CRASH500",
    "Crash600": "CRASH600N", "Crash900": "CRASH900",
    "Crash1000": "CRASH1000",
}
STEP_SYMBOLS:  Dict[str, str] = {"Step Index": "stpRNG"}
JUMP_SYMBOLS:  Dict[str, str] = {
    "Jump10": "JD10",  "Jump25": "JD25",  "Jump50": "JD50",
    "Jump75": "JD75",  "Jump100": "JD100",
}
ALL_SYMBOLS: Dict[str, str] = {
    **VOLATILITY_SYMBOLS, **BOOM_SYMBOLS,
    **CRASH_SYMBOLS, **STEP_SYMBOLS, **JUMP_SYMBOLS,
}
REVERSE_MAP: Dict[str, str] = {v: k for k, v in ALL_SYMBOLS.items()}

EXPECTED_JUMP_FREQ: Dict[str, int] = {
    "BOOM300N": 300,  "BOOM500":  500,  "BOOM600N": 600,
    "BOOM900":  900,  "BOOM1000": 1000,
    "CRASH300N":300,  "CRASH500": 500,  "CRASH600N":600,
    "CRASH900": 900,  "CRASH1000":1000,
    "JD10": 10, "JD25": 25, "JD50": 50,
    "JD75": 75, "JD100":100,
}
ADVERTISED_VOL: Dict[str, float] = {
    "R_10": 0.10, "R_25": 0.25, "R_50": 0.50,
    "R_75": 0.75, "R_100":1.00, "R_250":2.50,
    "1HZ10V":0.10, "1HZ25V":0.25, "1HZ50V":0.50,
    "1HZ75V":0.75, "1HZ100V":1.00,
    "JD10":0.10, "JD25":0.25, "JD50":0.50,
    "JD75":0.75, "JD100":1.00, "stpRNG":0.01,
}

# Chart colours
BG  = "#0a0e1a"; AX  = "#0f1629"; GR  = "#00ff88"
RD  = "#ff3355"; BL  = "#3d8eff"; YL  = "#ffd700"
OR  = "#ff8c00"; PU  = "#b44fff"; CY  = "#00e5ff"
GY  = "#5a6680"; WH  = "#e8eaf6"; DG  = "#1e2540"
GR2 = "#00cc6a"; RD2 = "#cc2244"

# ============================================================================
# HELPERS
# ============================================================================
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
        if k.upper().replace(" ", "") == s or v.upper() == s:
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

def _stars(n: int) -> str:
    """Returns star string e.g. ★★★☆☆"""
    n = max(1, min(5, n))
    return "★" * n + "☆" * (5 - n)

def _esc(text: str) -> str:
    return str(text).replace("`","'").replace("*","").replace("_","")

def _safe_md(val, fmt: str) -> str:
    return _esc(format(val, fmt))

def _tod_bucket() -> int:
    now = datetime.now(timezone.utc)
    return now.hour * 2 + (1 if now.minute >= 30 else 0)

def _empirical_move_cap(
    prices: np.ndarray, horizon: int,
    cap_percentile: float = EMPIRICAL_CAP_PERCENTILE,
    cap_multiplier: float = EMPIRICAL_CAP_MULTIPLIER,
    min_windows: int = EMPIRICAL_MIN_WINDOWS,
) -> Optional[float]:
    if prices is None or len(prices) < horizon * 2:
        return None
    step  = max(1, horizon // 4)
    moves = []
    for i in range(0, len(prices) - horizon, step):
        p0 = prices[i]; p1 = prices[i + horizon]
        if p0 > 1e-10:
            moves.append(abs(p1 - p0) / p0)
    if len(moves) < min_windows:
        return None
    return float(np.percentile(moves, cap_percentile)) * cap_multiplier

RISK_MSG = (
    "\n\n*Risk Warning:* No model guarantees profit. "
    "Max 1-2% account risk per trade. "
    "Statistical edge does not guarantee future results."
)
MAX_CAPTION = 950

async def _safe_send_photo(bot, chat_id: int,
                           photo, caption: str, **kwargs):
    try:
        if len(caption) <= MAX_CAPTION:
            return await bot.send_photo(
                chat_id, photo=photo, caption=caption,
                parse_mode=ParseMode.MARKDOWN, **kwargs,
            )
        short = caption[:MAX_CAPTION].rsplit("\n", 1)[0]
        await bot.send_photo(
            chat_id, photo=photo, caption=short,
            parse_mode=ParseMode.MARKDOWN, **kwargs,
        )
        rest = caption[len(short):]
        if rest.strip():
            await _safe_send_message(bot, chat_id, rest)
    except Exception as e:
        log.warning(f"send_photo error: {e}")
        try:
            plain = caption.replace("*","").replace("`","")
            await bot.send_photo(
                chat_id, photo=photo, caption=plain[:MAX_CAPTION]
            )
        except Exception as e2:
            log.error(f"send_photo fallback: {e2}")

async def _safe_send_message(bot, chat_id: int,
                             text: str, **kwargs):
    MAX_MSG = 4000
    chunks  = [text[i:i+MAX_MSG] for i in range(0, len(text), MAX_MSG)]
    for idx, chunk in enumerate(chunks):
        kw = kwargs.copy() if idx == 0 else {}
        try:
            await bot.send_message(
                chat_id, chunk,
                parse_mode=ParseMode.MARKDOWN, **kw,
            )
        except Exception as e:
            if any(x in str(e).lower()
                   for x in ["parse","entity","can't"]):
                plain = (chunk.replace("*","")
                              .replace("`","")
                              .replace("_",""))
                try:
                    await bot.send_message(chat_id, plain)
                except Exception as e2:
                    log.error(f"send_message fallback: {e2}")
            else:
                log.error(f"send_message: {e}")

# ============================================================================
# TIME-OF-DAY PROFILE
# ============================================================================
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
                TodBucket() for _ in range(TOD_BUCKETS)
            ]

    def update(self, sym: str, win: bool, edge: float,
               ev: float, vol_dev: float, tick_rate: float,
               bucket: Optional[int] = None):
        self._ensure(sym)
        b_idx = bucket if bucket is not None else _tod_bucket()
        b = self._buckets[sym][b_idx]
        a = TOD_ALPHA
        b.count    += 1.0
        b.win_rate    = (1-a)*b.win_rate    + a*(1.0 if win else 0.0)
        b.avg_edge    = (1-a)*b.avg_edge    + a*edge
        b.avg_ev      = (1-a)*b.avg_ev      + a*ev
        b.avg_vol_dev = (1-a)*b.avg_vol_dev + a*vol_dev
        b.tick_rate   = (1-a)*b.tick_rate   + a*tick_rate
        wr_factor   = float(np.clip(
            1.0+(b.win_rate-0.50)*0.60, 0.80, 1.25
        ))
        edge_factor = float(np.clip(
            1.0+b.avg_edge*0.30, 0.90, 1.15
        ))
        conf_weight = float(np.clip(b.count/20.0, 0.0, 1.0))
        raw_mult    = 1.0 + conf_weight*((wr_factor*edge_factor)-1.0)
        b.multiplier = float(np.clip(
            raw_mult, TOD_MULT_MIN, TOD_MULT_MAX
        ))

    def multiplier(self, sym: str,
                   bucket: Optional[int] = None) -> float:
        self._ensure(sym)
        b_idx = bucket if bucket is not None else _tod_bucket()
        b = self._buckets[sym][b_idx]
        return b.multiplier if b.count >= 3 else 1.0

    def prob_adjustment(self, sym: str,
                        bucket: Optional[int] = None) -> float:
        mult = self.multiplier(sym, bucket)
        return float(np.clip((mult-1.0)*0.05, -0.03, 0.03))

    def best_horizon(self, sym: str,
                     available: List[int]) -> int:
        self._ensure(sym)
        b = self._buckets[sym][_tod_bucket()]
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
        return ""

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
        if not isinstance(d, dict):
            log.warning("ToD data corrupted — resetting")
            return
        for sym, blist in d.items():
            if not isinstance(blist, list):
                continue
            self._buckets[sym] = []
            for bk in blist:
                if not isinstance(bk, dict):
                    self._buckets[sym].append(TodBucket())
                    continue
                tb = TodBucket()
                tb.count       = bk.get("count",       0.0)
                tb.win_rate    = bk.get("win_rate",     0.50)
                tb.avg_edge    = bk.get("avg_edge",     0.0)
                tb.avg_ev      = bk.get("avg_ev",       0.0)
                tb.avg_vol_dev = bk.get("avg_vol_dev",  0.0)
                tb.tick_rate   = bk.get("tick_rate",    0.5)
                tb.multiplier  = bk.get("multiplier",   1.0)
                self._buckets[sym].append(tb)
            while len(self._buckets[sym]) < TOD_BUCKETS:
                self._buckets[sym].append(TodBucket())

# ============================================================================
# ASSET PROFILES
# ============================================================================
@dataclass
class AssetProfile:
    name: str
    model_type: str = "GBM"
    edge_min: float = 0.20
    direction_gap: float = 0.008
    ci_width_min: float = 0.000005
    signal_edge_min: float = 0.15
    strong_edge: float = 0.25
    strong_prob: float = 0.58
    strong_rr: float = 2.0
    moderate_edge: float = 0.13
    moderate_prob: float = 0.54
    moderate_rr: float = 1.4
    trade_enabled: bool = True
    min_target_pct: float = 0.001
    min_stop_pct: float = 0.001
    min_rr: float = 1.4
    target_percentile: float = 60.0
    stop_percentile: float = 40.0
    direction_bias: str = "NONE"
    spike_enabled: bool = False
    spike_min_prob: float = 0.45
    spike_min_magnitude: float = 0.03
    spike_lam_dev_min: float = 8.0
    spike_hazard_high: float = 0.65
    spike_is_primary: bool = False
    alert_edge_min: float = 0.20
    alert_cooldown: float = 300.0
    alert_min_ticks: int = 400
    alert_min_rr: float = 1.4
    alert_strength: List[str] = field(
        default_factory=lambda: ["STRONG","MODERATE"]
    )
    mc_paths: int = 10_000
    sigma_multiplier_base: float = 1.5
    drift_signal_enabled: bool = False
    drift_tstat_min: float = 2.0
    drift_window: int = 200
    drift_edge_weight: float = 0.55
    vol_regime_enabled: bool = False
    vol_k_min: float = 2.0
    vol_k_max: float = 2.8
    resolve_threshold_pct: float = 0.05
    tod_enabled: bool = True

PROFILE_VOL = AssetProfile(
    name="VOLATILITY", model_type="GBM",
    edge_min=0.12, direction_gap=0.003, ci_width_min=0.000001,
    signal_edge_min=0.10, strong_edge=0.25, strong_prob=0.58,
    strong_rr=2.0, moderate_edge=0.12, moderate_prob=0.54,
    moderate_rr=1.4, trade_enabled=True, min_target_pct=0.001,
    min_stop_pct=0.001, min_rr=1.4, target_percentile=65.0,
    stop_percentile=35.0, direction_bias="NONE", spike_enabled=False,
    alert_edge_min=0.12, alert_cooldown=360.0, alert_min_ticks=400,
    alert_min_rr=1.4, alert_strength=["STRONG","MODERATE"],
    mc_paths=10_000, sigma_multiplier_base=2.2,
    drift_signal_enabled=True, drift_tstat_min=2.0,
    drift_window=200, drift_edge_weight=0.55,
    vol_regime_enabled=True, vol_k_min=2.0, vol_k_max=2.8,
    resolve_threshold_pct=0.02, tod_enabled=True,
)
PROFILE_BOOM = AssetProfile(
    name="BOOM", model_type="JumpDiffusion",
    edge_min=0.25, direction_gap=0.010, ci_width_min=0.00001,
    signal_edge_min=0.15, strong_edge=0.40, strong_prob=0.58,
    strong_rr=2.0, moderate_edge=0.20, moderate_prob=0.54,
    moderate_rr=1.4, trade_enabled=True, min_target_pct=0.0025,
    min_stop_pct=0.005, min_rr=1.4, target_percentile=65.0,
    stop_percentile=35.0, direction_bias="BUY", spike_enabled=True,
    spike_min_prob=0.45, spike_min_magnitude=0.03,
    spike_lam_dev_min=8.0, spike_hazard_high=0.65,
    spike_is_primary=True, alert_edge_min=0.20,
    alert_cooldown=300.0, alert_min_ticks=400, alert_min_rr=1.4,
    alert_strength=["STRONG","MODERATE"], mc_paths=10_000,
    sigma_multiplier_base=2.2, drift_signal_enabled=False,
    vol_regime_enabled=False, resolve_threshold_pct=0.10,
    tod_enabled=True,
)
PROFILE_CRASH = AssetProfile(
    name="CRASH", model_type="JumpDiffusion",
    edge_min=0.25, direction_gap=0.010, ci_width_min=0.00001,
    signal_edge_min=0.15, strong_edge=0.40, strong_prob=0.58,
    strong_rr=2.0, moderate_edge=0.20, moderate_prob=0.54,
    moderate_rr=1.4, trade_enabled=True, min_target_pct=0.0025,
    min_stop_pct=0.005, min_rr=1.4, target_percentile=35.0,
    stop_percentile=65.0, direction_bias="SELL", spike_enabled=True,
    spike_min_prob=0.45, spike_min_magnitude=0.03,
    spike_lam_dev_min=8.0, spike_hazard_high=0.65,
    spike_is_primary=True, alert_edge_min=0.20,
    alert_cooldown=300.0, alert_min_ticks=400, alert_min_rr=1.4,
    alert_strength=["STRONG","MODERATE"], mc_paths=10_000,
    sigma_multiplier_base=2.2, drift_signal_enabled=False,
    vol_regime_enabled=False, resolve_threshold_pct=0.10,
    tod_enabled=True,
)
PROFILE_JUMP = AssetProfile(
    name="JUMP", model_type="JumpDiffusion",
    edge_min=0.15, direction_gap=0.005, ci_width_min=0.000005,
    signal_edge_min=0.12, strong_edge=0.28, strong_prob=0.57,
    strong_rr=2.0, moderate_edge=0.15, moderate_prob=0.54,
    moderate_rr=1.4, trade_enabled=True, min_target_pct=0.001,
    min_stop_pct=0.005, min_rr=1.4, target_percentile=62.0,
    stop_percentile=38.0, direction_bias="NONE", spike_enabled=False,
    spike_is_primary=False, alert_edge_min=0.15,
    alert_cooldown=180.0, alert_min_ticks=200, alert_min_rr=1.4,
    alert_strength=["STRONG","MODERATE"], mc_paths=10_000,
    sigma_multiplier_base=2.0, drift_signal_enabled=True,
    drift_tstat_min=1.8, drift_window=150, drift_edge_weight=0.45,
    vol_regime_enabled=False, resolve_threshold_pct=0.05,
    tod_enabled=True,
)
PROFILE_STEP = AssetProfile(
    name="STEP", model_type="Step",
    edge_min=0.10, direction_gap=0.002, ci_width_min=0.0000001,
    signal_edge_min=0.08, strong_edge=0.22, strong_prob=0.56,
    strong_rr=1.5, moderate_edge=0.12, moderate_prob=0.54,
    moderate_rr=1.4, trade_enabled=True, min_target_pct=0.0001,
    min_stop_pct=0.00005, min_rr=1.4, target_percentile=65.0,
    stop_percentile=35.0, direction_bias="NONE", spike_enabled=False,
    alert_edge_min=0.10, alert_cooldown=360.0, alert_min_ticks=300,
    alert_min_rr=1.4, alert_strength=["STRONG","MODERATE"],
    mc_paths=8_000, sigma_multiplier_base=1.0,
    drift_signal_enabled=True, drift_tstat_min=2.2,
    drift_window=300, drift_edge_weight=0.55,
    vol_regime_enabled=False, resolve_threshold_pct=0.01,
    tod_enabled=True,
)

@dataclass
class SymbolTune:
    alert_cooldown_override:    Optional[float] = None
    min_target_pct_override:    Optional[float] = None
    min_rr_override:            Optional[float] = None
    spike_min_prob_override:    Optional[float] = None
    mc_paths_override:          Optional[int]   = None
    sigma_mult_override:        Optional[float] = None
    drift_tstat_override:       Optional[float] = None
    drift_window_override:      Optional[int]   = None
    vol_k_min_override:         Optional[float] = None
    vol_k_max_override:         Optional[float] = None

SYMBOL_TUNES: Dict[str, SymbolTune] = {
    "R_10":   SymbolTune(min_target_pct_override=0.001,
                         min_rr_override=1.4, sigma_mult_override=1.2,
                         drift_tstat_override=2.0,
                         drift_window_override=300,
                         vol_k_min_override=1.8, vol_k_max_override=2.5),
    "R_25":   SymbolTune(min_target_pct_override=0.0015,
                         min_rr_override=1.4, sigma_mult_override=1.3,
                         drift_tstat_override=2.0,
                         vol_k_min_override=1.9, vol_k_max_override=2.6),
    "R_50":   SymbolTune(min_target_pct_override=0.002,
                         min_rr_override=1.4, sigma_mult_override=1.6,
                         vol_k_min_override=1.8, vol_k_max_override=2.5),
    "R_75":   SymbolTune(min_target_pct_override=0.002,
                         min_rr_override=1.4, sigma_mult_override=1.8,
                         vol_k_min_override=2.0, vol_k_max_override=2.6),
    "R_100":  SymbolTune(min_target_pct_override=0.003,
                         min_rr_override=1.4, sigma_mult_override=2.2,
                         vol_k_min_override=2.2, vol_k_max_override=2.8),
    "R_250":  SymbolTune(min_target_pct_override=0.005,
                         min_rr_override=1.4, sigma_mult_override=2.5,
                         vol_k_min_override=2.3, vol_k_max_override=2.8),
    "1HZ10V": SymbolTune(min_target_pct_override=0.001,
                          min_rr_override=1.4, sigma_mult_override=1.2,
                          drift_tstat_override=2.0,
                          vol_k_min_override=1.8, vol_k_max_override=2.5),
    "1HZ25V": SymbolTune(min_target_pct_override=0.0015,
                          min_rr_override=1.4, sigma_mult_override=1.3,
                          vol_k_min_override=1.9, vol_k_max_override=2.6),
    "1HZ50V": SymbolTune(min_target_pct_override=0.002,
                          min_rr_override=1.4, sigma_mult_override=1.6,
                          vol_k_min_override=1.8, vol_k_max_override=2.5),
    "1HZ75V": SymbolTune(min_target_pct_override=0.002,
                          min_rr_override=1.4, sigma_mult_override=1.8,
                          vol_k_min_override=2.0, vol_k_max_override=2.6),
    "1HZ100V":SymbolTune(min_target_pct_override=0.003,
                          min_rr_override=1.4, sigma_mult_override=2.2,
                          vol_k_min_override=2.2, vol_k_max_override=2.8),
    "BOOM300N":  SymbolTune(alert_cooldown_override=180.0,
                             spike_min_prob_override=0.50),
    "BOOM500":   SymbolTune(alert_cooldown_override=240.0,
                             spike_min_prob_override=0.45),
    "BOOM600N":  SymbolTune(alert_cooldown_override=270.0,
                             spike_min_prob_override=0.45),
    "BOOM900":   SymbolTune(alert_cooldown_override=360.0,
                             spike_min_prob_override=0.42),
    "BOOM1000":  SymbolTune(alert_cooldown_override=400.0,
                             spike_min_prob_override=0.40),
    "CRASH300N": SymbolTune(alert_cooldown_override=180.0,
                             spike_min_prob_override=0.50),
    "CRASH500":  SymbolTune(alert_cooldown_override=240.0,
                             spike_min_prob_override=0.45),
    "CRASH600N": SymbolTune(alert_cooldown_override=270.0,
                             spike_min_prob_override=0.45),
    "CRASH900":  SymbolTune(alert_cooldown_override=360.0,
                             spike_min_prob_override=0.42),
    "CRASH1000": SymbolTune(alert_cooldown_override=400.0,
                             spike_min_prob_override=0.40),
    "JD10":  SymbolTune(alert_cooldown_override=60.0,
                         mc_paths_override=8_000,
                         drift_tstat_override=1.8,
                         drift_window_override=100),
    "JD25":  SymbolTune(alert_cooldown_override=90.0,
                         drift_tstat_override=1.8),
    "JD50":  SymbolTune(alert_cooldown_override=120.0),
    "JD75":  SymbolTune(alert_cooldown_override=150.0),
    "JD100": SymbolTune(alert_cooldown_override=180.0),
}

def get_profile(sym: str) -> AssetProfile:
    cat = _cat(sym)
    return {
        "vol":   PROFILE_VOL,  "boom":  PROFILE_BOOM,
        "crash": PROFILE_CRASH,"jump":  PROFILE_JUMP,
        "step":  PROFILE_STEP,
    }.get(cat, PROFILE_VOL)

def get_tune(sym: str) -> SymbolTune:
    return SYMBOL_TUNES.get(sym, SymbolTune())

def effective_profile(sym: str) -> AssetProfile:
    import copy
    p    = copy.deepcopy(get_profile(sym))
    tune = get_tune(sym)
    if tune.alert_cooldown_override    is not None:
        p.alert_cooldown        = tune.alert_cooldown_override
    if tune.min_target_pct_override    is not None:
        p.min_target_pct        = tune.min_target_pct_override
    if tune.min_rr_override            is not None:
        p.min_rr = p.alert_min_rr = tune.min_rr_override
    if tune.spike_min_prob_override    is not None:
        p.spike_min_prob        = tune.spike_min_prob_override
    if tune.mc_paths_override          is not None:
        p.mc_paths              = tune.mc_paths_override
    if tune.sigma_mult_override        is not None:
        p.sigma_multiplier_base = tune.sigma_mult_override
    if tune.drift_tstat_override       is not None:
        p.drift_tstat_min       = tune.drift_tstat_override
    if tune.drift_window_override      is not None:
        p.drift_window          = tune.drift_window_override
    if tune.vol_k_min_override         is not None:
        p.vol_k_min             = tune.vol_k_min_override
    if tune.vol_k_max_override         is not None:
        p.vol_k_max             = tune.vol_k_max_override
    return p

# ============================================================================
# NUMBA KERNELS
# ============================================================================
@njit(parallel=True, cache=False, fastmath=True)
def _gbm_kernel(S0, mu, sigma, dt, n_paths, n_steps, Z):
    paths = np.empty((n_paths, n_steps+1), dtype=np.float64)
    drift = (mu - 0.5*sigma*sigma)*dt
    vol   = sigma*math.sqrt(dt)
    for i in prange(n_paths):
        paths[i,0] = S0
        for t in range(n_steps):
            paths[i,t+1] = paths[i,t]*math.exp(drift + vol*Z[i,t])
    return paths

@njit(parallel=True, cache=False, fastmath=True)
def _jd_kernel(S0, mu, sigma, lam, jmean, jstd, jsign,
               dt, n_paths, n_steps, Z, U, Zj):
    paths  = np.empty((n_paths, n_steps+1), dtype=np.float64)
    drift  = (mu - 0.5*sigma*sigma)*dt
    vol    = sigma*math.sqrt(dt)
    lam_dt = lam*dt
    for i in prange(n_paths):
        paths[i,0] = S0
        for t in range(n_steps):
            gbm  = drift + vol*Z[i,t]
            jump = 0.0
            if U[i,t] < lam_dt:
                raw  = jmean + jstd*abs(Zj[i,t])
                jump = jsign*raw
            paths[i,t+1] = paths[i,t]*math.exp(gbm+jump)
    return paths

# ============================================================================
# DATA CLASSES
# ============================================================================
@dataclass
class GBMParams:
    mu: float = 0.0
    sigma: float = 0.01
    sigma_ewma: float = 0.01
    n_obs: int = 0
    advertised_vol: float = 0.0
    ks_pvalue: float = 1.0
    fit_quality: float = 1.0

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
    mu: float = 0.0
    sigma: float = 0.01
    lam: float = 0.002
    lam_posterior: float = 0.002
    jump_mean: float = 0.02
    jump_std: float = 0.01
    jump_sign: float = 1.0
    n_jumps: int = 0
    n_obs: int = 0
    expected_lam: float = 0.002
    lam_deviation: float = 0.0
    recent_cluster: float = 0.0
    inter_arrival_cv: float = 1.0
    hazard_intensity: float = 0.0
    counter_drift: float = 0.0
    spike_mag_p60: float = 0.02
    spike_mag_p75: float = 0.03

# ============================================================================
# v4.5 SPIKE SUPPRESSION RESULT
# ============================================================================
@dataclass
class SpikeSuppressionResult:
    """
    Holds the 5-factor quality assessment for a spike alert.

    IMPORTANT: This is INFORMATIONAL only — exactly like
    VolSignalQuality for directional signals.

    The alert ALWAYS fires if raw_confidence >= SPIKE_MIN_CONFIDENCE.
    Stars and warnings are shown IN the alert message so the
    user can make their own decision — high stars act, low stars watch.

    This mirrors the VolSignalQuality pattern exactly.
    """
    # Star rating 1-5 shown to user
    stars:               int

    # Adjusted confidence for DISPLAY (not used as gate)
    adjusted_confidence: float
    raw_confidence:      float

    # Individual factor multipliers (for logging/display)
    drought_factor:      float
    compression_factor:  float
    activity_factor:     float
    cooling_factor:      float
    range_factor:        float

    # User-facing content
    warnings:            List[str]
    confirmations:       List[str]

    # Context values for display
    ticks_since_spike:   int
    activity_ratio:      float
    vol_deviation:       float
    range_ratio:         float

    # Is the generator currently in a cooling period?
    # (still shown in alert, not a gate)
    in_cooling:          bool

# ============================================================================
# v4.5 VOL SIGNAL QUALITY RESULT
# ============================================================================
@dataclass
class VolSignalQuality:
    """
    5-factor quality assessment for directional signals.
    Informational only — attached to alert message.
    Never blocks a signal.
    """
    stars:               int
    regime_consistent:   bool
    momentum_persistent: float
    drift_noise_ratio:   float
    regime_supports:     bool
    tf_agreement:        float
    regime_label:        str
    persistence_label:   str
    drift_label:         str
    regime_support_label:str
    tf_agreement_label:  str
    warnings:            List[str]
    confirmations:       List[str]

@dataclass
class SpikeAlert:
    symbol:           str
    fname:            str
    direction:        str
    confidence:       float
    poisson_prob:     float
    hazard_intensity: float
    ticks_lo:         int
    ticks_hi:         int
    time_lo_str:      str
    time_hi_str:      str
    time_center_str:  str
    magnitude_lo:     float
    magnitude_hi:     float
    magnitude_p60:    float
    magnitude_p75:    float
    reasons:          List[str]
    is_imminent:      bool  = True
    counter_drift:    float = 0.0
    tod_multiplier:   float = 1.0
    timestamp:        float = field(default_factory=time.time)
    resolved:         bool  = False
    was_correct:      bool  = False
    tick_index_at_alert: int = 0
    # v4.5: quality info attached after alert passes raw gate
    suppression: Optional[SpikeSuppressionResult] = None

@dataclass
class TradeSetup:
    direction:       str
    entry:           float
    target:          float
    invalidation:    float
    target_pct:      float
    stop_pct:        float
    rr_ratio:        float
    prob_target:     float
    prob_stop:       float
    edge_pct:        float
    expected_value:  float
    signal_strength: str
    timeframe_label: str
    horizon_ticks:   int
    regime:          str
    profile_name:    str   = "UNKNOWN"
    signal_source:   str   = "MC"
    tod_multiplier:  float = 1.0
    quality: Optional[VolSignalQuality] = None

@dataclass
class MCResult:
    symbol:          str
    S0:              float
    horizon:         int
    horizon_label:   str
    paths:           np.ndarray
    p5:              np.ndarray
    p25:             np.ndarray
    p50:             np.ndarray
    p75:             np.ndarray
    p95:             np.ndarray
    target_up:       float
    target_down:     float
    target_label:    str
    prob_up_horizon: float
    prob_hit_up:     float
    prob_hit_down:   float
    sigma_horizon:   float
    ci_width_pct:    float
    prob_jump:       Optional[float] = None
    exp_jump_tick:   Optional[float] = None
    jump_magnitude_lo: Optional[float] = None
    jump_magnitude_hi: Optional[float] = None
    edge_score:      float = 0.0
    signal:          str   = "neutral"
    implied_edge_pct:float = 0.0
    trade_setup:     Optional[TradeSetup] = None
    timeframe_key:   Optional[str] = None
    regime:          str   = "NORMAL"
    profile_name:    str   = "UNKNOWN"
    drift_signal:    Optional[DriftSignal] = None
    vol_regime:      Optional[VolRegimeResult] = None
    tod_multiplier:  float = 1.0
    ens_agreement:   float = 1.0
    ens_n_runs:      int   = 1

@dataclass
class PatternRecord:
    symbol:       str
    horizon:      int
    timestamp:    float
    prediction:   str
    prob:         float
    edge_score:   float
    S0:           float
    target_up:    float
    target_down:  float
    resolved:     bool  = False
    correct:      bool  = False
    outcome_pct:  float = 0.0
    result_label: str   = ""

@dataclass
class UserState:
    pending_sym:    Optional[str] = None
    horizon_key:    str   = "300t"
    timeframe_key:  str   = "5m"
    watchlist:      List[str] = field(default_factory=list)
    risk_pct:       float = 1.0
    alerts_enabled: bool  = False

# ============================================================================
# v4.5 SPIKE SUPPRESSION ANALYSER
# ============================================================================
class SpikeSuppressionAnalyser:
    """
    5-factor quality assessment for spike alerts.
    Works identically to VolSignalQuality:
      — Always produces a star rating (1-5)
      — Always produces warnings and confirmations
      — Result is ATTACHED to the alert message
      — Result NEVER blocks the alert
      — User reads stars and decides how to act

    Key fix vs broken v4.5:
      — Stars/warnings are purely informational
      — The only spike gate is raw_confidence >= SPIKE_MIN_CONFIDENCE
      — Drought is initialised from backfill history on start
    """

    def __init__(self):
        # sym -> tick index of last confirmed spike
        # Initialised from backfill in DataManager
        self._last_spike_tick: Dict[str, int] = {}

    def initialise_from_history(
        self, sym: str, prices: np.ndarray, lr: np.ndarray
    ):
        """
        Called after backfill completes.
        Scans historical log-returns to find the most recent
        spike, so drought tracking starts correctly from restart.
        """
        if len(lr) < 20:
            return
        try:
            med = float(np.median(lr))
            mad = max(
                float(np.median(np.abs(lr - med))) * 1.4826,
                1e-10,
            )
            is_spike = np.abs(lr - med) > JUMP_THRESHOLD * mad
            spike_indices = np.where(is_spike)[0]
            if len(spike_indices) > 0:
                last_spike_offset = int(spike_indices[-1])
                # tick_index is relative to current buffer
                self._last_spike_tick[sym] = last_spike_offset
                ticks_ago = len(lr) - last_spike_offset
                log.info(
                    f"SpikeSuppress: init {_friendly(sym)} — "
                    f"last spike {ticks_ago} ticks ago "
                    f"(idx={last_spike_offset})"
                )
            else:
                # No spike found in history — set to zero
                # so drought = full history length (conservative)
                log.info(
                    f"SpikeSuppress: init {_friendly(sym)} — "
                    f"no spike in backfill history"
                )
        except Exception as e:
            log.warning(f"SpikeSuppress init {sym}: {e}")

    def record_spike(self, sym: str, tick_idx: int):
        """Called when a live spike is detected in _on_tick."""
        self._last_spike_tick[sym] = tick_idx
        log.info(
            f"SpikeSuppress: live spike recorded "
            f"{_friendly(sym)} @ tick {tick_idx}"
        )

    def analyse(
        self,
        sym:            str,
        jp:             JumpParams,
        raw_confidence: float,
        prices:         np.ndarray,
        current_tick:   int,
        vol_deviation:  float,
        lr:             np.ndarray,
    ) -> SpikeSuppressionResult:
        """
        Runs 5-factor analysis and returns a SpikeSuppressionResult.
        Result is for display only — does NOT gate the alert.
        """
        warnings_list:      List[str] = []
        confirmations_list: List[str] = []
        exp_freq = EXPECTED_JUMP_FREQ.get(sym, 500)

        # ------------------------------------------------------------------
        # Factor 1: Drought
        # ------------------------------------------------------------------
        last_tick    = self._last_spike_tick.get(sym, -1)
        if last_tick >= 0:
            ticks_since  = max(current_tick - last_tick, 0)
        else:
            # No history record — use n_obs as proxy
            ticks_since  = jp.n_obs

        drought_ratio = ticks_since / max(exp_freq, 1)

        if drought_ratio >= 5.0:
            drought_factor = DROUGHT_DISCOUNT_5X
            warnings_list.append(
                f"⚠️ Long drought: {ticks_since:,} ticks since last spike "
                f"({drought_ratio:.1f}× expected interval). "
                f"Generator may be in suppressed phase — "
                f"spike overdue but arrival timing uncertain."
            )
        elif drought_ratio >= 3.0:
            drought_factor = DROUGHT_DISCOUNT_3X
            warnings_list.append(
                f"⚠️ Extended quiet: {ticks_since:,} ticks "
                f"({drought_ratio:.1f}× expected). "
                f"Below-average recent activity."
            )
        elif drought_ratio >= 2.0:
            drought_factor = DROUGHT_DISCOUNT_2X
            warnings_list.append(
                f"⚠️ Moderate quiet: {ticks_since:,} ticks "
                f"({drought_ratio:.1f}× expected)."
            )
        elif drought_ratio <= 1.0:
            drought_factor = 1.0
            confirmations_list.append(
                f"✅ Within normal interval: "
                f"{ticks_since:,} ticks "
                f"(expected ~{exp_freq})."
            )
        else:
            drought_factor = 1.0

        # ------------------------------------------------------------------
        # Factor 2: Volatility compression
        # ------------------------------------------------------------------
        if vol_deviation < VOL_COMPRESSION_STRONG:
            compression_factor = VOL_COMPRESSION_DISCOUNT_STRONG
            warnings_list.append(
                f"⚠️ Vol STRONGLY compressed "
                f"({vol_deviation*100:+.1f}% vs target). "
                f"Generator in low-energy state — "
                f"spike may be smaller than typical."
            )
        elif vol_deviation < VOL_COMPRESSION_MILD:
            compression_factor = VOL_COMPRESSION_DISCOUNT_MILD
            warnings_list.append(
                f"⚠️ Vol compressed "
                f"({vol_deviation*100:+.1f}% vs target). "
                f"Quiet market conditions."
            )
        elif vol_deviation > 0.20:
            compression_factor = 1.05
            confirmations_list.append(
                f"✅ Vol EXPANDED "
                f"({vol_deviation*100:+.1f}% above target). "
                f"Active generator — spike may be larger."
            )
        else:
            compression_factor = 1.0
            confirmations_list.append(
                f"✅ Vol near target "
                f"({vol_deviation*100:+.1f}%). "
                f"Normal conditions."
            )

        # ------------------------------------------------------------------
        # Factor 3: Recent activity ratio
        # ------------------------------------------------------------------
        lam_mle        = jp.lam
        recent_cluster = jp.recent_cluster
        activity_ratio = (
            recent_cluster / max(lam_mle, 1e-10)
            if lam_mle > 1e-10 else 1.0
        )

        if activity_ratio < ACTIVITY_RATIO_VERY_QUIET:
            activity_factor = ACTIVITY_DISCOUNT_VERY_QUIET
            warnings_list.append(
                f"⚠️ Recent spike rate VERY LOW "
                f"({activity_ratio:.2f}× long-run average). "
                f"Generator currently inactive."
            )
        elif activity_ratio < ACTIVITY_RATIO_QUIET:
            activity_factor = ACTIVITY_DISCOUNT_QUIET
            warnings_list.append(
                f"⚠️ Recent spike rate below average "
                f"({activity_ratio:.2f}× long-run)."
            )
        elif activity_ratio > 1.2:
            activity_factor = 1.05
            confirmations_list.append(
                f"✅ Spike rate ELEVATED "
                f"({activity_ratio:.2f}× long-run). "
                f"Active spike regime."
            )
        else:
            activity_factor = 1.0
            confirmations_list.append(
                f"✅ Activity near average "
                f"({activity_ratio:.2f}×)."
            )

        # ------------------------------------------------------------------
        # Factor 4: Post-spike cooling
        # ------------------------------------------------------------------
        cooling_threshold = exp_freq * POST_SPIKE_COOLING_FRACTION
        in_cooling        = (
            last_tick >= 0 and ticks_since < cooling_threshold
        )

        if in_cooling:
            cooling_factor = 0.30   # soft discount, not zero
            warnings_list.append(
                f"⚠️ POST-SPIKE COOLING: Only {ticks_since} ticks "
                f"since last spike. Next spike expected in "
                f"~{max(int(cooling_threshold - ticks_since), 0)} "
                f"more ticks. Probability is low right now."
            )
        else:
            cooling_factor = 1.0
            if last_tick >= 0:
                confirmations_list.append(
                    f"✅ Past cooling period "
                    f"({ticks_since} ticks since last spike, "
                    f"threshold {int(cooling_threshold)})."
                )

        # ------------------------------------------------------------------
        # Factor 5: Price range compression
        # ------------------------------------------------------------------
        range_factor = 1.0
        range_ratio  = 1.0
        if len(prices) >= 101:
            recent_range = float(
                np.max(prices[-50:]) - np.min(prices[-50:])
            )
            prior_range  = float(
                np.max(prices[-100:-50]) - np.min(prices[-100:-50])
            )
            if prior_range > 1e-10:
                range_ratio = recent_range / prior_range
                if range_ratio < RANGE_COMPRESSION_RATIO:
                    range_factor = RANGE_COMPRESSION_DISCOUNT
                    warnings_list.append(
                        f"⚠️ Price range CONTRACTING "
                        f"(ratio={range_ratio:.2f}). "
                        f"Low-activity conditions — "
                        f"wait for range expansion."
                    )
                elif range_ratio > RANGE_EXPANSION_RATIO:
                    range_factor = RANGE_EXPANSION_BOOST
                    confirmations_list.append(
                        f"✅ Price range EXPANDING "
                        f"(ratio={range_ratio:.2f}). "
                        f"Favourable spike conditions."
                    )
                else:
                    confirmations_list.append(
                        f"✅ Price range stable "
                        f"(ratio={range_ratio:.2f})."
                    )

        # ------------------------------------------------------------------
        # Combined display confidence (NOT a gate)
        # ------------------------------------------------------------------
        combined_mult = float(np.clip(
            drought_factor
            * compression_factor
            * activity_factor
            * cooling_factor
            * range_factor,
            0.0, 1.10,
        ))
        adjusted_confidence = float(np.clip(
            raw_confidence * combined_mult, 0.0, 1.0
        ))

        # ------------------------------------------------------------------
        # Star rating — based on how many factors are favourable
        # ------------------------------------------------------------------
        positive_factors = 0
        if drought_ratio <= 1.5:         positive_factors += 1
        if vol_deviation >= -0.10:       positive_factors += 1
        if activity_ratio >= 0.70:       positive_factors += 1
        if not in_cooling:               positive_factors += 1
        if range_ratio >= 0.80:          positive_factors += 1

        stars = max(1, positive_factors)

        return SpikeSuppressionResult(
            stars               = stars,
            adjusted_confidence = adjusted_confidence,
            raw_confidence      = raw_confidence,
            drought_factor      = drought_factor,
            compression_factor  = compression_factor,
            activity_factor     = activity_factor,
            cooling_factor      = cooling_factor,
            range_factor        = range_factor,
            warnings            = warnings_list,
            confirmations       = confirmations_list,
            ticks_since_spike   = ticks_since,
            activity_ratio      = activity_ratio,
            vol_deviation       = vol_deviation,
            range_ratio         = range_ratio,
            in_cooling          = in_cooling,
        )

    @staticmethod
    def format_quality_block(
        sr: SpikeSuppressionResult,
    ) -> str:
        """
        Formats quality analysis into user-readable block.
        Appended to spike alert message — mirrors the
        VolSignalQuality format_quality_block exactly.
        Always shown. Never blocks.
        """
        lines = [
            f"\n*Spike Quality: {_stars(sr.stars)} "
            f"({sr.stars}/5)*\n"
        ]

        # Confirmations first (good news)
        if sr.confirmations:
            for c in sr.confirmations:
                lines.append(c)

        # Warnings after (caution items)
        if sr.warnings:
            if sr.confirmations:
                lines.append("")
            for w in sr.warnings:
                lines.append(w)

        # Show adjusted vs raw confidence as context
        if abs(sr.adjusted_confidence - sr.raw_confidence) > 0.02:
            disc = (sr.raw_confidence - sr.adjusted_confidence) * 100
            lines.append(
                f"\nCondition-adjusted confidence: "
                f"`{sr.raw_confidence:.0%}` → "
                f"`{sr.adjusted_confidence:.0%}` "
                f"(-{disc:.0f}pp from market conditions)"
            )

        # Action guidance based on stars
        lines.append("")
        if sr.stars >= 4:
            lines.append(
                "*Conditions FAVOURABLE* ✅ — "
                "Signal credible. Act with normal position size."
            )
        elif sr.stars == 3:
            lines.append(
                "*Conditions MODERATE* ⚠️ — "
                "Some caution advised. "
                "Consider reduced position size."
            )
        elif sr.stars == 2:
            lines.append(
                "*Conditions UNFAVOURABLE* 🔴 — "
                "Multiple suppression factors active. "
                "WATCH and wait for range expansion before acting."
            )
        else:
            lines.append(
                "*Conditions POOR* 🔴 — "
                "Strong suppression signals. "
                "Treat as background information only."
            )

        return "\n".join(lines)

# ============================================================================
# v4.5 VOL SIGNAL QUALITY ANALYSER
# ============================================================================
class VolSignalQualityAnalyser:
    """
    5-factor quality assessment for directional signals.
    Always informational — never blocks a signal.
    """

    def __init__(self):
        self._snapshots: Dict[str, deque] = {}

    def _update_snapshot(self, sym: str, direction: str):
        if sym not in self._snapshots:
            self._snapshots[sym] = deque(maxlen=PERSISTENCE_WINDOW)
        self._snapshots[sym].append((time.time(), direction))

    def analyse(
        self,
        sym:            str,
        lr:             np.ndarray,
        prices:         np.ndarray,
        signal_dir:     str,
        vol_regime:     Optional[VolRegimeResult],
        drift:          Optional[DriftSignal],
        sigma_tick:     float,
        drift_per_tick: float,
        profile:        AssetProfile,
    ) -> VolSignalQuality:
        warnings_list:      List[str] = []
        confirmations_list: List[str] = []
        r = lr[np.isfinite(lr)] if len(lr) > 0 else lr

        self._update_snapshot(sym, signal_dir)

        # Factor 1: Regime consistency
        regimes = []
        vm_tmp  = VolatilityModel()
        for window in VOL_QUALITY_WINDOWS:
            if len(r) >= window:
                regimes.append(vm_tmp.detect_regime(r[-window:]))
            else:
                regimes.append("UNKNOWN")

        valid_regimes     = [g for g in regimes if g != "UNKNOWN"]
        regime_consistent = (
            len(set(valid_regimes)) <= 1 and len(valid_regimes) >= 2
        )
        if regime_consistent:
            regime_label = "STABLE"
            confirmations_list.append(
                f"✅ Regime STABLE across windows "
                f"({valid_regimes[0] if valid_regimes else 'N/A'})."
            )
        else:
            regime_label = "TRANSITIONING"
            warnings_list.append(
                f"⚠️ Regime TRANSITIONING: "
                f"{' → '.join(valid_regimes)}. "
                f"Signal reliability reduced."
            )

        # Factor 2: Momentum persistence
        snaps = list(self._snapshots.get(sym, []))
        if len(snaps) >= 5:
            persistence = sum(
                1 for _, d in snaps if d == signal_dir
            ) / len(snaps)
        else:
            persistence = 0.5

        if persistence >= PERSISTENCE_MIN_AGREEMENT:
            persistence_label = "HIGH"
            confirmations_list.append(
                f"✅ Momentum PERSISTENT: "
                f"{persistence:.0%} of recent snapshots "
                f"confirm {signal_dir.upper()}."
            )
        elif persistence >= 0.50:
            persistence_label = "MODERATE"
            warnings_list.append(
                f"⚠️ Momentum MODERATE "
                f"({persistence:.0%} agreement). "
                f"May be oscillating around threshold."
            )
        else:
            persistence_label = "LOW"
            warnings_list.append(
                f"⚠️ Momentum UNSTABLE "
                f"({persistence:.0%} agreement). "
                f"Likely threshold noise."
            )

        # Factor 3: Drift/noise ratio
        drift_noise_ratio = (
            abs(drift_per_tick) / max(sigma_tick, 1e-10)
        )
        if drift_noise_ratio >= DRIFT_NOISE_MIN_RATIO:
            drift_label = "TRADEABLE"
            confirmations_list.append(
                f"✅ Drift/noise ratio `{drift_noise_ratio:.4f}` — "
                f"economically meaningful."
            )
        else:
            drift_label = "TOO_SMALL"
            warnings_list.append(
                f"⚠️ Drift/noise ratio `{drift_noise_ratio:.4f}` "
                f"too small. Move may not reach target reliably."
            )

        # Factor 4: Regime type supports direction
        current_regime = (
            vol_regime.regime if vol_regime
            else (drift.regime if drift else "NORMAL")
        )
        if current_regime == "TRENDING":
            regime_support_label = "STRONGLY_SUPPORTING"
            regime_supports      = True
            confirmations_list.append(
                "✅ Regime TRENDING — highest reliability "
                "for directional signals."
            )
        elif current_regime in ("VOLATILE","EXPANDED"):
            regime_support_label = "SUPPORTING"
            regime_supports      = True
            confirmations_list.append(
                f"✅ Regime {current_regime} — supports direction."
            )
        elif current_regime == "RANGING":
            regime_support_label = "OPPOSING"
            regime_supports      = False
            warnings_list.append(
                "⚠️ Regime RANGING — directional signals "
                "are unreliable in ranging markets."
            )
        elif current_regime == "COMPRESSED":
            regime_support_label = "OPPOSING"
            regime_supports      = False
            warnings_list.append(
                "⚠️ Regime COMPRESSED — vol too low "
                "for move to reach target reliably."
            )
        else:
            regime_support_label = "NEUTRAL"
            regime_supports      = True

        # Factor 5: Cross-timeframe agreement
        tf_directions = []
        for window in VOL_QUALITY_WINDOWS:
            if len(r) < window:
                continue
            seg     = r[-window:]
            mu_seg  = float(np.mean(seg))
            std_seg = max(float(np.std(seg, ddof=1)), 1e-10)
            tstat   = mu_seg / (std_seg / math.sqrt(len(seg)))
            if tstat > 1.0:
                tf_directions.append("bullish")
            elif tstat < -1.0:
                tf_directions.append("bearish")
            else:
                tf_directions.append("neutral")

        agree_tf = (
            sum(1 for d in tf_directions if d == signal_dir)
            / len(tf_directions)
            if tf_directions else 0.5
        )
        if agree_tf >= 0.80:
            tf_agreement_label = "STRONG"
            confirmations_list.append(
                f"✅ {agree_tf:.0%} of timeframe windows "
                f"confirm {signal_dir.upper()}."
            )
        elif agree_tf >= 0.55:
            tf_agreement_label = "MODERATE"
            warnings_list.append(
                f"⚠️ {agree_tf:.0%} of timeframe windows agree. "
                f"Mixed cross-timeframe evidence."
            )
        else:
            tf_agreement_label = "WEAK"
            warnings_list.append(
                f"⚠️ Only {agree_tf:.0%} of timeframe windows agree. "
                f"Signal may be short-window only."
            )

        # Star rating
        score = 0
        if regime_consistent:                                score += 1
        if persistence >= 0.60:                              score += 1
        if drift_noise_ratio >= DRIFT_NOISE_MIN_RATIO:       score += 1
        if regime_supports:                                  score += 1
        if agree_tf >= 0.60:                                 score += 1
        stars = max(1, score)

        return VolSignalQuality(
            stars               = stars,
            regime_consistent   = regime_consistent,
            momentum_persistent = persistence,
            drift_noise_ratio   = drift_noise_ratio,
            regime_supports     = regime_supports,
            tf_agreement        = agree_tf,
            regime_label        = regime_label,
            persistence_label   = persistence_label,
            drift_label         = drift_label,
            regime_support_label= regime_support_label,
            tf_agreement_label  = tf_agreement_label,
            warnings            = warnings_list,
            confirmations       = confirmations_list,
        )

    @staticmethod
    def format_quality_block(vq: VolSignalQuality) -> str:
        lines = [
            f"\n*Signal Quality: {_stars(vq.stars)} "
            f"({vq.stars}/5)*\n"
        ]
        if vq.confirmations:
            for c in vq.confirmations:
                lines.append(c)
        if vq.warnings:
            if vq.confirmations:
                lines.append("")
            for w in vq.warnings:
                lines.append(w)
        lines.append("")
        if vq.stars >= 4:
            lines.append(
                "*Quality HIGH* ✅ — "
                "Proceed with standard position size."
            )
        elif vq.stars == 3:
            lines.append(
                "*Quality MODERATE* ⚠️ — "
                "Consider half position size."
            )
        elif vq.stars == 2:
            lines.append(
                "*Quality LOW* 🔴 — "
                "WATCH only. Monitor for improvement."
            )
        else:
            lines.append(
                "*Quality POOR* 🔴 — "
                "Signal unreliable. Do not trade."
            )
        return "\n".join(lines)

# ============================================================================
# ADAPTIVE THRESHOLDS
# ============================================================================
class AdaptiveThresholds:
    def __init__(self, pe: "PersistenceEngine"):
        self._pe   = pe
        self._data: Dict[str, dict] = pe.get(
            "adaptive_thresholds_v45", {}
        )

    def _sym_data(self, sym: str) -> dict:
        if sym not in self._data:
            self._data[sym] = {
                "prob_alpha": ADAPTIVE_PRIOR_STRENGTH * MIN_PROB_TARGET,
                "prob_beta":  ADAPTIVE_PRIOR_STRENGTH * (1 - MIN_PROB_TARGET),
                "rr_alpha":   ADAPTIVE_PRIOR_STRENGTH * 0.40,
                "rr_beta":    ADAPTIVE_PRIOR_STRENGTH * 0.60,
                "n_updates":  0,
            }
        return self._data[sym]

    def update(self, sym: str, won: bool,
               prob_used: float, rr_used: float):
        d  = self._sym_data(sym)
        lr = ADAPTIVE_LEARNING_RATE
        if won:
            d["prob_alpha"] += lr
            d["rr_alpha"]   += lr
        else:
            d["prob_beta"]  += lr
            d["rr_beta"]    += lr
        d["n_updates"] += 1
        if d["n_updates"] % 10 == 0:
            self._save()

    def get_prob_threshold(self, sym: str) -> float:
        d   = self._sym_data(sym)
        raw = d["prob_alpha"] / (d["prob_alpha"] + d["prob_beta"])
        return float(np.clip(raw, ADAPTIVE_MIN_PROB, ADAPTIVE_MAX_PROB))

    def get_rr_threshold(self, sym: str) -> float:
        d        = self._sym_data(sym)
        raw_norm = d["rr_alpha"] / (d["rr_alpha"] + d["rr_beta"])
        rr = ADAPTIVE_MIN_RR + raw_norm * (
            ADAPTIVE_MAX_RR - ADAPTIVE_MIN_RR
        )
        return float(np.clip(rr, ADAPTIVE_MIN_RR, ADAPTIVE_MAX_RR))

    def _save(self):
        self._pe.set("adaptive_thresholds_v45", self._data)

    def summary(self, sym: str) -> str:
        d = self._sym_data(sym)
        return (
            f"AdaptThresh[{_friendly(sym)}]: "
            f"P≥{self.get_prob_threshold(sym):.4f} "
            f"RR≥{self.get_rr_threshold(sym):.3f} "
            f"(n={d['n_updates']})"
        )

# ============================================================================
# PERSISTENCE ENGINE
# ============================================================================
MAX_PATTERN_RECORDS  = 10_000
PATTERN_MAX_AGE_DAYS = 30

class PersistenceEngine:
    def __init__(self):
        self.data: dict = {}
        self._load()

    def _load(self):
        for fn in (
            PERSIST_FILE,
            "sqe_v44_state.pkl","sqe_v43_state.pkl",
            "sqe_v42_state.pkl",
        ):
            if os.path.exists(fn):
                try:
                    with open(fn,"rb") as f:
                        self.data = pickle.load(f)
                    log.info(f"Persistence: loaded {fn}")
                    return
                except Exception:
                    self.data = {}

    def save(self):
        try:
            with open(PERSIST_FILE,"wb") as f:
                pickle.dump(self.data, f)
        except Exception:
            pass

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

# ============================================================================
# PATTERN MEMORY
# ============================================================================
class PatternMemory:
    def __init__(self,
                 tod_profile: Optional[TimeOfDayProfile] = None):
        self.records: List[PatternRecord] = []
        self._tod = tod_profile
        self._load()

    def _load(self):
        for fn in (
            PATTERN_FILE,
            "pattern_memory_v44.pkl",
            "pattern_memory_v43.pkl",
        ):
            if os.path.exists(fn):
                try:
                    with open(fn,"rb") as f:
                        self.records = pickle.load(f)
                    log.info(
                        f"Pattern memory: {len(self.records)} records."
                    )
                    return
                except Exception:
                    self.records = []

    def _save(self):
        cutoff = time.time() - PATTERN_MAX_AGE_DAYS * 86400
        self.records = [
            r for r in self.records
            if not r.resolved or r.timestamp > cutoff
        ][-MAX_PATTERN_RECORDS:]
        try:
            with open(PATTERN_FILE,"wb") as f:
                pickle.dump(self.records, f)
        except Exception:
            pass

    def record(self, sym, horizon, pred, prob,
               edge, S0, tup, tdn) -> PatternRecord:
        r = PatternRecord(
            symbol=sym, horizon=horizon,
            timestamp=time.time(), prediction=pred,
            prob=prob, edge_score=edge, S0=S0,
            target_up=tup, target_down=tdn,
            result_label="PENDING",
        )
        self.records.append(r)
        self._save()
        return r

    def resolve(self, sym: str, price: float,
                bot=None, chats: Optional[Set[int]] = None,
                adaptive: Optional[AdaptiveThresholds] = None):
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
            thresh = profile.resolve_threshold_pct / 100
            if r.prediction == "bullish":
                r.correct = price >= r.target_up
            elif r.prediction == "bearish":
                r.correct = price <= r.target_down
            elif r.prediction in ("spike_imminent","jump_imminent"):
                r.correct = abs(r.outcome_pct / 100) > thresh
            else:
                r.correct = False
            r.result_label = "WIN" if r.correct else "LOSS"
            if self._tod:
                bucket = int((r.timestamp % 86400) / 1800)
                self._tod.update(
                    sym, r.correct, r.edge_score,
                    r.outcome_pct, 0.0, 0.5,
                    bucket=bucket % TOD_BUCKETS,
                )
            if adaptive is not None:
                adaptive.update(sym, r.correct, r.prob, 1.4)
            if bot and chats:
                icon = "✅ WIN" if r.correct else "❌ LOSS"
                msg  = (
                    f"*Signal Outcome — {_friendly(sym)}*\n\n"
                    f"Result: *{icon}*\n"
                    f"Prediction: `{r.prediction.upper()}`\n"
                    f"Entry: `{r.S0:.6f}`\n"
                    f"Exit:  `{price:.6f}`\n"
                    f"Move:  `{r.outcome_pct:+.4f}%`"
                )
                for cid in list(chats):
                    asyncio.ensure_future(
                        _safe_send_message(bot, cid, msg)
                    )
        if changed:
            self._save()

    def stats(self, sym,
              horizon=None) -> Tuple[int, float, float]:
        done = [
            r for r in self.records
            if r.symbol == sym and r.resolved
            and (horizon is None
                 or abs(r.horizon - horizon) < 50)
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
        return float(np.clip(0.85 + (wr * 0.30), 0.70, 1.20))

    def note(self, sym, horizon=None) -> str:
        return ""

    def recent_results(self, sym: str, limit: int = 5) -> str:
        return ""

    def winning_moves(self, sym: str, is_bull: bool,
                      horizon: int) -> List[float]:
        return [
            abs(r.outcome_pct) / 100
            for r in self.records
            if r.symbol == sym
            and r.resolved and r.correct
            and (r.prediction == "bullish") == is_bull
            and abs(r.horizon - horizon) <= horizon * 0.5
            and r.outcome_pct != 0.0
        ]

# ============================================================================
# HYBRID + CONDITIONAL TARGET REFINER
# ============================================================================
class HybridTargetRefiner:
    def refine(self, sym, S0, is_bull, mc_target,
               mc_stop, finals, prices, horizon, records):
        n_records = len(
            [r for r in records
             if r.symbol == sym and r.resolved]
        )
        cold    = n_records < REFINER_MIN_PATTERNS
        emp_cap = _empirical_move_cap(prices, horizon)
        if cold:
            if emp_cap is not None:
                direction = 1.0 if is_bull else -1.0
                return (
                    S0 * (1 + direction * emp_cap),
                    S0 * (1 - direction * emp_cap * 0.40),
                    "empirical_cold",
                )
            return mc_target, mc_stop, "mc_fallback"
        try:
            kde      = gaussian_kde(finals)
            x        = np.linspace(finals.min(), finals.max(), 500)
            kde_mode = float(x[np.argmax(kde(x))])
            blended  = 0.5 * kde_mode + 0.5 * mc_target
            if emp_cap is not None:
                if abs(blended - S0) / S0 > emp_cap:
                    direction = 1.0 if is_bull else -1.0
                    blended   = S0 * (1 + direction * emp_cap)
            return blended, mc_stop, "kde_warm"
        except Exception:
            return mc_target, mc_stop, "mc_fallback"

class ConditionalTargetRefiner:
    MIN_WINS     = 20
    BLEND_WEIGHT = 0.40
    PERCENTILE   = 55

    def __init__(self, base: HybridTargetRefiner):
        self._base = base

    def refine(self, sym, S0, is_bull, mc_target, mc_stop,
               finals, prices, horizon, records,
               pm: Optional[PatternMemory] = None,
               ) -> Tuple[float, float, str]:
        base_target, base_stop, base_method = self._base.refine(
            sym, S0, is_bull, mc_target, mc_stop,
            finals, prices, horizon, records,
        )
        winning_moves: List[float] = (
            pm.winning_moves(sym, is_bull, horizon)
            if pm is not None else []
        )
        if len(winning_moves) < self.MIN_WINS:
            return base_target, base_stop, base_method + "+base"
        moves              = np.array(winning_moves)
        conditional_move   = float(
            np.percentile(moves, self.PERCENTILE)
        )
        conditional_target = (
            S0 * (1.0 + conditional_move) if is_bull
            else S0 * (1.0 - conditional_move)
        )
        emp_cap = _empirical_move_cap(prices, horizon)
        if emp_cap is not None:
            if abs(conditional_target - S0) / S0 > emp_cap:
                direction          = 1.0 if is_bull else -1.0
                conditional_target = S0 * (1.0 + direction * emp_cap)
        final_target = (
            self.BLEND_WEIGHT * conditional_target
            + (1.0 - self.BLEND_WEIGHT) * base_target
        )
        if abs(final_target - S0) / S0 * 100 < MIN_TARGET_MOVE * 100:
            return base_target, base_stop, base_method + "+belowmin"
        return final_target, base_stop, "cond_blend"
# SQE v4.5 — Part 2
# ============================================================================
# DATA MANAGER
# ============================================================================
class DataManager:
    def __init__(self):
        self.buffers: Dict[str, deque] = {
            s: deque(maxlen=BUFFER_SIZE) for s in ALL_SYMBOLS.values()
        }
        self.timestamps: Dict[str, deque] = {
            s: deque(maxlen=BUFFER_SIZE) for s in ALL_SYMBOLS.values()
        }
        self.latest: Dict[str, float] = {}
        self._tick_times: Dict[str, deque] = {
            s: deque(maxlen=300) for s in ALL_SYMBOLS.values()
        }
        self._tick_counts: Dict[str, int] = {
            s: 0 for s in ALL_SYMBOLS.values()
        }
        self._ws        = None
        self._running   = False
        self._connected = False
        self._init_done = False
        self._ready     = asyncio.Event()
        self._req_id    = 0
        self._id_lock   = asyncio.Lock()
        self._pending: Dict[int, asyncio.Future] = {}
        self._tick_cbs: List = []

    def add_tick_cb(self, fn):
        self._tick_cbs.append(fn)

    def prices(self, sym: str) -> np.ndarray:
        return np.array(list(self.buffers[sym]), dtype=np.float64)

    def log_returns(self, sym: str) -> np.ndarray:
        p = self.prices(sym)
        if len(p) < 2:
            return np.array([])
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.diff(np.log(p))
        return lr[np.isfinite(lr)]

    def last(self, sym: str) -> Optional[float]:
        return self.latest.get(sym)

    def n(self, sym: str) -> int:
        return len(self.buffers[sym])

    def tick_count(self, sym: str) -> int:
        return self._tick_counts.get(sym, 0)

    def observed_tick_rate(self, sym: str) -> float:
        times = list(self._tick_times[sym])
        if len(times) < 10:
            cat = _cat(sym)
            return {
                "boom": 0.5, "crash": 0.5,
                "jump": 1.0, "step":  1.0,
            }.get(cat, 0.5)
        diffs = np.diff(times)
        diffs = diffs[diffs > 0]
        if len(diffs) == 0:
            return 0.5
        return 1.0 / max(float(np.median(diffs)), 0.1)

    def ticks_for_timeframe(
        self, sym: str, tf_key: str
    ) -> Tuple[int, str]:
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

    async def _send_wait(
        self, payload: dict, timeout: float = 30.0
    ) -> dict:
        rid = await self._next_id()
        payload["req_id"] = rid
        loop = asyncio.get_running_loop()
        fut  = loop.create_future()
        self._pending[rid] = fut
        await self._ws.send(json.dumps(payload))
        try:
            return await asyncio.wait_for(
                asyncio.shield(fut), timeout=timeout
            )
        except asyncio.TimeoutError:
            self._pending.pop(rid, None)
            if not fut.done():
                fut.cancel()
            raise asyncio.TimeoutError(
                f"req_id={rid} timed out"
            )

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
                            self._tick_counts[sym] = (
                                self._tick_counts.get(sym, 0) + 1
                            )
                            self.latest[sym] = prc
                            for cb in self._tick_cbs:
                                asyncio.ensure_future(
                                    cb(sym, prc, ts)
                                )
                except Exception:
                    pass
        except Exception:
            pass

    async def _initialise(self, ssa: Optional["SpikeSuppressionAnalyser"] = None):
        try:
            await self._send_wait({"ping": 1}, timeout=8.0)
        except Exception:
            pass
        log.info("Backfilling tick history ...")
        sem = asyncio.Semaphore(3)

        async def _fill(sym: str):
            async with sem:
                try:
                    resp = await self._send_wait(
                        {
                            "ticks_history":     sym,
                            "count":             HISTORY_COUNT,
                            "end":               "latest",
                            "style":             "ticks",
                            "adjust_start_time": 1,
                        },
                        timeout=30.0,
                    )
                    if "error" in resp:
                        return
                    hist   = resp.get("history", {})
                    prices = hist.get("prices", [])
                    times  = hist.get("times",  [])
                    for p, t in zip(prices, times):
                        self.buffers[sym].append(float(p))
                        self.timestamps[sym].append(float(t))
                        self._tick_times[sym].append(float(t))
                        self._tick_counts[sym] = (
                            self._tick_counts.get(sym, 0) + 1
                        )
                    if prices:
                        self.latest[sym] = float(prices[-1])
                    fn = REVERSE_MAP.get(sym, sym)
                    log.info(
                        f"  OK {fn:<14} {len(prices):>5} ticks"
                    )
                    # v4.5: initialise spike suppression drought
                    # tracking from backfill data for Boom/Crash
                    cat = _cat(sym)
                    if (ssa is not None
                            and cat in ("boom", "crash")
                            and len(prices) >= 20):
                        prices_arr = self.prices(sym)
                        lr_arr     = self.log_returns(sym)
                        ssa.initialise_from_history(
                            sym, prices_arr, lr_arr
                        )
                except Exception as e:
                    log.warning(f"Backfill {sym}: {e}")

        await asyncio.gather(
            *[_fill(s) for s in ALL_SYMBOLS.values()]
        )
        total = sum(self.n(s) for s in ALL_SYMBOLS.values())
        log.info(f"Backfill complete — {total:,} ticks")

        for sym in ALL_SYMBOLS.values():
            rid = await self._next_id()
            try:
                await self._ws.send(json.dumps({
                    "ticks": sym, "subscribe": 1, "req_id": rid
                }))
                await asyncio.sleep(0.03)
            except Exception:
                pass
        self._init_done = True
        self._ready.set()
        log.info("Live subscriptions active OK")

    async def run(self, ssa: Optional["SpikeSuppressionAnalyser"] = None):
        self._running = True
        backoff       = 2.0
        while self._running:
            try:
                log.info(f"Connecting to {DERIV_WS_URL}")
                async with websockets.connect(
                    DERIV_WS_URL,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                    max_size=2**23,
                    extra_headers={
                        "Origin": "https://smarttrader.deriv.com"
                    },
                ) as ws:
                    self._ws        = ws
                    self._connected = True
                    self._init_done = False
                    self._ready.clear()
                    backoff = 2.0
                    log.info("WebSocket connected OK")
                    recv_task = asyncio.ensure_future(
                        self._receive_loop()
                    )
                    try:
                        await self._initialise(ssa=ssa)
                    except Exception as e:
                        log.warning(f"Initialise error: {e}")
                    await recv_task
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._connected = False
                self._ws        = None
                for fut in list(self._pending.values()):
                    if not fut.done():
                        fut.cancel()
                self._pending.clear()
                log.warning(
                    f"WS: {exc!r} — retry in {backoff:.1f}s"
                )
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

# ============================================================================
# VOLATILITY MODEL
# ============================================================================
class VolatilityModel:
    ADVERTISED = ADVERTISED_VOL

    def _ewma_sigma(
        self, r: np.ndarray, lam: float = EWMA_LAMBDA
    ) -> float:
        if len(r) < 2:
            return 0.01
        var = (
            float(np.var(r[:20], ddof=1))
            if len(r) >= 20
            else float(np.var(r, ddof=1))
        )
        if var <= 0:
            var = 1e-10
        for ret in r[20:]:
            var = lam * var + (1 - lam) * ret * ret
        return max(math.sqrt(var), 1e-10)

    def fit(self, sym: str, lr: np.ndarray) -> GBMParams:
        r = lr[np.isfinite(lr)]
        if len(r) < MIN_TICKS:
            return GBMParams(
                advertised_vol=self.ADVERTISED.get(sym, 0.0)
            )
        mu       = float(np.mean(r))
        sig      = max(float(np.std(r, ddof=1)), 1e-10)
        sig_ewma = self._ewma_sigma(r)
        try:
            ksp = float(kstest((r - mu) / sig, "norm").pvalue)
        except Exception:
            ksp = 1.0
        n_half = len(r) // 2
        sig1   = (
            float(np.std(r[:n_half], ddof=1))
            if n_half > 10 else sig
        )
        sig2   = (
            float(np.std(r[n_half:], ddof=1))
            if n_half > 10 else sig
        )
        stab = 1.0 - min(abs(sig1 - sig2) / (sig + 1e-12), 1.0)
        fq   = float(np.clip((ksp + stab) / 2.0, 0.10, 1.0))
        return GBMParams(
            mu=mu, sigma=sig, sigma_ewma=sig_ewma,
            n_obs=len(r),
            advertised_vol=self.ADVERTISED.get(sym, 0.0),
            ks_pvalue=ksp, fit_quality=fq,
        )

    def ann_vol(
        self, p: GBMParams, tpy: int = 86400 * 365
    ) -> float:
        return p.sigma_ewma * math.sqrt(tpy)

    def deviation(
        self, p: GBMParams, tpy: int = 86400 * 365
    ) -> float:
        if p.advertised_vol == 0:
            return 0.0
        return (
            (self.ann_vol(p, tpy) - p.advertised_vol)
            / p.advertised_vol * 100
        )

    def detect_regime(self, r: np.ndarray) -> str:
        if len(r) < 50:
            return "NORMAL"
        recent_std = float(np.std(r[-50:], ddof=1))
        old_std    = (
            float(np.std(r[:-50], ddof=1))
            if len(r) > 100 else recent_std
        )
        ratio = recent_std / max(old_std, 1e-10)
        if ratio > 1.5:
            return "VOLATILE"
        if ratio < 0.6:
            return "RANGING"
        drift = float(np.mean(r[-50:]))
        if abs(drift) > 2.5 * recent_std / math.sqrt(50):
            return "TRENDING"
        return "NORMAL"

# ============================================================================
# TICK IMBALANCE & MOMENTUM QUALITY HELPERS
# ============================================================================
def _compute_tick_imbalance(
    prices: np.ndarray, window: int = IMBALANCE_WINDOW
) -> float:
    if len(prices) < window + 1:
        return 0.0
    recent = prices[-(window + 1):]
    diffs  = np.diff(recent)
    ups    = int(np.sum(diffs > 0))
    downs  = int(np.sum(diffs < 0))
    total  = ups + downs
    if total == 0:
        return 0.0
    return (ups - downs) / total

def _apply_imbalance_adjustment(
    edge_score: float,
    signal_direction: str,
    imbalance: float,
) -> Tuple[float, str]:
    if signal_direction == "neutral":
        return edge_score, "N/A"
    if abs(imbalance) <= IMBALANCE_THRESH:
        return edge_score, "NEUTRAL"
    aligned = (
        (imbalance > 0) == (signal_direction == "bullish")
    )
    if aligned:
        return float(np.clip(
            edge_score * IMBALANCE_BOOST, 0.0, 1.0
        )), "ALIGNED"
    return float(np.clip(
        edge_score * IMBALANCE_PENALTY, 0.0, 1.0
    )), "CONFLICT"

def _momentum_quality(
    prices: np.ndarray,
    signal_direction: str,
    window: int = MOMENTUM_QUALITY_WINDOW,
) -> Tuple[float, str]:
    if len(prices) < window + 1 or signal_direction == "neutral":
        return 1.0, "NEUTRAL"
    diffs = np.diff(prices[-(window + 1):])
    n     = len(diffs)
    if n == 0:
        return 1.0, "NEUTRAL"
    fraction = (
        float(np.sum(diffs > 0)) / n
        if signal_direction == "bullish"
        else float(np.sum(diffs < 0)) / n
    )
    if fraction >= MOMENTUM_QUALITY_STRONG:
        return MOMENTUM_BOOST, "CONFIRMING"
    elif fraction <= MOMENTUM_QUALITY_CONFLICT:
        return MOMENTUM_PENALTY, "CONFLICTING"
    return 1.0, "NEUTRAL"

# ============================================================================
# VOLATILITY REGIME ENGINE
# ============================================================================
class VolatilityRegimeEngine:
    def __init__(self, vm: VolatilityModel):
        self.vm = vm

    def _target_sigma_per_tick(
        self, sym: str, tpy: int = 86400 * 365
    ) -> float:
        adv = ADVERTISED_VOL.get(sym, 0.0)
        if adv == 0:
            return 0.01 / math.sqrt(tpy)
        return adv / math.sqrt(tpy)

    def _normalized_momentum(
        self, r: np.ndarray,
        realized_sigma: float, window: int
    ) -> float:
        recent = r[-window:] if len(r) >= window else r
        if len(recent) < 10 or realized_sigma < 1e-10:
            return 0.0
        mu_r = float(np.mean(recent))
        se   = realized_sigma / math.sqrt(len(recent))
        return float(np.clip(mu_r / max(se, 1e-12), -5.0, 5.0))

    def _ewma_momentum(
        self, r: np.ndarray, window: int
    ) -> float:
        if len(r) < 20:
            return 0.0
        recent     = r[-window:]
        fast_alpha = 2.0 / (min(20, len(recent)) + 1)
        slow_alpha = 2.0 / (len(recent) + 1)
        fast_ewma  = float(recent[0])
        slow_ewma  = float(recent[0])
        for v in recent[1:]:
            fast_ewma = fast_alpha * v + (1 - fast_alpha) * fast_ewma
            slow_ewma = slow_alpha * v + (1 - slow_alpha) * slow_ewma
        std = float(np.std(recent, ddof=1))
        if std < 1e-12:
            return 0.0
        return float(np.clip(
            (fast_ewma - slow_ewma) / std, -1.0, 1.0
        ))

    def analyse(
        self,
        sym:             str,
        lr:              np.ndarray,
        S0:              float,
        horizon:         int,
        profile:         AssetProfile,
        tod_mult:        float = 1.0,
        window_override: Optional[int] = None,
        prices:          Optional[np.ndarray] = None,
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
            / max(target_sigma, 1e-10)
        )
        ratio = realized_sigma / max(target_sigma, 1e-10)
        if ratio < 0.80:
            regime = "COMPRESSED"
        elif ratio > 1.20:
            regime = "EXPANDED"
        else:
            regime = "NORMAL"

        dev_factor = float(np.clip((ratio - 0.70) / 0.80, 0.0, 1.0))
        k_adaptive = float(np.clip(
            profile.vol_k_min
            + dev_factor * (profile.vol_k_max - profile.vol_k_min),
            profile.vol_k_min, profile.vol_k_max,
        ))

        if window_override is not None:
            window = max(min(window_override, len(r) // 2), 30)
        else:
            window = min(profile.drift_window, len(r) // 2, 300)
        short_window = max(window // 3, 20)

        norm_mom_long  = self._normalized_momentum(
            r, realized_sigma, window
        )
        norm_mom_short = self._normalized_momentum(
            r, realized_sigma, short_window
        )
        ewma_mom       = self._ewma_momentum(r, window)
        norm_mom_blended = float(np.clip(
            norm_mom_short * 0.65 + norm_mom_long * 0.35,
            -5.0, 5.0,
        ))
        norm_mom = norm_mom_short

        sigma_h     = realized_sigma * math.sqrt(max(horizon, 1))
        stop_factor = {
            "COMPRESSED": 0.55, "NORMAL": 0.70, "EXPANDED": 0.85,
        }.get(regime, 0.70)
        target_up   = S0 * math.exp(k_adaptive * sigma_h)
        target_down = S0 * math.exp(-k_adaptive * sigma_h)
        stop_up     = S0 * math.exp(stop_factor * sigma_h)
        stop_down   = S0 * math.exp(-stop_factor * sigma_h)

        if prices is not None and len(prices) >= horizon * 2:
            emp_cap = _empirical_move_cap(prices, horizon)
            if emp_cap is not None:
                raw_up   = (target_up   - S0) / S0
                raw_down = (S0 - target_down) / S0
                if raw_up > emp_cap:
                    rr_up     = raw_up / max(
                        (S0 - stop_down) / S0, 1e-10
                    )
                    target_up = S0 * (1.0 + emp_cap)
                    stop_down = S0 * (1.0 - min(
                        emp_cap / max(rr_up, 1e-10),
                        emp_cap * 0.8,
                    ))
                if raw_down > emp_cap:
                    rr_dn       = raw_down / max(
                        (stop_up - S0) / S0, 1e-10
                    )
                    target_down = S0 * (1.0 - emp_cap)
                    stop_up     = S0 * (1.0 + min(
                        emp_cap / max(rr_dn, 1e-10),
                        emp_cap * 0.8,
                    ))

        t_thresh_eff  = max(profile.drift_tstat_min * 0.60, 1.2)
        pvalue        = float(2 * (1 - norm.cdf(abs(norm_mom))))
        is_bullish    = norm_mom > t_thresh_eff and ewma_mom > 0.02
        is_bearish    = norm_mom < -t_thresh_eff and ewma_mom < -0.02
        long_agrees   = (
            (norm_mom_long > 0 and norm_mom_short > 0)
            or (norm_mom_long < 0 and norm_mom_short < 0)
        )
        p_significant = pvalue < 0.25
        mom_score     = float(np.clip(
            abs(norm_mom) / max(t_thresh_eff * 1.5, 1), 0.0, 1.0
        ))
        ewma_score    = float(np.clip(abs(ewma_mom), 0.0, 1.0))
        regime_bonus  = {
            "EXPANDED": 1.35, "NORMAL": 1.00, "COMPRESSED": 0.80,
        }.get(regime, 1.0)
        long_bonus    = 1.15 if long_agrees else 0.90
        raw_edge      = (
            (mom_score * 0.65 + ewma_score * 0.35)
            * regime_bonus * long_bonus * tod_mult
        )
        edge_score = float(np.clip(raw_edge, 0.0, 1.0))

        edge_ok       = edge_score >= profile.signal_edge_min
        strong_enough = edge_score >= 0.18
        if (edge_ok
                and (p_significant or strong_enough)
                and (is_bullish or is_bearish)):
            direction = "bullish" if is_bullish else "bearish"
        else:
            direction = "neutral"

        if prices is not None and len(prices) > IMBALANCE_WINDOW:
            imb = _compute_tick_imbalance(prices, IMBALANCE_WINDOW)
            edge_score, _ = _apply_imbalance_adjustment(
                edge_score, direction, imb
            )

        if prices is not None and direction != "neutral":
            mom_mult, _ = _momentum_quality(prices, direction)
            edge_score  = float(np.clip(
                edge_score * mom_mult, 0.0, 1.0
            ))
            if edge_score < profile.signal_edge_min:
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
            normalized_momentum=norm_mom_blended,
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
        self, sym: str, S0: float, horizon: int,
        profile: AssetProfile, tod_mult: float,
    ) -> VolRegimeResult:
        ts = self._target_sigma_per_tick(sym)
        sh = ts * math.sqrt(max(horizon, 1))
        k  = profile.vol_k_min
        return VolRegimeResult(
            regime="NORMAL",
            realized_sigma=ts, target_sigma=ts,
            vol_deviation=0.0, normalized_momentum=0.0,
            momentum_direction="neutral", k_adaptive=k,
            target_up=S0 * math.exp(k * sh),
            target_down=S0 * math.exp(-k * sh),
            stop_up=S0 * math.exp(0.70 * sh),
            stop_down=S0 * math.exp(-0.70 * sh),
            edge_score=0.0, signal="neutral",
            signal_strength="WEAK", tod_multiplier=tod_mult,
        )

# ============================================================================
# DRIFT ENGINE
# ============================================================================
class DriftEngine:
    def _ewma_momentum(
        self, r: np.ndarray, window: int
    ) -> float:
        if len(r) < 20:
            return 0.0
        recent     = r[-window:]
        fast_alpha = 2.0 / (min(20, len(recent)) + 1)
        slow_alpha = 2.0 / (len(recent) + 1)
        fast_ewma  = float(recent[0])
        slow_ewma  = float(recent[0])
        for v in recent[1:]:
            fast_ewma = fast_alpha * v + (1 - fast_alpha) * fast_ewma
            slow_ewma = slow_alpha * v + (1 - slow_alpha) * slow_ewma
        std = float(np.std(recent, ddof=1))
        if std < 1e-12:
            return 0.0
        return float(np.clip(
            (fast_ewma - slow_ewma) / std, -1.0, 1.0
        ))

    def _rolling_tstat(
        self, r: np.ndarray, window: int
    ) -> Tuple[float, float]:
        recent = r[-window:] if len(r) >= window else r
        if len(recent) < 20:
            return 0.0, 1.0
        mu     = float(np.mean(recent))
        std    = float(np.std(recent, ddof=1))
        if std < 1e-12:
            return 0.0, 1.0
        tstat  = mu / (std / math.sqrt(len(recent)))
        pvalue = float(2 * (1 - norm.cdf(abs(tstat))))
        return tstat, pvalue

    def _multi_window_agree(
        self, r: np.ndarray, window: int
    ) -> bool:
        if len(r) < window * 2:
            return False
        short_w     = max(window // 3, 20)
        ts_short, _ = self._rolling_tstat(r, short_w)
        ts_long,  _ = self._rolling_tstat(r, window)
        return (ts_short * ts_long) > 0

    def analyse(
        self,
        sym:             str,
        lr:              np.ndarray,
        S0:              float,
        profile:         AssetProfile,
        tod_mult:        float = 1.0,
        window_override: Optional[int] = None,
        prices:          Optional[np.ndarray] = None,
    ) -> DriftSignal:
        r = lr[np.isfinite(lr)]
        if window_override is not None:
            window = max(min(window_override, len(r) // 2), 30)
        else:
            window = profile.drift_window
        if len(r) < 30:
            return DriftSignal(
                direction="neutral", tstat=0.0, pvalue=1.0,
                drift_per_tick=0.0, drift_pct=0.0,
                regime="NORMAL", ewma_momentum=0.0,
                window_used=window, edge_score=0.0,
                signal_strength="WEAK",
            )
        tstat, pvalue  = self._rolling_tstat(r, window)
        momentum       = self._ewma_momentum(r, window)
        vm             = VolatilityModel()
        regime         = vm.detect_regime(r)
        agrees         = self._multi_window_agree(r, window)
        recent         = r[-window:] if len(r) >= window else r
        drift_per_tick = float(np.mean(recent))
        drift_pct      = abs(drift_per_tick) / max(S0, 1e-10) * 100

        tstat_score    = float(np.clip(
            abs(tstat) / max(profile.drift_tstat_min * 2, 1),
            0.0, 1.0,
        ))
        momentum_score = float(np.clip(abs(momentum), 0.0, 1.0))
        regime_bonus   = {
            "TRENDING": 1.40, "VOLATILE": 1.15,
            "NORMAL":   1.00, "RANGING":  0.65,
        }.get(regime, 1.0)
        agree_bonus    = 1.20 if agrees else 0.80
        raw_edge       = (
            (tstat_score * 0.55 + momentum_score * 0.45)
            * regime_bonus * agree_bonus * tod_mult
        )
        edge_score = float(np.clip(raw_edge, 0.0, 1.0))

        is_bullish    = tstat > 0 and momentum > 0.05
        is_bearish    = tstat < 0 and momentum < -0.05
        t_significant = abs(tstat) >= profile.drift_tstat_min
        p_significant = pvalue < 0.15

        if (t_significant and p_significant
                and edge_score >= profile.signal_edge_min
                and (is_bullish or is_bearish)):
            direction = "bullish" if is_bullish else "bearish"
        else:
            direction = "neutral"

        if prices is not None and direction != "neutral":
            mom_mult, _ = _momentum_quality(prices, direction)
            edge_score  = float(np.clip(
                edge_score * mom_mult, 0.0, 1.0
            ))
            if edge_score < profile.signal_edge_min:
                direction = "neutral"

        if (edge_score >= profile.strong_edge
                and abs(tstat) >= profile.drift_tstat_min * 1.5
                and agrees and pvalue < 0.05):
            strength = "STRONG"
        elif (edge_score >= profile.moderate_edge
              and abs(tstat) >= profile.drift_tstat_min
              and pvalue < 0.15):
            strength = "MODERATE"
        else:
            strength = "WEAK"

        return DriftSignal(
            direction=direction, tstat=tstat, pvalue=pvalue,
            drift_per_tick=drift_per_tick, drift_pct=drift_pct,
            regime=regime, ewma_momentum=momentum,
            window_used=window, edge_score=edge_score,
            signal_strength=strength,
        )

# ============================================================================
# JUMP-DIFFUSION MODEL
# ============================================================================
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

    def _recent_cluster(
        self, lr: np.ndarray, window: int = 200
    ) -> float:
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
        self,
        lr:            np.ndarray,
        lam_posterior: float,
        expected_lam:  float,
        tod_mult:      float = 1.0,
    ) -> float:
        if len(lr) < 20 or lam_posterior <= 0:
            return 0.0
        is_j, _     = self.detect(lr)
        jump_idx    = np.where(is_j)[0]
        ticks_since = (
            len(lr) if len(jump_idx) == 0
            else len(lr) - int(jump_idx[-1])
        )
        raw_hazard = 1.0 - math.exp(-lam_posterior * ticks_since)
        if (expected_lam > 0
                and lam_posterior < expected_lam * 0.7):
            raw_hazard = min(raw_hazard * 1.5, 1.0)
        return float(np.clip(raw_hazard * tod_mult, 0.0, 1.0))

    def _bayesian_lambda(
        self,
        n_jumps:        int,
        n_obs:          int,
        expected_lam:   float,
        prior_strength: float = 10.0,
    ) -> float:
        if expected_lam <= 0:
            return max(n_jumps / max(n_obs, 1), 1e-6)
        alpha = prior_strength * expected_lam
        beta  = prior_strength
        return (alpha + n_jumps) / (beta + n_obs)

    def _spike_magnitude_percentiles(
        self, jumps: np.ndarray
    ) -> Tuple[float, float]:
        abj = np.abs(jumps)
        if len(abj) < 3:
            return 0.02, 0.03
        return (
            float(np.percentile(abj, 60)),
            float(np.percentile(abj, 75)),
        )

    def _counter_drift(
        self, lr: np.ndarray, is_jump: np.ndarray
    ) -> float:
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
        sig  = (
            max(float(np.std(diff, ddof=1)), 1e-10)
            if len(diff) > 2 else 0.01
        )
        n_jumps  = int(np.sum(is_j))
        n_obs    = len(r)
        lam_mle  = n_jumps / max(n_obs, 1)
        exp_f    = EXPECTED_JUMP_FREQ.get(sym, 0)
        t_lam    = (1.0 / exp_f) if exp_f > 0 else lam_mle
        lam_post = self._bayesian_lambda(n_jumps, n_obs, t_lam)
        lam_dev  = (
            (lam_mle - t_lam) / (t_lam + 1e-12) * 100
            if t_lam > 0 else 0.0
        )
        rc  = self._recent_cluster(r)
        cv  = self._inter_arrival_cv(is_j)
        hz  = self._hazard_intensity_bayesian(r, lam_post, t_lam)
        cd  = self._counter_drift(r, is_j)
        abj = np.abs(jumps)
        jm  = float(np.mean(abj))  if len(abj) > 0 else 0.02
        js  = (
            float(np.std(abj, ddof=1))
            if len(abj) > 1 else 0.01
        )
        p60, p75 = self._spike_magnitude_percentiles(jumps)
        return JumpParams(
            mu=mu, sigma=sig,
            lam=lam_mle, lam_posterior=lam_post,
            jump_mean=jm, jump_std=js, jump_sign=sign,
            n_jumps=n_jumps, n_obs=n_obs,
            expected_lam=t_lam, lam_deviation=lam_dev,
            recent_cluster=rc, inter_arrival_cv=cv,
            hazard_intensity=hz, counter_drift=cd,
            spike_mag_p60=p60, spike_mag_p75=p75,
        )

    def prob_in_n(self, p: JumpParams, n: int) -> float:
        lam = max(p.lam_posterior, p.lam, 1e-10)
        return 1.0 - math.exp(-lam * n)

    def ticks_to_next_range(
        self, p: JumpParams, tick_rate: float
    ) -> Tuple[int, int, str, str, str]:
        lam = max(p.lam_posterior, p.lam, 1e-10)
        if lam <= 0:
            return 300, 600, "~5 min", "~10 min", "~7 min"
        mean_inter   = 1.0 / lam
        accel        = 1.0 - (p.hazard_intensity * 0.60)
        center_ticks = max(int(mean_inter * accel), 5)
        lo_ticks     = max(int(center_ticks * 0.40), 1)
        hi_ticks     = int(center_ticks * 1.60)

        def _fmt(ticks):
            secs = ticks / max(tick_rate, 0.1)
            if secs < 60:      return f"~{secs:.0f}s"
            elif secs < 3600:  return f"~{secs/60:.1f}min"
            else:              return f"~{secs/3600:.1f}hr"

        return (
            lo_ticks, hi_ticks,
            _fmt(lo_ticks), _fmt(hi_ticks), _fmt(center_ticks),
        )

    def jump_magnitude_range(
        self, p: JumpParams
    ) -> Tuple[float, float]:
        lo = max(p.jump_mean - 2 * p.jump_std, 0) * 100
        hi = (p.jump_mean + 2 * p.jump_std) * 100
        return lo, hi

# ============================================================================
# SPIKE ENGINE  (v4.5: suppression is informational, not a gate)
# ============================================================================
class SpikeEngine:
    def __init__(self, jm: JumpDiffusionModel):
        self.jm = jm

    def assess(
        self,
        sym:       str,
        jp:        JumpParams,
        horizon:   int,
        tick_rate: float,
        tod_mult:  float = 1.0,
    ) -> Optional[SpikeAlert]:
        """
        Returns a SpikeAlert if raw_confidence >= SPIKE_MIN_CONFIDENCE.
        Suppression analysis is run separately and attached to the alert.
        Stars and warnings are shown to the user — they never block alerts.
        """
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
        poisson_prob_eff = float(np.clip(
            poisson_prob * tod_mult, 0.0, 1.0
        ))
        jlo, jhi   = self.jm.jump_magnitude_range(jp)
        mag_target = jp.spike_mag_p60 * 100
        tlo, thi, slo, shi, scenter = self.jm.ticks_to_next_range(
            jp, tick_rate
        )

        c1 = poisson_prob_eff >= profile.spike_min_prob
        c2 = mag_target       >= profile.spike_min_magnitude
        c3 = (
            abs(jp.lam_deviation) >= profile.spike_lam_dev_min
            or jp.recent_cluster > jp.expected_lam * 1.2
            or jp.inter_arrival_cv > 1.15
            or jp.hazard_intensity >= profile.spike_hazard_high
        )
        if not (c1 and c2 and c3):
            return None

        p1 = min(
            poisson_prob_eff / max(profile.spike_min_prob, 0.01),
            1.0,
        ) * 0.30
        p2 = min(
            mag_target / max(profile.spike_min_magnitude, 0.001),
            1.0,
        ) * 0.25
        p3 = min(jp.hazard_intensity, 1.0) * 0.25
        p4 = min(abs(jp.lam_deviation) / 30.0, 1.0) * 0.10
        p5 = min(tod_mult - 1.0, 0.12) / 0.12 * 0.10
        confidence = float(np.clip(
            p1 + p2 + p3 + p4 + p5, 0.0, 1.0
        ))

        # Only gate: raw confidence must meet threshold
        if confidence < SPIKE_MIN_CONFIDENCE:
            return None

        return SpikeAlert(
            symbol=sym, fname=fname, direction=direction,
            confidence=confidence,
            poisson_prob=poisson_prob_eff,
            hazard_intensity=jp.hazard_intensity,
            ticks_lo=tlo, ticks_hi=thi,
            time_lo_str=slo, time_hi_str=shi,
            time_center_str=scenter,
            magnitude_lo=jlo, magnitude_hi=jhi,
            magnitude_p60=jp.spike_mag_p60 * 100,
            magnitude_p75=jp.spike_mag_p75 * 100,
            reasons=[],
            is_imminent=True,
            counter_drift=jp.counter_drift,
            tod_multiplier=tod_mult,
        )

    def format_standalone(self, sa: SpikeAlert) -> str:
        """
        Formats spike alert with quality block appended.
        Mirrors the VolSignalQuality pattern:
          — Raw confidence shown
          — Star rating shown
          — Warnings/confirmations shown
          — User decides how to act based on stars
        """
        conf_bar = (
            "█" * int(sa.confidence * 10)
            + "░" * (10 - int(sa.confidence * 10))
        )
        stars = (
            sa.suppression.stars
            if sa.suppression else 5
        )

        # Header with star rating — mirrors vol quality pattern
        dir_arrow = "📈" if sa.direction == "up" else "📉"
        lines = [
            f"⚡ *SPIKE INCOMING — {sa.fname}* {dir_arrow}\n",
            f"*Quality: {_stars(stars)} ({stars}/5)*\n",
            f"Confidence: [{conf_bar}] {sa.confidence:.0%}\n",
            f"Window:     *{sa.time_lo_str} — {sa.time_hi_str}*",
            f"Best est.:  *{sa.time_center_str}*",
            f"Ticks:      *{sa.ticks_lo} — {sa.ticks_hi}*",
            f"Hazard:     {sa.hazard_intensity:.0%} (Bayesian)\n",
        ]

        # Attach quality block — always shown
        if sa.suppression:
            lines.append(
                SpikeSuppressionAnalyser.format_quality_block(
                    sa.suppression
                )
            )

        lines.append(RISK_MSG)
        return "\n".join(lines)

# ============================================================================
# TRADE SETUP BUILDER
# ============================================================================
class TradeSetupBuilder:
    @staticmethod
    def _signal_strength(
        edge: float, prob: float, rr: float,
        profile: AssetProfile,
    ) -> str:
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
        mc:              MCResult,
        timeframe_label: str,
        horizon_ticks:   int,
        regime:          str = "NORMAL",
        profile:         Optional[AssetProfile] = None,
        drift:           Optional[DriftSignal] = None,
        vol_regime:      Optional[VolRegimeResult] = None,
        tod_mult:        float = 1.0,
        quality:         Optional[VolSignalQuality] = None,
    ) -> Optional[TradeSetup]:
        if profile is None:
            profile = effective_profile(mc.symbol)
        S0 = mc.S0
        if S0 <= 0:
            return None
        finals        = mc.paths[:, -1]
        signal_source = "MC"

        if profile.direction_bias == "BUY":
            is_bull, signal_source = True, "PROFILE"
        elif profile.direction_bias == "SELL":
            is_bull, signal_source = False, "PROFILE"
        elif (vol_regime is not None
              and profile.vol_regime_enabled
              and vol_regime.signal != "neutral"):
            is_bull       = vol_regime.signal == "bullish"
            signal_source = "VOL_REGIME"
        elif (vol_regime is not None
              and profile.vol_regime_enabled
              and vol_regime.signal == "neutral"):
            return None
        elif (drift is not None
              and profile.drift_signal_enabled
              and drift.direction != "neutral"):
            is_bull       = drift.direction == "bullish"
            signal_source = "DRIFT"
        else:
            is_bull       = mc.prob_hit_up > mc.prob_hit_down
            signal_source = "MC"
        direction = "BUY" if is_bull else "SELL"

        if (vol_regime is not None
                and profile.vol_regime_enabled
                and vol_regime.signal != "neutral"):
            target = vol_regime.target_up if is_bull else vol_regime.target_down
            inv    = vol_regime.stop_down  if is_bull else vol_regime.stop_up
        else:
            if is_bull:
                winners = finals[finals > S0]
                losers  = finals[finals <= S0]
            else:
                winners = finals[finals < S0]
                losers  = finals[finals >= S0]
            if len(winners) < 10 or len(losers) < 10:
                return None
            if is_bull:
                target = float(np.percentile(
                    winners, profile.target_percentile
                ))
                inv = float(np.percentile(
                    losers, profile.stop_percentile
                ))
            else:
                target = float(np.percentile(
                    winners, 100 - profile.target_percentile
                ))
                inv = float(np.percentile(
                    losers, 100 - profile.stop_percentile
                ))

        try:
            if is_bull:
                mc_t = float(np.percentile(finals, 75))
                mc_s = float(np.percentile(finals, 15))
            else:
                mc_t = float(np.percentile(finals, 25))
                mc_s = float(np.percentile(finals, 85))
            mc_t_pct = abs(mc_t - S0) / S0 * 100
            mc_s_pct = abs(mc_s - S0) / S0 * 100
            if mc_s_pct > 1e-6 and mc_t_pct >= MIN_TARGET_MOVE * 100:
                mc_rr = mc_t_pct / max(mc_s_pct, 1e-10)
                r_t_p = abs(target - S0) / S0 * 100
                r_s_p = abs(inv    - S0) / S0 * 100
                r_rr  = r_t_p / max(r_s_p, 1e-10)
                if mc_rr > r_rr:
                    target = mc_t
                    inv    = mc_s
        except Exception:
            pass

        target_pct   = abs(target - S0) / S0 * 100
        stop_pct     = abs(inv    - S0) / S0 * 100
        min_stop_pct = max(target_pct * 0.30, profile.min_stop_pct * 100)
        if stop_pct < min_stop_pct:
            stop_pct = min_stop_pct
            inv = (
                S0 * (1 - stop_pct / 100) if is_bull
                else S0 * (1 + stop_pct / 100)
            )

        if target_pct < max(
            profile.min_target_pct * 100, MIN_TARGET_MOVE * 100
        ):
            return None
        if stop_pct < 1e-6:
            return None
        rr_ratio   = min(target_pct / max(stop_pct, 1e-10), 4.0)
        min_rr_eff = max(profile.min_rr, MIN_RR_GLOBAL)
        if rr_ratio < min_rr_eff:
            return None

        prob_mc = mc.prob_hit_up   if is_bull else mc.prob_hit_down
        prob_s  = mc.prob_hit_down if is_bull else mc.prob_hit_up

        if (vol_regime is not None
                and profile.vol_regime_enabled
                and vol_regime.signal != "neutral"):
            w_v           = profile.drift_edge_weight
            combined_edge = float(np.clip(
                vol_regime.edge_score * w_v
                + mc.edge_score * (1 - w_v), 0.0, 1.0,
            ))
            prob_t = float(np.clip(
                max(prob_mc + vol_regime.edge_score * 0.12, 0.54),
                0.50, 0.72,
            ))
        elif (drift is not None
              and profile.drift_signal_enabled
              and drift.direction != "neutral"):
            w_d           = profile.drift_edge_weight
            combined_edge = float(np.clip(
                drift.edge_score * w_d
                + mc.edge_score * (1 - w_d), 0.0, 1.0,
            ))
            prob_t = float(np.clip(
                max(prob_mc + drift.edge_score * 0.08, 0.53),
                0.50, 0.70,
            ))
            signal_source = (
                "COMBINED" if signal_source == "MC"
                else signal_source
            )
        else:
            combined_edge = mc.edge_score
            prob_t        = float(np.clip(prob_mc, 0.50, 0.68))

        tod_adj = float(np.clip(
            (tod_mult - 1.0) * 0.05, -0.03, 0.03
        ))
        prob_t  = float(np.clip(
            prob_t + tod_adj, MIN_PROB_TARGET, 0.75
        ))
        if prob_t < MIN_PROB_TARGET:
            return None

        ev = prob_t * target_pct - (1 - prob_t) * stop_pct
        if ev <= 0:
            return None

        ev_min = {
            "VOLATILITY": 0.15, "BOOM":  0.30,
            "CRASH":      0.30, "JUMP":  0.20, "STEP": 0.05,
        }.get(profile.name, 0.15)
        if ev < ev_min:
            return None

        edge_pct = (prob_t - 0.5) * 200
        if edge_pct <= 0:
            return None

        strength = TradeSetupBuilder._signal_strength(
            combined_edge, prob_t, rr_ratio, profile
        )
        return TradeSetup(
            direction=direction, entry=S0,
            target=target, invalidation=inv,
            target_pct=target_pct, stop_pct=stop_pct,
            rr_ratio=rr_ratio, prob_target=prob_t,
            prob_stop=prob_s, edge_pct=edge_pct,
            expected_value=ev, signal_strength=strength,
            timeframe_label=timeframe_label,
            horizon_ticks=horizon_ticks,
            regime=regime, profile_name=profile.name,
            signal_source=signal_source,
            tod_multiplier=tod_mult,
            quality=quality,
        )

    @staticmethod
    def format_setup(
        ts:         TradeSetup,
        fname:      str,
        drift=None,
        vol_regime=None,
    ) -> str:
        dir_icon = "📈" if ts.direction == "BUY" else "📉"
        s_icon   = {
            "STRONG": "🟢", "MODERATE": "🟡", "WEAK": "🔴",
        }.get(ts.signal_strength, "⬜")
        sg  = "+" if ts.direction == "BUY" else "-"
        sgs = "-" if ts.direction == "BUY" else "+"

        quality_line = ""
        if ts.quality:
            quality_line = (
                f"*Quality: {_stars(ts.quality.stars)} "
                f"({ts.quality.stars}/5)*\n\n"
            )

        msg = (
            f"*TRADE ALERT v4.5 — {fname}* "
            f"{dir_icon} {_se(ts.direction)}\n\n"
            f"{quality_line}"
            f"{s_icon} *{ts.direction}* "
            f"{ts.signal_strength} {ts.signal_source}\n"
            f"Entry:  `{ts.entry:.6f}`\n"
            f"Target: `{ts.target:.6f}` "
            f"({sg}{ts.target_pct:.3f}%)\n"
            f"Stop:   `{ts.invalidation:.6f}` "
            f"({sgs}{ts.stop_pct:.3f}%)\n"
        )
        if ts.quality:
            msg += VolSignalQualityAnalyser.format_quality_block(
                ts.quality
            )
        msg += RISK_MSG
        return msg

# ============================================================================
# CONFLICT RESOLVER
# ============================================================================
class ConflictResolver:
    @staticmethod
    def resolve(
        sym:   str,
        mc:    MCResult,
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
                return None, False, "spike_primary_conflict"
            return trade, True, "both_independent"

# ============================================================================
# MC ENGINE
# ============================================================================
class MCEngine:
    def __init__(self, tod: TimeOfDayProfile):
        self.vm  = VolatilityModel()
        self.jm  = JumpDiffusionModel()
        self.se  = SpikeEngine(self.jm)
        self.tsb = TradeSetupBuilder()
        self.cr  = ConflictResolver()
        self.de  = DriftEngine()
        self.vre = VolatilityRegimeEngine(self.vm)
        self.tod = tod
        self.htr = HybridTargetRefiner()
        self.vqa = VolSignalQualityAnalyser()

    def _is_jump_sym(self, sym: str) -> bool:
        f = REVERSE_MAP.get(sym, "")
        return any(x in f for x in ("Boom", "Crash", "Jump"))

    def _adaptive_targets(
        self, S0: float, sigma_tick: float,
        horizon: int, profile: AssetProfile,
    ):
        base = profile.sigma_multiplier_base
        if   horizon <= 50:  mult = base
        elif horizon <= 100: mult = base * 1.15
        elif horizon <= 300: mult = base * 1.35
        elif horizon <= 600: mult = base * 1.55
        else:                mult = base * 1.75
        sigma_h = sigma_tick * math.sqrt(max(horizon, 1))
        tup     = S0 * math.exp(mult * sigma_h)
        tdn     = S0 * math.exp(-mult * sigma_h)
        pct     = (tup - S0) / S0 * 100
        return tup, tdn, sigma_h, f"+-{mult:.1f}s ({pct:.3f}%)"

    @staticmethod
    def _first_passage(paths, target_up, target_down, n_paths):
        hit_u = np.any(paths >= target_up,   axis=1)
        hit_d = np.any(paths <= target_down, axis=1)
        both  = hit_u & hit_d
        fw    = np.zeros(n_paths, dtype=bool)
        for i in np.where(both)[0]:
            iu  = int(np.argmax(paths[i] >= target_up))
            id_ = int(np.argmax(paths[i] <= target_down))
            fw[i] = iu < id_
        p_up = (
            np.sum(hit_u & ~hit_d) + np.sum(fw)
        ) / max(n_paths, 1)
        p_dn = (
            np.sum(hit_d & ~hit_u) + np.sum(both & ~fw)
        ) / max(n_paths, 1)
        return p_up, p_dn

    def _edge_score(
        self,
        prob_hit_up:  float,
        prob_hit_dn:  float,
        fit_quality:  float,
        lam_deviation:float,
        is_jump:      bool,
        ci_width_pct: float,
        wr_weight:    float,
        profile:      AssetProfile,
        drift:        Optional[DriftSignal] = None,
        vol_regime:   Optional[VolRegimeResult] = None,
        tod_mult:     float = 1.0,
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
            regime_bonus = 1.0 + min(
                abs(lam_deviation) / 80.0, 0.20
            )

        mc_score = float(np.clip(
            skew * fit_quality * regime_bonus * wr_weight, 0.0, 1.0
        ))
        mc_score = float(np.clip(
            mc_score + (tod_mult - 1.0) * 0.05 * mc_score,
            0.0, 1.0,
        ))

        if (vol_regime is not None
                and profile.vol_regime_enabled
                and vol_regime.signal != "neutral"):
            w_v   = profile.drift_edge_weight
            score = float(np.clip(
                vol_regime.edge_score * w_v
                + mc_score * (1 - w_v), 0.0, 1.0,
            ))
            sig = (
                vol_regime.signal
                if score >= profile.signal_edge_min
                else "neutral"
            )
            return score, sig

        if (drift is not None
                and profile.drift_signal_enabled
                and drift.direction != "neutral"):
            w_d   = profile.drift_edge_weight
            score = float(np.clip(
                drift.edge_score * w_d
                + mc_score * (1 - w_d), 0.0, 1.0,
            ))
            sig = (
                drift.direction
                if score >= profile.signal_edge_min
                else "neutral"
            )
            return score, sig

        score = mc_score
        if (gap >= profile.direction_gap
                and score >= profile.signal_edge_min):
            sig = (
                "bullish" if prob_hit_up > prob_hit_dn
                else "bearish"
            )
        elif is_jump and score >= profile.signal_edge_min * 0.75:
            sig = "jump_imminent"
        else:
            sig = "neutral"
        return score, sig

    def _bootstrap_ensemble(
        self, sym: str, lr: np.ndarray, S0: float,
        ns: int, is_j: bool, jp, gp, n_paths: int,
        tup: float, tdn: float,
    ) -> Tuple[float, int]:
        if not ENABLE_ENSEMBLE:
            return 1.0, 1
        sample_size = max(n_paths // 2, 200)
        bull_votes  = 0
        bear_votes  = 0
        for seed_offset in range(N_BOOTSTRAP_RUNS):
            rng = np.random.default_rng(
                int(time.time() * 1000) % 2**31 + seed_offset
            )
            try:
                if is_j and jp is not None:
                    Z  = rng.standard_normal(
                        (sample_size, ns)
                    ).astype(np.float64)
                    U  = rng.uniform(
                        0, 1, (sample_size, ns)
                    ).astype(np.float64)
                    Zj = rng.standard_normal(
                        (sample_size, ns)
                    ).astype(np.float64)
                    p  = _jd_kernel(
                        float(S0), float(jp.mu), float(jp.sigma),
                        float(jp.lam_posterior),
                        float(jp.jump_mean), float(jp.jump_std),
                        float(jp.jump_sign),
                        1.0, sample_size, ns, Z, U, Zj,
                    )
                else:
                    Z   = rng.standard_normal(
                        (sample_size, ns)
                    ).astype(np.float64)
                    sig = float(gp.sigma_ewma) if gp else 0.01
                    mu_v= float(gp.mu)         if gp else 0.0
                    p   = _gbm_kernel(
                        float(S0), mu_v, sig,
                        1.0, sample_size, ns, Z,
                    )
                if not np.all(np.isfinite(p)):
                    p = np.where(np.isfinite(p), p, S0)
                p_up_b, _ = self._first_passage(
                    p, tup, tdn, sample_size
                )
                if p_up_b > 0.5:
                    bull_votes += 1
                else:
                    bear_votes += 1
            except Exception as e:
                log.debug(f"Bootstrap run {seed_offset}: {e}")
                continue

        total = bull_votes + bear_votes
        if total == 0:
            return 1.0, 0
        agreement = max(bull_votes, bear_votes) / total
        return agreement, total

    def run(
        self,
        sym:           str,
        dm:            DataManager,
        horizon:       int,
        horizon_label: str,
        pm:            PatternMemory,
        n:             Optional[int] = None,
        timeframe_key: Optional[str] = None,
    ) -> Optional[MCResult]:
        profile    = effective_profile(sym)
        n_paths    = n or profile.mc_paths
        lr         = dm.log_returns(sym)
        S0         = dm.last(sym)
        if S0 is None or len(lr) < MIN_TICKS:
            return None
        ns         = min(horizon, MC_STEPS_MAX)
        is_j       = self._is_jump_sym(sym)
        regime     = "NORMAL"
        tod_mult   = (
            self.tod.multiplier(sym) if profile.tod_enabled else 1.0
        )
        prices_arr = dm.prices(sym)

        try:
            drift      = None
            vol_regime = None
            hw         = max(min(ns * 2, profile.drift_window), 30)
            rng        = np.random.default_rng()
            jp         = None
            gp         = None

            if is_j:
                jp = self.jm.fit_unbiased(sym, lr)
                Z  = rng.standard_normal(
                    (n_paths, ns)
                ).astype(np.float64)
                U  = rng.uniform(
                    0, 1, (n_paths, ns)
                ).astype(np.float64)
                Zj = rng.standard_normal(
                    (n_paths, ns)
                ).astype(np.float64)
                paths = _jd_kernel(
                    float(S0), float(jp.mu), float(jp.sigma),
                    float(jp.lam_posterior),
                    float(jp.jump_mean), float(jp.jump_std),
                    float(jp.jump_sign),
                    1.0, n_paths, ns, Z, U, Zj,
                )
                pj         = self.jm.prob_in_n(jp, ns)
                ej         = self.jm.ticks_to_next_range(
                    jp, dm.observed_tick_rate(sym)
                )[0]
                jlo, jhi   = self.jm.jump_magnitude_range(jp)
                sigma_tick = max(jp.sigma, 1e-10)
                fq         = 0.80
                lam_dev    = jp.lam_deviation
                if profile.drift_signal_enabled:
                    drift = self.de.analyse(
                        sym, lr, S0, profile, tod_mult,
                        window_override=hw, prices=prices_arr,
                    )
                    regime = drift.regime
            else:
                gp = self.vm.fit(sym, lr)
                Z  = rng.standard_normal(
                    (n_paths, ns)
                ).astype(np.float64)
                paths = _gbm_kernel(
                    float(S0), float(gp.mu),
                    float(gp.sigma_ewma),
                    1.0, n_paths, ns, Z,
                )
                pj = ej = jlo = jhi = None
                sigma_tick = max(gp.sigma_ewma, 1e-10)
                fq         = gp.fit_quality
                lam_dev    = 0.0
                if profile.vol_regime_enabled:
                    vol_regime = self.vre.analyse(
                        sym, lr, S0, ns, profile, tod_mult,
                        window_override=hw, prices=prices_arr,
                    )
                    regime = vol_regime.regime
                elif profile.drift_signal_enabled:
                    drift = self.de.analyse(
                        sym, lr, S0, profile, tod_mult,
                        window_override=hw, prices=prices_arr,
                    )
                    regime = drift.regime

            if not np.all(np.isfinite(paths)):
                paths = np.where(np.isfinite(paths), paths, S0)

            if (vol_regime is not None
                    and profile.vol_regime_enabled
                    and vol_regime.signal != "neutral"):
                tup     = vol_regime.target_up
                tdn     = vol_regime.target_down
                sigma_h = (
                    vol_regime.realized_sigma * math.sqrt(max(ns, 1))
                )
                k       = vol_regime.k_adaptive
                pct     = (tup - S0) / S0 * 100
                tlabel  = f"VReg k={k:.2f} ({pct:.3f}%)"
            else:
                tup, tdn, sigma_h, tlabel = self._adaptive_targets(
                    S0, sigma_tick, ns, profile
                )

            finals       = paths[:, -1]
            prob_up_h    = float(np.mean(finals > S0))
            prob_hit_up, prob_hit_dn = self._first_passage(
                paths, tup, tdn, n_paths
            )
            p5_end       = float(np.percentile(finals, 5))
            p95_end      = float(np.percentile(finals, 95))
            ci_width_pct = (p95_end - p5_end) / max(S0, 1e-10)
            wr_weight    = pm.win_rate_weight(sym, ns)

            score, sig = self._edge_score(
                prob_hit_up, prob_hit_dn, fq, lam_dev,
                is_j, ci_width_pct, wr_weight, profile,
                drift, vol_regime, tod_mult,
            )

            ens_agreement = 1.0
            ens_n_runs    = 1
            if ENABLE_ENSEMBLE:
                ens_agreement, ens_n_runs = self._bootstrap_ensemble(
                    sym, lr, S0, ns, is_j, jp, gp,
                    min(n_paths, 1000), tup, tdn,
                )
                if ens_agreement < ENSEMBLE_AGREEMENT_FLOOR:
                    score = score * ens_agreement
                    sig   = "neutral"

            implied_edge = (
                max(prob_hit_up, prob_hit_dn) - 0.5
            ) * 200

            mc = MCResult(
                symbol=sym, S0=float(S0),
                horizon=ns, horizon_label=horizon_label,
                paths=paths,
                p5=np.percentile(paths, 5, axis=0),
                p25=np.percentile(paths, 25, axis=0),
                p50=np.percentile(paths, 50, axis=0),
                p75=np.percentile(paths, 75, axis=0),
                p95=np.percentile(paths, 95, axis=0),
                target_up=tup, target_down=tdn,
                target_label=tlabel,
                prob_up_horizon=prob_up_h,
                prob_hit_up=prob_hit_up,
                prob_hit_down=prob_hit_dn,
                sigma_horizon=sigma_h,
                ci_width_pct=float(ci_width_pct),
                prob_jump=pj, exp_jump_tick=ej,
                jump_magnitude_lo=jlo, jump_magnitude_hi=jhi,
                edge_score=score, signal=sig,
                implied_edge_pct=implied_edge,
                timeframe_key=timeframe_key,
                regime=(
                    vol_regime.regime if vol_regime
                    else (drift.regime if drift else regime)
                ),
                profile_name=profile.name,
                drift_signal=drift, vol_regime=vol_regime,
                tod_multiplier=tod_mult,
                ens_agreement=ens_agreement,
                ens_n_runs=ens_n_runs,
            )

            # v4.5: vol signal quality for non-jump directional signals
            quality = None
            if (sig not in ("neutral", "jump_imminent")
                    and not is_j
                    and len(lr) >= 50):
                drift_per_tick = float(np.mean(
                    lr[-min(ns, len(lr)):]
                )) if len(lr) > 0 else 0.0
                quality = self.vqa.analyse(
                    sym=sym, lr=lr, prices=prices_arr,
                    signal_dir=sig,
                    vol_regime=vol_regime, drift=drift,
                    sigma_tick=sigma_tick,
                    drift_per_tick=drift_per_tick,
                    profile=profile,
                )

            ts = self.tsb.build(
                mc, horizon_label, ns, mc.regime,
                profile, drift, vol_regime, tod_mult,
                quality=quality,
            )
            if ts is not None:
                try:
                    ctr = ConditionalTargetRefiner(self.htr)
                    ref_t, ref_s, ref_m = ctr.refine(
                        sym=sym, S0=float(S0),
                        is_bull=(ts.direction == "BUY"),
                        mc_target=ts.target,
                        mc_stop=ts.invalidation,
                        finals=finals, prices=prices_arr,
                        horizon=ns, records=pm.records, pm=pm,
                    )
                    ref_t_pct = abs(ref_t - S0) / S0 * 100
                    ref_s_pct = abs(ref_s - S0) / S0 * 100
                    if (ref_t_pct >= MIN_TARGET_MOVE * 100
                            and ref_s_pct > 1e-6):
                        ref_rr = min(
                            ref_t_pct / max(ref_s_pct, 1e-10), 4.0
                        )
                        if ref_rr >= MIN_RR_GLOBAL:
                            ts.target       = ref_t
                            ts.invalidation = ref_s
                            ts.target_pct   = ref_t_pct
                            ts.stop_pct     = ref_s_pct
                            ts.rr_ratio     = ref_rr
                            ts.signal_source += f"+{ref_m}"
                except Exception as e:
                    log.debug(f"ConditionalRefiner: {e}")
            mc.trade_setup = ts
            return mc

        except Exception as e:
            log.error(f"MCEngine.run error ({sym}): {e}")
            return None

# ============================================================================
# NARRATIVE ENGINE
# ============================================================================
class NarrativeEngine:
    def __init__(self, vm: VolatilityModel,
                 jm: JumpDiffusionModel):
        self.vm = vm
        self.jm = jm

    def _regime_desc(self, regime: str) -> str:
        return {
            "VOLATILE":   "Volatility ELEVATED.",
            "RANGING":    "Market RANGING — directional signals unreliable.",
            "TRENDING":   "Market TRENDING — best for directional signals.",
            "NORMAL":     "Market behaving NORMALLY.",
            "COMPRESSED": "Vol COMPRESSED — below advertised target.",
            "EXPANDED":   "Vol EXPANDED — above advertised target.",
        }.get(regime, "Normal regime.")

    def vol_context(
        self, sym: str, gp: GBMParams,
        mc: MCResult, tf_label: str,
    ) -> str:
        profile     = effective_profile(sym)
        ann         = self.vm.ann_vol(gp) * 100
        dev         = self.vm.deviation(gp)
        lines = [
            f"*{_friendly(sym)} ({tf_label})* "
            f"[{profile.name}] | Regime: *{mc.regime}* | "
            f"ToD: x{mc.tod_multiplier:.3f} | "
            f"Ens: {mc.ens_agreement:.3f}({mc.ens_n_runs}r)",
            self._regime_desc(mc.regime),
        ]
        if gp.advertised_vol > 0:
            lines.append(
                f"Design vol: *{gp.advertised_vol*100:.0f}%* ann | "
                f"Live EWMA: *{ann:.2f}%* "
                f"({'above' if dev > 0 else 'below'} by {abs(dev):.1f}%)"
            )
        if mc.vol_regime:
            vr = mc.vol_regime
            d  = "📈" if vr.signal == "bullish" else (
                "📉" if vr.signal == "bearish" else "⬜"
            )
            lines.append(
                f"VolReg: *{vr.signal.upper()}* {d} "
                f"z={vr.normalized_momentum:+.3f} "
                f"k={vr.k_adaptive:.3f} "
                f"edge={vr.edge_score:.3f}"
            )
        if mc.drift_signal:
            ds = mc.drift_signal
            lines.append(
                f"Drift: *{ds.direction.upper()}* "
                f"t={ds.tstat:+.3f} p={ds.pvalue:.3f} "
                f"edge={ds.edge_score:.3f}"
            )
        if mc.trade_setup and mc.trade_setup.quality:
            lines.append(
                VolSignalQualityAnalyser.format_quality_block(
                    mc.trade_setup.quality
                )
            )
        if mc.signal == "bullish":
            lines.append(
                f"*Signal — BULLISH* 🟢 Edge={mc.edge_score:.3f}"
            )
        elif mc.signal == "bearish":
            lines.append(
                f"*Signal — BEARISH* 🔴 Edge={mc.edge_score:.3f}"
            )
        else:
            lines.append("*Signal — NEUTRAL* ⬜")
        if gp.ks_pvalue < 0.05:
            lines.append("*Fat tails* (KS p<0.05) — wider stops.")
        if mc.trade_setup:
            ts = mc.trade_setup
            lines.append(
                f"Setup: {ts.direction} R/R *1:{ts.rr_ratio:.2f}* "
                f"P={ts.prob_target:.2%} "
                f"EV=`{ts.expected_value:+.4f}%`"
            )
        else:
            lines.append(
                f"No setup — P>={MIN_PROB_TARGET:.0%} "
                f"EV>0 RR>={MIN_RR_GLOBAL:.1f}"
            )
        return "\n".join(lines)

    def jump_context(
        self, sym: str, jp: JumpParams,
        mc: MCResult, tf_label: str,
    ) -> str:
        profile = effective_profile(sym)
        cat     = _cat(sym)
        fname   = _friendly(sym)
        freq    = EXPECTED_JUMP_FREQ.get(sym, 0)
        lines   = [
            f"*{fname} ({tf_label})* [{profile.name}] | "
            f"Regime: *{mc.regime}* | ToD: x{mc.tod_multiplier:.3f}",
        ]
        if cat == "boom":
            lines.append(
                f"Boom: ~1 upward spike per *{freq} ticks*. "
                f"Fitted: 1/{1/max(jp.lam_posterior,1e-10):.0f}t"
            )
        elif cat == "crash":
            lines.append(
                f"Crash: ~1 downward spike per *{freq} ticks*. "
                f"Fitted: 1/{1/max(jp.lam_posterior,1e-10):.0f}t"
            )
        else:
            lines.append(
                f"Jump: ~1 per *{freq} ticks*. "
                f"Direction: *{'UP' if jp.jump_sign > 0 else 'DOWN'}*"
            )
        lines.append(
            f"Spike P60: *{jp.spike_mag_p60*100:.3f}%* | "
            f"Hazard: *{jp.hazard_intensity:.0%}*"
        )
        if mc.prob_jump is not None:
            lines.append(
                f"P(spike/{mc.horizon}t): "
                f"*{mc.prob_jump:.1%}* (posterior λ)"
            )
        if mc.signal in ("bullish","bearish"):
            icon = "🟢" if mc.signal == "bullish" else "🔴"
            lines.append(
                f"*Signal — {mc.signal.upper()}* {icon} "
                f"Edge={mc.edge_score:.3f}"
            )
        elif mc.signal == "jump_imminent":
            lines.append(
                f"*Signal — JUMP IMMINENT* ⚡ "
                f"Edge={mc.edge_score:.3f}"
            )
        else:
            lines.append("*Signal — NEUTRAL* ⬜")
        return "\n".join(lines)

    def step_context(
        self, sym: str, gp: GBMParams,
        mc: MCResult, tf_label: str,
    ) -> str:
        lines = [
            f"*Step Index ({tf_label})* | "
            f"Regime: *{mc.regime}* | ToD: x{mc.tod_multiplier:.3f}",
            "Fixed +-0.1 per tick. Random direction.",
            f"EWMA sigma: *{gp.sigma_ewma:.8f}*/tick",
        ]
        if mc.drift_signal:
            ds = mc.drift_signal
            lines.append(
                f"Drift: *{ds.direction.upper()}* "
                f"t={ds.tstat:+.3f} p={ds.pvalue:.3f}"
            )
        if mc.trade_setup:
            ts = mc.trade_setup
            lines.append(
                f"Setup: {ts.direction} R/R *1:{ts.rr_ratio:.2f}*"
            )
        return "\n".join(lines)

    def risk_reward_block(
        self, mc: MCResult, risk_pct: float
    ) -> str:
        if mc.trade_setup:
            ts  = mc.trade_setup
            sg  = "+" if ts.direction == "BUY" else "-"
            sgs = "-" if ts.direction == "BUY" else "+"
            return (
                f"\n*R/R ({risk_pct:.0f}% risk):*\n"
                f"Direction:    *{ts.direction}*\n"
                f"Entry:        `{ts.entry:.6f}`\n"
                f"Target:       `{ts.target:.6f}` "
                f"({sg}{ts.target_pct:.3f}%)\n"
                f"Invalidation: `{ts.invalidation:.6f}` "
                f"({sgs}{ts.stop_pct:.3f}%)\n"
                f"R/R Ratio:    *1 : {ts.rr_ratio:.2f}*\n"
                f"P(target):    `{ts.prob_target:.2%}`\n"
                f"EV/trade:     `{ts.expected_value:+.4f}%`\n"
                f"Source:       *{ts.signal_source}*\n"
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
            f"Note:   Trade gates not met."
        )

# ============================================================================
# ALERT ENGINE
# ============================================================================
class AlertEngine:
    def __init__(
        self,
        mce:      MCEngine,
        pm:       PatternMemory,
        pe:       PersistenceEngine,
        adaptive: AdaptiveThresholds,
    ):
        self.mce      = mce
        self.pm       = pm
        self.pe       = pe
        self.adaptive = adaptive
        self._last: Dict[str, float] = pe.get_alert_last()

    def _save_last(self):
        self.pe.save_alert_last(self._last)

    def check_trade(
        self,
        sym:           str,
        dm:            DataManager,
        horizon:       int,
        horizon_label: str,
    ) -> Optional[Tuple[float, str, MCResult]]:
        profile = effective_profile(sym)
        now     = time.time()
        if now - self._last.get(sym, 0) < profile.alert_cooldown:
            return None
        if dm.n(sym) < profile.alert_min_ticks:
            return None

        r = self.mce.run(sym, dm, horizon, horizon_label,
                         self.pm, n=5000)
        if r is None:
            return None

        if profile.vol_regime_enabled:
            if (r.vol_regime is None
                    or r.vol_regime.signal == "neutral"):
                return None
        elif profile.drift_signal_enabled:
            if (r.drift_signal is None
                    or r.drift_signal.direction == "neutral"):
                return None

        if r.signal == "neutral":
            return None
        if r.ens_agreement < ENSEMBLE_AGREEMENT_FLOOR:
            return None
        if r.trade_setup is None:
            return None

        ts       = r.trade_setup
        eff_prob = max(
            self.adaptive.get_prob_threshold(sym),
            MIN_PROB_TARGET,
        )
        eff_rr   = max(
            self.adaptive.get_rr_threshold(sym),
            MIN_RR_GLOBAL,
        )

        if ts.signal_strength not in profile.alert_strength:
            return None
        if ts.rr_ratio < max(profile.alert_min_rr, eff_rr):
            return None
        if r.edge_score < profile.alert_edge_min:
            return None
        if ts.prob_target < eff_prob:
            return None
        if ts.expected_value <= 0:
            return None
        if ts.edge_pct <= MIN_EDGE_PCT:
            return None
        if ts.target_pct < MIN_TARGET_MOVE * 100:
            return None

        self._last[sym] = now
        self._save_last()
        log.info(
            f"Trade PASSED {sym}: {ts.direction} "
            f"[{ts.signal_strength}] P={ts.prob_target:.2%} "
            f"EV={ts.expected_value:+.4f}% "
            f"RR=1:{ts.rr_ratio:.2f} "
            f"Ens={r.ens_agreement:.3f} "
            f"Q={ts.quality.stars if ts.quality else 'N/A'}/5"
        )
        return r.edge_score, r.signal, r

    def check_spike(
        self,
        sym:  str,
        dm:   DataManager,
        jm:   JumpDiffusionModel,
        se:   SpikeEngine,
        ssa:  SpikeSuppressionAnalyser,
        vm:   VolatilityModel,
    ) -> Optional[SpikeAlert]:
        """
        Returns a spike alert if raw_confidence >= SPIKE_MIN_CONFIDENCE.
        Always attaches suppression quality info to the alert.
        Suppression NEVER blocks — only informs.
        """
        profile   = effective_profile(sym)
        now       = time.time()
        spike_key = f"spike_{sym}"
        if not profile.spike_enabled:
            return None
        if now - self._last.get(spike_key, 0) < profile.alert_cooldown:
            return None
        if dm.n(sym) < MIN_TICKS:
            return None

        lr         = dm.log_returns(sym)
        jp         = jm.fit_unbiased(sym, lr)
        tr         = dm.observed_tick_rate(sym)
        tod_mult   = self.mce.tod.multiplier(sym)
        prices_arr = dm.prices(sym)

        # Raw spike assessment — only gate is raw_confidence >= 0.95
        sa = se.assess(sym, jp, 300, tr, tod_mult)
        if sa is None:
            return None

        # Compute vol deviation for suppression analysis
        gp_tmp  = vm.fit(sym, lr)
        adv     = ADVERTISED_VOL.get(sym, gp_tmp.sigma_ewma)
        vol_dev = (
            (gp_tmp.sigma_ewma - adv) / max(adv, 1e-10)
        )

        # Attach quality info — INFORMATIONAL ONLY
        suppression = ssa.analyse(
            sym=sym, jp=jp,
            raw_confidence=sa.confidence,
            prices=prices_arr,
            current_tick=dm.tick_count(sym),
            vol_deviation=vol_dev,
            lr=lr,
        )
        sa.suppression = suppression

        # Cooldown and logging
        self._last[spike_key] = now
        self._save_last()

        log.info(
            f"Spike ALERT {sym}: conf={sa.confidence:.2f} "
            f"stars={suppression.stars}/5 "
            f"drought={suppression.ticks_since_spike}t "
            f"activity={suppression.activity_ratio:.2f} "
            f"vol_dev={vol_dev*100:+.1f}%"
        )
        return sa

# SQE v4.5 — Part 3
# ============================================================================
# CHART GENERATOR
# ============================================================================
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
        self,
        fname: str, sym: str, dm: DataManager,
        mc:    Optional[MCResult], gp: Optional[GBMParams],
        jp:    Optional[JumpParams], vm: VolatilityModel,
        note:  str, tod: TimeOfDayProfile,
    ) -> BytesIO:
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

        # Panel 1: Price + MC paths
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
                replace=False,
            )
            for i in idx:
                a1.plot(xf, mc.paths[i], color=PU,
                        alpha=0.03, lw=0.5)
            a1.fill_between(xf, mc.p5,  mc.p95, alpha=0.12,
                            color=PU, label="90% CI", zorder=2)
            a1.fill_between(xf, mc.p25, mc.p75, alpha=0.25,
                            color=BL, label="50% CI", zorder=3)
            a1.plot(xf, mc.p50, color=GR, lw=2.2,
                    ls="--", label="Median", zorder=6)
            a1.axhline(mc.S0, color=YL, lw=0.8, ls=":",
                       alpha=0.7, label="Entry", zorder=4)

            if mc.trade_setup:
                ts  = mc.trade_setup
                tc  = GR2 if ts.direction == "BUY" else RD2
                ic  = RD2 if ts.direction == "BUY" else GR2
                a1.axhline(ts.target, color=tc, lw=2.0,
                           ls="-", alpha=0.90, label="TARGET")
                a1.axhline(ts.invalidation, color=ic, lw=1.8,
                           ls="--", alpha=0.85, label="STOP")
                xmid  = ns * 0.55
                sg    = "+" if ts.direction == "BUY" else "-"
                q_str = (
                    f" {_stars(ts.quality.stars)}"
                    if ts.quality else ""
                )
                a1.annotate(
                    f"TARGET {sg}{ts.target_pct:.2f}%\n"
                    f"RR 1:{ts.rr_ratio:.2f} "
                    f"P={ts.prob_target:.1%}{q_str}",
                    xy=(xmid, ts.target),
                    fontsize=8, color=tc, fontweight="bold",
                    va=(
                        "bottom" if ts.direction == "BUY"
                        else "top"
                    ),
                    bbox=dict(boxstyle="round,pad=0.2",
                              fc=BG, ec=tc, alpha=0.85),
                )
                sg2 = "-" if ts.direction == "BUY" else "+"
                a1.annotate(
                    f"STOP {sg2}{ts.stop_pct:.2f}%",
                    xy=(xmid, ts.invalidation),
                    fontsize=8, color=ic, fontweight="bold",
                    va=(
                        "top" if ts.direction == "BUY"
                        else "bottom"
                    ),
                    bbox=dict(boxstyle="round,pad=0.2",
                              fc=BG, ec=ic, alpha=0.85),
                )
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
                    f"{'▲' if vr.signal=='bullish' else '▼'} "
                    f"k={vr.k_adaptive:.2f} "
                    f"Ens:{mc.ens_agreement:.2f}",
                    xy=(ns * 0.05, mc.S0),
                    fontsize=8.5, color=arc, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3",
                              fc=BG, ec=arc, alpha=0.90),
                )
            elif (mc.drift_signal
                  and mc.drift_signal.direction != "neutral"):
                ds  = mc.drift_signal
                arc = GR2 if ds.direction == "bullish" else RD2
                a1.annotate(
                    f"DRIFT "
                    f"{'▲' if ds.direction=='bullish' else '▼'} "
                    f"t={ds.tstat:+.2f} p={ds.pvalue:.3f}",
                    xy=(ns * 0.05, mc.S0),
                    fontsize=8.5, color=arc, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3",
                              fc=BG, ec=arc, alpha=0.90),
                )
            self._prob_box(a1, mc, profile)

        a1.set_title(
            f"  SQE v4.5  ·  {fname}  ·  [{profile.name}]  ·  "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            fontsize=12, color=WH, loc="left",
            pad=12, fontweight="bold",
        )
        if mc:
            src    = (mc.trade_setup.signal_source
                      if mc.trade_setup else "—")
            ts_str = ""
            if mc.trade_setup:
                ts    = mc.trade_setup
                q_str = (
                    f" Q:{ts.quality.stars}/5"
                    if ts.quality else ""
                )
                ts_str = (
                    f"{ts.direction} RR1:{ts.rr_ratio:.2f} "
                    f"P={ts.prob_target:.1%}{q_str} | "
                )
            a1.set_title(
                f"{mc.horizon_label} | "
                f"{ts_str}{mc.signal.upper()} | "
                f"Edge {mc.edge_score:.3f} | src:{src} | "
                f"Ens:{mc.ens_agreement:.2f}({mc.ens_n_runs}r) | "
                f"ToD:x{mc.tod_multiplier:.3f}",
                fontsize=9, color=OR, loc="right", pad=12,
            )
        a1.set_xlabel("Ticks (history | forecast)",
                      fontsize=9, color=GY)
        a1.set_ylabel("Price", fontsize=9, color=GY)
        a1.legend(fontsize=7.5, framealpha=0.2, facecolor=BG,
                  edgecolor=GY, ncol=7, loc="upper left")
        a1.grid(True, lw=0.35, color=DG)

        # Panel 2: Return distribution
        a2 = fig.add_subplot(gs[1, 0])
        a2.set_facecolor(AX)
        lr = dm.log_returns(sym)
        if len(lr) > 50:
            lo  = np.percentile(lr, 0.5)
            hi  = np.percentile(lr, 99.5)
            lc  = lr[(lr >= lo) & (lr <= hi)]
            n_b = min(100, max(30, len(lc) // 20))
            a2.hist(lc, bins=n_b, color=BL, alpha=0.65,
                    density=True, label="Returns", edgecolor="none")
            if gp and gp.sigma > 1e-10:
                x = np.linspace(lc.min(), lc.max(), 500)
                a2.plot(x, norm.pdf(x, gp.mu, gp.sigma),
                        color=GR, lw=2.0, label="GBM fit")
                thr = JUMP_THRESHOLD * gp.sigma
                a2.axvline(gp.mu + thr, color=RD, lw=0.9,
                           ls="--", alpha=0.7)
                a2.axvline(gp.mu - thr, color=RD, lw=0.9,
                           ls="--", alpha=0.7)
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

        # Panel 3: Vol Regime / Drift panel
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

        # Panel 4: Probability gauges
        a4 = fig.add_subplot(gs[1, 2])
        a4.set_facecolor(AX)
        if mc:
            self._prob_gauges(a4, mc, profile)
        a4.set_title("Probability Gauges",
                     fontsize=9, color=GY, pad=5)
        a4.grid(True, axis="x", lw=0.25)

        # Panel 5: ToD heatmap
        a5 = fig.add_subplot(gs[1, 3])
        a5.set_facecolor(AX)
        self._tod_panel(a5, sym, tod)
        a5.set_title("Time-of-Day Profile",
                     fontsize=9, color=GY, pad=5)
        a5.grid(True, lw=0.25)

        # Panel 6: Summary bar
        a6 = fig.add_subplot(gs[2, :])
        a6.set_facecolor(BG)
        a6.axis("off")
        self._param_box(a6, sym, dm, mc, gp, jp, vm,
                        note, profile, tod)

        buf = BytesIO()
        try:
            plt.savefig(buf, format="png", dpi=115,
                        bbox_inches="tight", facecolor=BG)
        finally:
            plt.close(fig)
        buf.seek(0)
        return buf

    @staticmethod
    def _vol_regime_panel(ax, lr, mc, profile):
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
            seg   = lr[i-window:i]
            std_s = max(float(np.std(seg, ddof=1)), 1e-10)
            mu_s  = float(np.mean(seg))
            se    = std_s / math.sqrt(len(seg))
            zscores.append(float(np.clip(
                mu_s / max(se, 1e-12), -5, 5
            )))
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
            ax.bar(xvals[pos_ns],  zscores[pos_ns],
                   color=GR2, alpha=0.3, width=w)
        if np.any(neg_ns):
            ax.bar(xvals[neg_ns],  zscores[neg_ns],
                   color=RD2, alpha=0.3, width=w)
        ax.axhline(thr,  color=YL, lw=1.2, ls="--", alpha=0.8)
        ax.axhline(-thr, color=YL, lw=1.2, ls="--", alpha=0.8)
        ax.axhline(0,    color=GY, lw=0.8, alpha=0.5)
        if mc and mc.vol_regime:
            vr  = mc.vol_regime
            col = (GR2 if vr.signal == "bullish"
                   else (RD2 if vr.signal == "bearish" else GY))
            ax.set_title(
                f"[{vr.signal.upper()}] "
                f"z={vr.normalized_momentum:+.2f}",
                fontsize=8, color=col, pad=3,
            )
        ax.set_ylabel("Mom z-score", fontsize=8)
        ax.set_xlabel("Tick index",  fontsize=8)

    @staticmethod
    def _drift_panel(ax, lr, mc, profile):
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
                tstats.append(0.0); pvalues.append(1.0)
            else:
                t = mu / (std / math.sqrt(len(seg)))
                p = float(2 * (1 - norm.cdf(abs(t))))
                tstats.append(float(t)); pvalues.append(p)
        if not tstats:
            return
        tstats  = np.array(tstats)
        pvalues = np.array(pvalues)
        xvals   = np.array(indices)
        w       = max(1, step)
        sig     = pvalues < 0.15
        pos_sig = (tstats >= 0) & sig
        neg_sig = (tstats <  0) & sig
        pos_ns  = (tstats >= 0) & ~sig
        neg_ns  = (tstats <  0) & ~sig
        if np.any(pos_sig):
            ax.bar(xvals[pos_sig], tstats[pos_sig],
                   color=GR2, alpha=0.9, width=w)
        if np.any(neg_sig):
            ax.bar(xvals[neg_sig], tstats[neg_sig],
                   color=RD2, alpha=0.9, width=w)
        if np.any(pos_ns):
            ax.bar(xvals[pos_ns],  tstats[pos_ns],
                   color=GR2, alpha=0.3, width=w)
        if np.any(neg_ns):
            ax.bar(xvals[neg_ns],  tstats[neg_ns],
                   color=RD2, alpha=0.3, width=w)
        thr = profile.drift_tstat_min
        ax.axhline(thr,  color=YL, lw=1.2, ls="--", alpha=0.8)
        ax.axhline(-thr, color=YL, lw=1.2, ls="--", alpha=0.8)
        ax.axhline(0,    color=GY, lw=0.8, alpha=0.5)
        if mc and mc.drift_signal:
            ds  = mc.drift_signal
            col = (GR2 if ds.direction == "bullish"
                   else (RD2 if ds.direction == "bearish" else GY))
            sig_str = " ✓" if ds.pvalue < 0.15 else " ✗"
            ax.set_title(
                f"[{ds.direction.upper()}] "
                f"t={ds.tstat:+.2f}{sig_str}",
                fontsize=8, color=col, pad=3,
            )
        ax.set_ylabel("t-stat",     fontsize=8)
        ax.set_xlabel("Tick index", fontsize=8)

    @staticmethod
    def _tod_panel(ax, sym, tod):
        tod._ensure(sym)
        buckets = tod._buckets[sym]
        mults   = [b.multiplier for b in buckets]
        hours   = [i / 2 for i in range(TOD_BUCKETS)]
        colors  = [
            GR2 if m >= 1.05 else (RD2 if m <= 0.95 else GY)
            for m in mults
        ]
        ax.bar(hours, mults, width=0.45, color=colors, alpha=0.75)
        ax.axhline(1.0, color=YL, lw=1.0, ls="--",
                   alpha=0.7, label="Neutral")
        ax.axhline(TOD_MULT_MAX, color=GR2, lw=0.7,
                   ls=":", alpha=0.5)
        ax.axhline(TOD_MULT_MIN, color=RD2, lw=0.7,
                   ls=":", alpha=0.5)
        cur_h = _tod_bucket() / 2
        ax.axvline(cur_h, color=OR, lw=1.5, ls="-",
                   alpha=0.9, label="NOW")
        ax.set_xlim(0, 24)
        ax.set_ylim(TOD_MULT_MIN - 0.02, TOD_MULT_MAX + 0.02)
        ax.set_xlabel("UTC Hour",   fontsize=8)
        ax.set_ylabel("Multiplier", fontsize=8)
        ax.legend(fontsize=7, framealpha=0.15, facecolor=BG)
        b = buckets[_tod_bucket()]
        ax.set_title(
            f"Now: x{b.multiplier:.3f} "
            f"wr={b.win_rate:.0%} n={b.count:.0f}",
            fontsize=8, color=OR, pad=3,
        )

    @staticmethod
    def _prob_box(ax, mc, profile):
        sig_c = {
            "bullish":       GR, "bearish":       RD,
            "jump_imminent": OR, "neutral":        GY,
        }.get(mc.signal, GY)
        ts_line = ""
        if mc.trade_setup:
            ts    = mc.trade_setup
            q_str = (
                f" {_stars(ts.quality.stars)}"
                if ts.quality else ""
            )
            ts_line = (
                f"\n{ts.direction} RR1:{ts.rr_ratio:.2f} "
                f"P={ts.prob_target:.1%} "
                f"EV={ts.expected_value:+.3f}%{q_str}"
            )
        else:
            ts_line = "\nNo valid setup (quality gates)"
        vr_line = ""
        if mc.vol_regime:
            vr = mc.vol_regime
            vr_line = (
                f"\nVolReg: {vr.signal.upper()} "
                f"z={vr.normalized_momentum:+.2f}"
            )
        drift_line = ""
        if mc.drift_signal:
            ds       = mc.drift_signal
            sig_icon = "✓" if ds.pvalue < 0.15 else "✗"
            drift_line = (
                f"\nDrift: {ds.direction.upper()} "
                f"t={ds.tstat:+.2f} {sig_icon}"
            )
        pj_line = ""
        if mc.prob_jump is not None:
            pj_line = (
                f"\nP(spike {mc.horizon}t): "
                f"{mc.prob_jump:.1%}"
            )
        ens_line = (
            f"\nEns: {mc.ens_agreement:.3f}"
            f"({mc.ens_n_runs}r)"
            + (" ⚠️" if mc.ens_agreement < ENSEMBLE_AGREEMENT_FLOOR
               else " ✓")
        )
        txt = (
            f"Signal: {mc.signal.upper()} [{profile.name}]\n"
            f"Edge: {mc.edge_score:.3f} | "
            f"CI: {mc.ci_width_pct*100:.4f}%\n"
            f"P(up): {mc.prob_hit_up:.1%}  "
            f"P(dn): {mc.prob_hit_down:.1%}"
            f"{vr_line}{drift_line}{pj_line}"
            f"{ens_line}{ts_line}"
            f"\nToD: x{mc.tod_multiplier:.3f}"
        )
        ax.text(
            0.01, 0.97, txt,
            transform=ax.transAxes,
            fontsize=7.8, va="top", color=WH,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.45",
                      fc="#0d1b2a", ec=sig_c, alpha=0.92),
        )

    @staticmethod
    def _prob_gauges(ax, mc, profile):
        labels = [
            "P(up\nhorizon)", "P(hit\ntarget)", "P(hit\nstop)",
        ]
        vals   = [
            mc.prob_up_horizon, mc.prob_hit_up, mc.prob_hit_down,
        ]
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
                else (RD if vr.signal == "bearish" else GY)
            )
        elif mc.drift_signal:
            ds = mc.drift_signal
            labels.append("Drift\nedge")
            vals.append(ds.edge_score)
            colors.append(
                GR if ds.direction == "bullish"
                else (RD if ds.direction == "bearish" else GY)
            )
        if mc.prob_jump is not None:
            labels.append(f"P(spike\n{mc.horizon}t)")
            vals.append(mc.prob_jump)
            colors.append(
                OR if mc.prob_jump > profile.spike_min_prob else GY
            )
        if mc.trade_setup:
            ts = mc.trade_setup
            labels.append("R/R\nrating")
            rr_norm = min(ts.rr_ratio / 4.0, 1.0)
            vals.append(rr_norm)
            colors.append(
                GR if rr_norm > 0.5
                else (YL if rr_norm > 0.3 else RD)
            )
            labels.append("EV\n(norm)")
            ev_norm = float(np.clip(
                ts.expected_value / 0.5, 0.0, 1.0
            ))
            vals.append(ev_norm)
            colors.append(
                GR if ev_norm > 0.3
                else (YL if ev_norm > 0.1 else RD)
            )
            # Signal quality gauge for vol signals
            if ts.quality:
                labels.append("Signal\nQuality")
                q_norm = ts.quality.stars / 5.0
                vals.append(q_norm)
                colors.append(
                    GR  if ts.quality.stars >= 4
                    else (YL if ts.quality.stars == 3 else RD)
                )

        labels.append("Bootstrap\nagree")
        vals.append(mc.ens_agreement)
        colors.append(
            GR  if mc.ens_agreement >= 0.80
            else (YL if mc.ens_agreement >= ENSEMBLE_AGREEMENT_FLOOR
                  else RD)
        )

        bars = ax.barh(labels, vals, color=colors,
                       alpha=0.85, height=0.55)
        ax.set_xlim(0, 1)
        ax.axvline(0.5, color=GY, lw=0.9, ls="--", alpha=0.6)
        for bar, v, lbl in zip(bars, vals, labels):
            if "R/R" in lbl and mc.trade_setup:
                disp = f"1:{mc.trade_setup.rr_ratio:.1f}"
            elif "EV" in lbl and mc.trade_setup:
                disp = f"{mc.trade_setup.expected_value:+.3f}%"
            elif "Quality" in lbl:
                disp = (
                    f"{mc.trade_setup.quality.stars}/5"
                    if mc.trade_setup and mc.trade_setup.quality
                    else "—"
                )
            elif "Bootstrap" in lbl:
                disp = f"{v:.3f}({mc.ens_n_runs}r)"
            else:
                disp = f"{v:.1%}"
            ax.text(
                min(v + 0.03, 0.90),
                bar.get_y() + bar.get_height() / 2,
                disp, va="center", fontsize=8.5,
                color=WH, fontweight="bold",
            )

    def _param_box(self, ax, sym, dm, mc, gp, jp,
                   vm, note, profile, tod):
        S0v  = dm.last(sym)
        s0s  = f"{S0v:.6f}" if S0v else "N/A"
        rate = dm.observed_tick_rate(sym)
        b    = tod._buckets.get(
            sym, [TodBucket()] * TOD_BUCKETS
        )[_tod_bucket()]
        info = [
            f"  | {REVERSE_MAP.get(sym, sym)} | "
            f"PRICE={s0s} | TICKS={dm.n(sym):,} | "
            f"RATE={rate:.2f}t/s | PROFILE={profile.name} | "
            f"ENGINE="
            f"{'VOL_REGIME+MC' if profile.vol_regime_enabled else ('DRIFT+MC' if profile.drift_signal_enabled else 'MC+SPIKE')}"
            f" | {'LIVE' if dm._connected else 'RECON'} | "
            f"SQEv4.5 | ToD={b.multiplier:.3f}"
            + (
                f" | Ens={mc.ens_agreement:.3f}({mc.ens_n_runs}r)"
                + (
                    f" | Q={mc.trade_setup.quality.stars}/5"
                    if mc.trade_setup and mc.trade_setup.quality
                    else ""
                )
                if mc else ""
            )
        ]
        if gp and gp.n_obs > 0:
            ann = vm.ann_vol(gp) * 100
            dev = vm.deviation(gp)
            info.append(
                f"  | GBM mu={gp.mu:.8f} "
                f"ewma={gp.sigma_ewma:.8f} "
                f"ann={ann:.3f}% "
                f"KS={gp.ks_pvalue:.4f} "
                f"FQ={gp.fit_quality:.3f}"
                + (f" Dev={dev:+.1f}%"
                   if gp.advertised_vol else "")
            )
        if jp and jp.n_obs > 0:
            et = (1 / jp.lam_posterior
                  if jp.lam_posterior > 1e-10
                  else float("inf"))
            info.append(
                f"  | JUMP lam_post={jp.lam_posterior:.6f}"
                f"(~1/{et:.0f}t) "
                f"nj={jp.n_jumps} "
                f"dev={jp.lam_deviation:+.1f}% "
                f"hz={jp.hazard_intensity:.0%} "
                f"p60={jp.spike_mag_p60*100:.3f}%"
            )
        if mc:
            rng  = (
                (mc.p95[-1] - mc.p5[-1]) / max(mc.S0, 1e-10) * 100
            )
            ts_s = ""
            if mc.trade_setup:
                ts   = mc.trade_setup
                q_s  = (
                    f" Q={ts.quality.stars}/5"
                    if ts.quality else ""
                )
                ts_s = (
                    f" | {ts.direction} "
                    f"RR1:{ts.rr_ratio:.2f} "
                    f"P={ts.prob_target:.2%} "
                    f"EV={ts.expected_value:+.4f}%{q_s} "
                    f"src={ts.signal_source}"
                )
            vr_s = ""
            if mc.vol_regime:
                vr   = mc.vol_regime
                vr_s = (
                    f" | vr={vr.signal} "
                    f"z={vr.normalized_momentum:+.3f} "
                    f"k={vr.k_adaptive:.3f}"
                )
            info.append(
                f"  | MC({mc.horizon}t) "
                f"CI={mc.ci_width_pct*100:.4f}% "
                f"90w={rng:.4f}% "
                f"edge={mc.edge_score:.4f} "
                f"sig={mc.signal}"
                f"{ts_s}{vr_s}"
            )
        info += [
            f"  | {note}",
            f"  | Gates v4.5: "
            f"P>={MIN_PROB_TARGET:.0%} EV>0 "
            f"RR>={MIN_RR_GLOBAL:.1f} "
            f"Ens>={ENSEMBLE_AGREEMENT_FLOOR:.2f} "
            f"Bootstrap N={N_BOOTSTRAP_RUNS} "
            f"Spike gate: raw_conf>={SPIKE_MIN_CONFIDENCE:.0%} only",
            "  | No model guarantees profit. Max 1-2% risk.",
        ]
        ax.text(
            0.004, 0.95, "\n".join(info),
            transform=ax.transAxes, fontsize=6.8,
            va="top", color=WH, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.45",
                      fc=AX, ec=GY, alpha=0.93),
        )

# ============================================================================
# KEYBOARD HELPERS
# ============================================================================
def _tf_kb(sym: str,
           prefix: str = "tf") -> List[List[InlineKeyboardButton]]:
    return [[
        InlineKeyboardButton("1m",  callback_data=f"{prefix}:{sym}:1m"),
        InlineKeyboardButton("5m",  callback_data=f"{prefix}:{sym}:5m"),
        InlineKeyboardButton("15m", callback_data=f"{prefix}:{sym}:15m"),
        InlineKeyboardButton("30m", callback_data=f"{prefix}:{sym}:30m"),
        InlineKeyboardButton("1h",  callback_data=f"{prefix}:{sym}:1h"),
    ]]

def _tick_kb(sym: str,
             prefix: str = "tk") -> List[List[InlineKeyboardButton]]:
    return [[
        InlineKeyboardButton("50t",  callback_data=f"{prefix}:{sym}:50t"),
        InlineKeyboardButton("100t", callback_data=f"{prefix}:{sym}:100t"),
        InlineKeyboardButton("300t", callback_data=f"{prefix}:{sym}:300t"),
        InlineKeyboardButton("600t", callback_data=f"{prefix}:{sym}:600t"),
    ]]

def _action_kb(
    sym: str, tf_key: str, cat: str
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

# ============================================================================
# TELEGRAM BOT
# ============================================================================
class SQEBotV45:
    def __init__(self):
        self.pe       = PersistenceEngine()
        self.tod      = TimeOfDayProfile()
        tod_data      = self.pe.get_tod_data()
        if tod_data:
            try:
                self.tod.from_dict(tod_data)
                log.info("ToD profiles loaded.")
            except Exception as e:
                log.warning(f"ToD load: {e}")
        self.dm       = DataManager()
        self.vm       = VolatilityModel()
        self.jm       = JumpDiffusionModel()
        self.mce      = MCEngine(self.tod)
        self.cg       = ChartGen()
        self.pm       = PatternMemory(tod_profile=self.tod)
        self.adaptive = AdaptiveThresholds(self.pe)
        self.ssa      = SpikeSuppressionAnalyser()
        self.vqa      = VolSignalQualityAnalyser()
        self.ae       = AlertEngine(
            self.mce, self.pm, self.pe, self.adaptive
        )
        self.narr     = NarrativeEngine(self.vm, self.jm)
        self.se       = SpikeEngine(self.jm)
        self.de       = DriftEngine()
        self.vre      = VolatilityRegimeEngine(self.vm)
        self.tsb      = TradeSetupBuilder()
        self.cr       = ConflictResolver()
        self._chats:  Set[int]             = self.pe.get_chats()
        self._states: Dict[int, UserState] = self.pe.get_user_states()
        self._ws_task = None
        self._al_task = None
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

    async def _on_tick(self, sym: str,
                       price: float, ts: float):
        self.pm.resolve(
            sym, price,
            bot=self._bot_ref, chats=self._chats,
            adaptive=self.adaptive,
        )
        rate = self.dm.observed_tick_rate(sym)
        self.tod.update(sym, True, 0.0, 0.0, 0.0, rate)

        # v4.5: detect live spikes for drought tracking
        if _cat(sym) in ("boom", "crash"):
            lr = self.dm.log_returns(sym)
            if len(lr) >= 5:
                recent_lr = lr[-5:]
                med = float(np.median(lr))
                mad = max(
                    float(np.median(np.abs(lr - med))) * 1.4826,
                    1e-10,
                )
                if abs(recent_lr[-1] - med) > JUMP_THRESHOLD * mad:
                    tick_idx = self.dm.tick_count(sym)
                    self.ssa.record_spike(sym, tick_idx)

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------
    async def cmd_start(self, update: Update,
                        ctx: ContextTypes.DEFAULT_TYPE):
        self._track(update)
        cid   = update.effective_chat.id
        state = self._state(cid)
        conn  = self.dm._connected
        total = sum(self.dm.n(s) for s in ALL_SYMBOLS.values())
        kb    = [
            [InlineKeyboardButton(
                "🚨 Spike Scanner",
                callback_data="menu:spike_scan")],
            [InlineKeyboardButton(
                "🔔 Alerts ON"
                if state.alerts_enabled
                else "🔕 Alerts OFF",
                callback_data="toggle:alerts")],
        ]
        st_icon = "🟢" if conn else "🔴"
        await update.message.reply_text(
            f"*SYNTHETIC QUANT*\n\n"
            f"{st_icon} `{'Live' if conn else 'Reconnecting'}`\n"
            f"📊 `{total:,} ticks | "
            f"{len(ALL_SYMBOLS)} instruments`\n\n"
            f"*Alerts:* `{'ON' if state.alerts_enabled else 'OFF'}`\n"
            f"*v4.5:* Spike quality {_stars(5)} rating on every alert\n"
            "Enjoy Profitable Signals - Message the developers @ +2348130009747",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup(kb),
        )

    async def cmd_spike(self, update: Update,
                        ctx: ContextTypes.DEFAULT_TYPE):
        self._track(update)
        if not ctx.args:
            await update.message.reply_text(
                "Usage: `/spike <symbol>`",
                parse_mode=ParseMode.MARKDOWN,
            )
            return
        sym = _resolve(" ".join(ctx.args))
        if not sym:
            await update.message.reply_text("Unknown symbol.")
            return
        if _cat(sym) not in ("boom", "crash"):
            await update.message.reply_text(
                f"Spike detection: Boom/Crash only.\n"
                f"`{_friendly(sym)}` is {_cat(sym).upper()}.",
                parse_mode=ParseMode.MARKDOWN,
            )
            return
        await self._do_spike_check(
            update.effective_chat.id, ctx, sym
        )

    async def cmd_drift(self, update: Update,
                        ctx: ContextTypes.DEFAULT_TYPE):
        self._track(update)
        if not ctx.args:
            await update.message.reply_text(
                "Usage: `/drift <symbol>`",
                parse_mode=ParseMode.MARKDOWN,
            )
            return
        sym = _resolve(" ".join(ctx.args))
        if not sym:
            await update.message.reply_text("Unknown symbol.")
            return
        await self._do_drift_report(
            update.effective_chat.id, ctx, sym
        )

    async def cmd_tod(self, update: Update,
                      ctx: ContextTypes.DEFAULT_TYPE):
        self._track(update)
        if not ctx.args:
            await update.message.reply_text(
                "Usage: `/tod <symbol>`",
                parse_mode=ParseMode.MARKDOWN,
            )
            return
        sym = _resolve(" ".join(ctx.args))
        if not sym:
            await update.message.reply_text("Unknown symbol.")
            return
        await self._do_tod_report(
            update.effective_chat.id, ctx, sym
        )

    async def cmd_prob(self, update: Update,
                       ctx: ContextTypes.DEFAULT_TYPE):
        self._track(update)
        if not ctx.args:
            await update.message.reply_text(
                "Usage: `/prob <symbol>`",
                parse_mode=ParseMode.MARKDOWN,
            )
            return
        sym = _resolve(" ".join(ctx.args))
        if not sym:
            await update.message.reply_text("Unknown symbol.")
            return
        cid = update.effective_chat.id
        self._state(cid).pending_sym = sym
        kb  = _tf_kb(sym, "prob_tf") + _tick_kb(sym, "prob_tk")
        await update.message.reply_text(
            f"*{_friendly(sym)}* — Select timeframe:",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup(kb),
        )

    async def cmd_parameters(self, update: Update,
                              ctx: ContextTypes.DEFAULT_TYPE):
        self._track(update)
        if not ctx.args:
            await update.message.reply_text(
                "Usage: `/parameters <symbol>`",
                parse_mode=ParseMode.MARKDOWN,
            )
            return
        sym = _resolve(" ".join(ctx.args))
        if not sym:
            await update.message.reply_text("Unknown symbol.")
            return
        await self._send_parameters(
            update.effective_chat.id, ctx, sym
        )

    async def cmd_status(self, update: Update,
                         ctx: ContextTypes.DEFAULT_TYPE):
        self._track(update)
        lines = [
            "*Data Feed — SQE v4.5*\n",
            f"WS: {'🟢 Live' if self.dm._connected else '🔴 Reconnecting'}\n",
            f"{'Symbol':<14} {'Ticks':>7} {'Rate':>7} "
            f"{'Profile':>8} {'Engine':>12}",
            "─" * 60,
        ]
        for fn, ds in ALL_SYMBOLS.items():
            n       = self.dm.n(ds)
            rate    = self.dm.observed_tick_rate(ds)
            profile = effective_profile(ds)
            engine  = (
                "VOL_REGIME" if profile.vol_regime_enabled
                else (
                    "DRIFT+MC" if profile.drift_signal_enabled
                    else "MC+SPIKE"
                )
            )
            tod_m   = self.tod.multiplier(ds)
            ok      = "✅" if n >= MIN_TICKS else f"({n})"
            lines.append(
                f"{fn:<14} {n:>7,} {rate:>6.2f}t/s "
                f"{profile.name:>8} {engine:>12} "
                f"x{tod_m:.2f} {ok}"
            )
        await _safe_send_message(
            ctx.bot, update.effective_chat.id,
            "\n".join(lines),
        )

    async def cmd_results(self, update: Update,
                          ctx: ContextTypes.DEFAULT_TYPE):
        self._track(update)
        if not ctx.args:
            await update.message.reply_text(
                "Usage: `/results <symbol>`",
                parse_mode=ParseMode.MARKDOWN,
            )
            return
        sym = _resolve(" ".join(ctx.args))
        if not sym:
            await update.message.reply_text("Unknown symbol.")
            return
        await self._send_results(
            update.effective_chat.id, ctx, sym
        )

    async def cmd_watch(self, update: Update,
                        ctx: ContextTypes.DEFAULT_TYPE):
        self._track(update)
        cid = update.effective_chat.id
        if not ctx.args:
            await update.message.reply_text(
                "Usage: `/watch <symbol>`",
                parse_mode=ParseMode.MARKDOWN,
            )
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
            f"Watchlist: "
            f"{', '.join(_friendly(s) for s in st.watchlist)}",
            parse_mode=ParseMode.MARKDOWN,
        )

    async def cmd_unwatch(self, update: Update,
                          ctx: ContextTypes.DEFAULT_TYPE):
        self._track(update)
        cid = update.effective_chat.id
        if not ctx.args:
            await update.message.reply_text(
                "Usage: `/unwatch <symbol>`",
                parse_mode=ParseMode.MARKDOWN,
            )
            return
        sym = _resolve(" ".join(ctx.args))
        if not sym:
            await update.message.reply_text("Unknown symbol.")
            return
        st = self._state(cid)
        if sym in st.watchlist:
            st.watchlist.remove(sym)
        self._save_states()
        await update.message.reply_text(
            f"Removed. Watchlist: "
            f"{', '.join(_friendly(s) for s in st.watchlist) or 'empty'}",
            parse_mode=ParseMode.MARKDOWN,
        )

    async def cmd_quick_sym(self, update: Update,
                            ctx: ContextTypes.DEFAULT_TYPE):
        self._track(update)
        cid = update.effective_chat.id
        cmd = (
            update.message.text.lstrip("/")
                               .split("@")[0].upper()
        )
        sym_map = {
            "V10": "R_10",   "V25": "R_25",   "V50": "R_50",
            "V75": "R_75",   "V100": "R_100", "V250": "R_250",
            "BOOM300":   "BOOM300N",  "BOOM500":   "BOOM500",
            "BOOM600":   "BOOM600N",  "BOOM900":   "BOOM900",
            "BOOM1000":  "BOOM1000",
            "CRASH300":  "CRASH300N", "CRASH500":  "CRASH500",
            "CRASH600":  "CRASH600N", "CRASH900":  "CRASH900",
            "CRASH1000": "CRASH1000",
            "STEP":      "stpRNG",    "STEPINDEX": "stpRNG",
            "JUMP10":  "JD10", "JUMP25":  "JD25",
            "JUMP50":  "JD50", "JUMP75":  "JD75",
            "JUMP100": "JD100",
        }
        sym = sym_map.get(cmd)
        if not sym:
            await update.message.reply_text(
                "Unknown quick command."
            )
            return
        self._state(cid).pending_sym = sym
        cat     = _cat(sym)
        profile = effective_profile(sym)
        kb      = _tf_kb(sym, "tf") + _tick_kb(sym, "tk")
        if cat in ("boom", "crash"):
            kb.append([InlineKeyboardButton(
                "⚡ Spike Check",
                callback_data=f"spike:{sym}",
            )])
        engine_str = (
            "VOL_REGIME+MC" if profile.vol_regime_enabled
            else (
                "DRIFT+MC" if profile.drift_signal_enabled
                else "MC+SPIKE"
            )
        )
        tod_m = self.tod.multiplier(sym)
        await update.message.reply_text(
            f"*{_friendly(sym)}* [{profile.name}] "
            f"Engine: *{engine_str}* | ToD: x{tod_m:.3f}\n"
            f"Select timeframe:",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup(kb),
        )

    # ------------------------------------------------------------------
    # Callback query handler
    # ------------------------------------------------------------------
    async def cb_query(self, update: Update,
                       ctx: ContextTypes.DEFAULT_TYPE):
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
                sym = d[4:]
                cat = _cat(sym)
                self._state(cid).pending_sym = sym
                kb  = _tf_kb(sym, "tf") + _tick_kb(sym, "tk")
                if cat in ("boom", "crash"):
                    kb.append([InlineKeyboardButton(
                        "⚡ Spike Check",
                        callback_data=f"spike:{sym}",
                    )])
                await q.message.reply_text(
                    f"*{_friendly(sym)}* — Select timeframe:",
                    parse_mode=ParseMode.MARKDOWN,
                    reply_markup=InlineKeyboardMarkup(kb),
                )
            elif d.startswith("tf:"):
                _, sym, tf_key = d.split(":")
                h, hl = self.dm.ticks_for_timeframe(sym, tf_key)
                await q.message.reply_text(
                    f"Analysing *{_friendly(sym)}* — "
                    f"*{hl}* ({h}t) ...",
                    parse_mode=ParseMode.MARKDOWN,
                )
                await self._do_analysis(
                    cid, ctx, sym, h, hl, tf_key
                )
            elif d.startswith("tk:"):
                parts     = d.split(":")
                sym, hkey = parts[1], parts[2]
                h         = HORIZONS.get(hkey, 300)
                hl        = HORIZON_LABELS.get(hkey, "300 Ticks")
                await q.message.reply_text(
                    f"Analysing *{_friendly(sym)}* — *{hl}* ...",
                    parse_mode=ParseMode.MARKDOWN,
                )
                await self._do_analysis(cid, ctx, sym, h, hl)
            elif d.startswith("prob_tf:"):
                _, sym, tf_key = d.split(":")
                h, hl = self.dm.ticks_for_timeframe(sym, tf_key)
                await self._send_prob_table(cid, ctx, sym, h, hl)
            elif d.startswith("prob_tk:"):
                parts     = d.split(":")
                sym, hkey = parts[1], parts[2]
                h         = HORIZONS.get(hkey, 300)
                hl        = HORIZON_LABELS.get(hkey, "300 Ticks")
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
                    parse_mode=ParseMode.MARKDOWN,
                )
            elif d.startswith("rmwatch:"):
                sym = d[8:]
                st  = self._state(cid)
                if sym in st.watchlist:
                    st.watchlist.remove(sym)
                self._save_states()
                await q.message.reply_text(
                    f"Removed *{_friendly(sym)}*.",
                    parse_mode=ParseMode.MARKDOWN,
                )
            elif d == "toggle:alerts":
                st = self._state(cid)
                st.alerts_enabled = not st.alerts_enabled
                self._save_states()
                s = (
                    "ENABLED 🔔" if st.alerts_enabled
                    else "DISABLED 🔕"
                )
                await q.message.reply_text(
                    f"Alerts *{s}*.\n"
                    + (
                        f"Gates v4.5:\n"
                        f"  P>={MIN_PROB_TARGET:.0%} (adaptive)\n"
                        f"  EV>0 RR>={MIN_RR_GLOBAL:.1f}\n"
                        f"  Ens>={ENSEMBLE_AGREEMENT_FLOOR:.2f} "
                        f"({N_BOOTSTRAP_RUNS} runs)\n"
                        f"  Spike: raw conf≥{SPIKE_MIN_CONFIDENCE:.0%}\n"
                        f"  Quality {_stars(5)} shown on every alert\n"
                        f"  STRONG/MODERATE only"
                        if st.alerts_enabled
                        else "Tap to re-enable."
                    ),
                    parse_mode=ParseMode.MARKDOWN,
                )
            elif d == "menu:spike_scan":
                await self._do_spike_scan(cid, ctx)
        except Exception as e:
            log.error(f"cb_query error: {e}")
            try:
                await q.message.reply_text(
                    f"Error: {str(e)[:100]}"
                )
            except Exception:
                pass

    async def _handle_menu(self, q, cat: str,
                           cid: int, ctx):
        if cat == "spike_scan":
            await self._do_spike_scan(cid, ctx)

    # ------------------------------------------------------------------
    # Core analysis handler
    # ------------------------------------------------------------------
    async def _do_analysis(
        self, cid: int, ctx, sym: str,
        horizon: int, hlabel: str,
        tf_key: Optional[str] = None,
    ):
        fname   = _friendly(sym)
        n       = self.dm.n(sym)
        S0      = self.dm.last(sym)
        profile = effective_profile(sym)
        cat     = _cat(sym)
        if n < MIN_TICKS or S0 is None:
            await ctx.bot.send_message(
                cid,
                f"*{fname}* [{profile.name}]: {n} ticks "
                f"(need {MIN_TICKS}+). Retry in ~30s.",
                parse_mode=ParseMode.MARKDOWN,
            )
            return

        lr = self.dm.log_returns(sym)
        gp = jp = None
        if cat in ("vol", "step"):
            gp = self.vm.fit(sym, lr)
        else:
            jp = self.jm.fit_unbiased(sym, lr)
            gp = GBMParams(
                mu=jp.mu, sigma=jp.sigma,
                sigma_ewma=jp.sigma, n_obs=jp.n_obs,
                fit_quality=0.80,
            )

        loop = asyncio.get_running_loop()
        mc   = await loop.run_in_executor(
            None, self.mce.run,
            sym, self.dm, horizon, hlabel,
            self.pm, None, tf_key,
        )
        if mc is None:
            await ctx.bot.send_message(
                cid,
                f"Analysis failed for {fname}. Retry.",
                parse_mode=ParseMode.MARKDOWN,
            )
            return

        # Spike assessment with quality info attached
        spike_alert = None
        if profile.spike_enabled and jp:
            tr         = self.dm.observed_tick_rate(sym)
            tod_mult   = self.tod.multiplier(sym)
            prices_arr = self.dm.prices(sym)
            raw_sa     = self.se.assess(
                sym, jp, horizon, tr, tod_mult
            )
            if raw_sa is not None:
                gp_tmp  = self.vm.fit(sym, lr)
                adv     = ADVERTISED_VOL.get(
                    sym, gp_tmp.sigma_ewma
                )
                vol_dev = (
                    (gp_tmp.sigma_ewma - adv)
                    / max(adv, 1e-10)
                )
                suppression = self.ssa.analyse(
                    sym=sym, jp=jp,
                    raw_confidence=raw_sa.confidence,
                    prices=prices_arr,
                    current_tick=self.dm.tick_count(sym),
                    vol_deviation=vol_dev,
                    lr=lr,
                )
                raw_sa.suppression = suppression
                spike_alert = raw_sa  # always attach, never gate

        validated_trade, trade_valid, reason = self.cr.resolve(
            sym, mc, spike_alert, mc.trade_setup
        )
        mc.trade_setup = validated_trade if trade_valid else None

        if mc.signal not in ("neutral",):
            tup = (mc.trade_setup.target
                   if mc.trade_setup else mc.target_up)
            tdn = (mc.trade_setup.invalidation
                   if mc.trade_setup else mc.target_down)
            self.pm.record(
                sym, mc.horizon, mc.signal,
                max(mc.prob_hit_up, mc.prob_hit_down),
                mc.edge_score, S0, tup, tdn,
            )

        note    = self.pm.note(sym, mc.horizon)
        st      = self._state(cid)
        img     = await loop.run_in_executor(
            None, self.cg.make,
            fname, sym, self.dm, mc, gp, jp,
            self.vm, note, self.tod,
        )
        cap     = self._caption(
            fname, sym, mc, gp, jp, note, st.risk_pct
        )
        used_tf = tf_key or "5m"
        kb      = _action_kb(sym, used_tf, cat)
        await _safe_send_photo(
            ctx.bot, cid, photo=img, caption=cap,
            reply_markup=InlineKeyboardMarkup(kb),
        )

        full_text = self._full_analysis(
            fname, sym, mc, gp, jp,
            note, st.risk_pct, hlabel,
        )
        if full_text:
            await _safe_send_message(ctx.bot, cid, full_text)

        # Send spike alert with quality block always attached
        if spike_alert:
            await _safe_send_message(
                ctx.bot, cid,
                self.se.format_standalone(spike_alert),
            )

        if not trade_valid and reason == "spike_primary_conflict":
            await _safe_send_message(
                ctx.bot, cid,
                "*Note:* Trade suppressed — spike direction priority.",
            )

        self.pe.save_tod_data(self.tod.to_dict())

    async def _do_spike_check(
        self, cid: int, ctx, sym: str
    ):
        fname      = _friendly(sym)
        n          = self.dm.n(sym)
        if n < MIN_TICKS:
            await ctx.bot.send_message(
                cid,
                f"{fname}: {n} ticks — need {MIN_TICKS}+.",
                parse_mode=ParseMode.MARKDOWN,
            )
            return
        lr         = self.dm.log_returns(sym)
        jp         = self.jm.fit_unbiased(sym, lr)
        tr         = self.dm.observed_tick_rate(sym)
        tod_mult   = self.tod.multiplier(sym)
        prices_arr = self.dm.prices(sym)
        raw_sa     = self.se.assess(sym, jp, 300, tr, tod_mult)

        if raw_sa is not None:
            gp_tmp  = self.vm.fit(sym, lr)
            adv     = ADVERTISED_VOL.get(sym, gp_tmp.sigma_ewma)
            vol_dev = (
                (gp_tmp.sigma_ewma - adv) / max(adv, 1e-10)
            )
            suppression = self.ssa.analyse(
                sym=sym, jp=jp,
                raw_confidence=raw_sa.confidence,
                prices=prices_arr,
                current_tick=self.dm.tick_count(sym),
                vol_deviation=vol_dev,
                lr=lr,
            )
            raw_sa.suppression = suppression
            report = self.se.format_standalone(raw_sa)
            S0     = self.dm.last(sym) or 0.0
            self.pm.record(
                sym, 300, "spike_imminent",
                raw_sa.confidence, raw_sa.confidence, S0,
                S0 * (1 + raw_sa.magnitude_p75 / 100),
                S0 * (1 - raw_sa.magnitude_p75 / 100),
            )
        else:
            pp     = self.jm.prob_in_n(jp, 300)
            report = (
                f"*Spike Check — {fname}*\n\n"
                f"Status: ⬜ NOT IMMINENT\n"
                f"P(spike 300t): {pp:.1%}\n"
                f"Threshold: raw conf ≥{SPIKE_MIN_CONFIDENCE:.0%}\n"
                f"{RISK_MSG}"
            )
        kb = [[
            InlineKeyboardButton(
                "📊 Full Analysis",
                callback_data=f"sym:{sym}"),
            InlineKeyboardButton(
                "🔄 Refresh",
                callback_data=f"spike:{sym}"),
        ]]
        await _safe_send_message(
            ctx.bot, cid, report,
            reply_markup=InlineKeyboardMarkup(kb),
        )

    async def _do_spike_scan(self, cid: int, ctx):
        await ctx.bot.send_message(
            cid,
            "🔍 Scanning Boom/Crash with quality rating v4.5 ...",
            parse_mode=ParseMode.MARKDOWN,
        )
        results = []
        scan_syms = {**BOOM_SYMBOLS, **CRASH_SYMBOLS}
        for fn, ds in scan_syms.items():
            if self.dm.n(ds) < MIN_TICKS:
                continue
            try:
                lr         = self.dm.log_returns(ds)
                jp         = self.jm.fit_unbiased(ds, lr)
                tr         = self.dm.observed_tick_rate(ds)
                tod_mult   = self.tod.multiplier(ds)
                prices_arr = self.dm.prices(ds)
                raw_sa     = self.se.assess(
                    ds, jp, 300, tr, tod_mult
                )
                if raw_sa is None:
                    continue
                gp_tmp  = self.vm.fit(ds, lr)
                adv     = ADVERTISED_VOL.get(ds, gp_tmp.sigma_ewma)
                vol_dev = (
                    (gp_tmp.sigma_ewma - adv) / max(adv, 1e-10)
                )
                suppression = self.ssa.analyse(
                    sym=ds, jp=jp,
                    raw_confidence=raw_sa.confidence,
                    prices=prices_arr,
                    current_tick=self.dm.tick_count(ds),
                    vol_deviation=vol_dev,
                    lr=lr,
                )
                raw_sa.suppression = suppression
                results.append((fn, ds, raw_sa))
            except Exception:
                continue

        if not results:
            await ctx.bot.send_message(
                cid,
                "No Boom/Crash spike signals currently "
                f"meeting raw confidence ≥{SPIKE_MIN_CONFIDENCE:.0%}.",
            )
            return

        lines = ["*🚨 Spike Scanner v4.5:*\n"]
        for fn, ds, sa in results[:5]:
            dir_icon = "📈" if sa.direction == "up" else "📉"
            stars    = sa.suppression.stars if sa.suppression else 5
            lines.append(
                f"{dir_icon} `{fn}` | "
                f"{_stars(stars)} | "
                f"Conf:{sa.confidence:.0%} | "
                f"Window:{sa.time_lo_str}–{sa.time_hi_str}"
            )
        kb = [[
            InlineKeyboardButton(fn, callback_data=f"spike:{ds}")
        ] for fn, ds, _ in results[:5]]
        await _safe_send_message(
            ctx.bot, cid, "\n".join(lines),
            reply_markup=InlineKeyboardMarkup(kb),
        )

    async def _do_tod_report(self, cid: int, ctx, sym: str):
        fname   = _friendly(sym)
        profile = effective_profile(sym)
        self.tod._ensure(sym)
        buckets = self.tod._buckets[sym]
        cur_b   = _tod_bucket()
        lines   = [
            f"*Time-of-Day — {fname}* [{profile.name}]\n",
            f"Current: UTC {cur_b//2:02d}:"
            f"{'30' if cur_b%2 else '00'}\n",
            "*Best 5:*",
        ]
        sorted_b = sorted(
            enumerate(buckets),
            key=lambda x: x[1].multiplier,
            reverse=True,
        )[:5]
        for idx, b in sorted_b:
            h    = idx // 2
            mins = "30" if idx % 2 else "00"
            icon = "🟢" if b.multiplier >= 1.05 else "🟡"
            lines.append(
                f"{icon} {h:02d}:{mins} UTC | "
                f"x{b.multiplier:.3f} | "
                f"wr={b.win_rate:.0%} | n={b.count:.0f}"
            )
        lines += ["", RISK_MSG]
        kb = [[
            InlineKeyboardButton(
                "📊 Full Analysis",
                callback_data=f"sym:{sym}"),
            InlineKeyboardButton(
                "🔄 Refresh",
                callback_data=f"tod:{sym}"),
        ]]
        await _safe_send_message(
            ctx.bot, cid, "\n".join(lines),
            reply_markup=InlineKeyboardMarkup(kb),
        )

    async def _do_drift_report(self, cid: int, ctx, sym: str):
        fname      = _friendly(sym)
        n          = self.dm.n(sym)
        profile    = effective_profile(sym)
        if n < MIN_TICKS:
            await ctx.bot.send_message(
                cid,
                f"{fname}: {n} ticks — need {MIN_TICKS}+.",
                parse_mode=ParseMode.MARKDOWN,
            )
            return
        lr         = self.dm.log_returns(sym)
        S0         = self.dm.last(sym) or 1.0
        tod_mult   = self.tod.multiplier(sym)
        prices_arr = self.dm.prices(sym)
        window     = max(min(300 * 2, profile.drift_window), 30)

        if profile.vol_regime_enabled:
            vr = self.vre.analyse(
                sym, lr, S0, 300, profile, tod_mult,
                window_override=window, prices=prices_arr,
            )
            dir_icon = (
                "📈" if vr.signal == "bullish"
                else ("📉" if vr.signal == "bearish" else "⬜")
            )
            sig_icon = (
                "✅ CONFIRMED" if vr.signal != "neutral"
                else "❌ NOT CONFIRMED"
            )
            lines = [
                f"*Vol Regime — {fname}* [{profile.name}]\n",
                f"Signal:   *{vr.signal.upper()}* "
                f"{dir_icon} {sig_icon}",
                f"Strength: *{vr.signal_strength}*",
                f"Regime:   *{vr.regime}*",
                f"z-score:  `{vr.normalized_momentum:+.4f}`",
                f"V-Edge:   `{vr.edge_score:.4f}`",
            ]
            if vr.signal != "neutral":
                gp_f           = self.vm.fit(sym, lr)
                drift_per_tick = float(np.mean(
                    lr[-min(300, len(lr)):]
                )) if len(lr) > 0 else 0.0
                vq = self.vqa.analyse(
                    sym=sym, lr=lr, prices=prices_arr,
                    signal_dir=vr.signal,
                    vol_regime=vr, drift=None,
                    sigma_tick=gp_f.sigma_ewma,
                    drift_per_tick=drift_per_tick,
                    profile=profile,
                )
                lines.append(
                    VolSignalQualityAnalyser.format_quality_block(vq)
                )
            else:
                lines.append(
                    "\n*Signal NOT CONFIRMED* — Monitor for change."
                )
        else:
            ds = self.de.analyse(
                sym, lr, S0, profile, tod_mult,
                window_override=window, prices=prices_arr,
            )
            dir_icon = (
                "📈" if ds.direction == "bullish"
                else ("📉" if ds.direction == "bearish" else "⬜")
            )
            sig_icon = (
                "✅ CONFIRMED" if ds.direction != "neutral"
                else "❌ NOT CONFIRMED"
            )
            lines = [
                f"*Drift — {fname}* [{profile.name}]\n",
                f"Direction: *{ds.direction.upper()}* "
                f"{dir_icon} {sig_icon}",
                f"Strength:  *{ds.signal_strength}*",
                f"t-stat:  `{ds.tstat:+.6f}`"
                + (" ✓" if abs(ds.tstat) >= profile.drift_tstat_min
                   else " ✗"),
                f"p-value: `{ds.pvalue:.6f}`"
                + (" ✓" if ds.pvalue < 0.15 else " ✗"),
                f"D-Edge:  `{ds.edge_score:.4f}`",
            ]
            if ds.direction != "neutral":
                gp_f = self.vm.fit(sym, lr)
                vq   = self.vqa.analyse(
                    sym=sym, lr=lr, prices=prices_arr,
                    signal_dir=ds.direction,
                    vol_regime=None, drift=ds,
                    sigma_tick=gp_f.sigma_ewma,
                    drift_per_tick=ds.drift_per_tick,
                    profile=profile,
                )
                lines.append(
                    VolSignalQualityAnalyser.format_quality_block(vq)
                )
            else:
                lines.append(
                    "\n*Signal NOT CONFIRMED* — Monitor for change."
                )
        lines.append(RISK_MSG)
        kb = [[
            InlineKeyboardButton(
                "📊 Full Analysis",
                callback_data=f"sym:{sym}"),
            InlineKeyboardButton(
                "🔄 Refresh",
                callback_data=f"drift:{sym}"),
        ]]
        await _safe_send_message(
            ctx.bot, cid, "\n".join(lines),
            reply_markup=InlineKeyboardMarkup(kb),
        )

    async def _send_prob_table(
        self, cid: int, ctx, sym: str,
        horizon: int, hlabel: str,
    ):
        fname   = _friendly(sym)
        S0      = self.dm.last(sym)
        n       = self.dm.n(sym)
        profile = effective_profile(sym)
        if not S0 or n < MIN_TICKS:
            await ctx.bot.send_message(
                cid, f"{fname}: {n} ticks — retry."
            )
            return
        results: Dict[str, MCResult] = {}
        for tf_key, tf_label in TIMEFRAMES.items():
            try:
                ticks, lbl = self.dm.ticks_for_timeframe(
                    sym, tf_key
                )
                r = self.mce.run(
                    sym, self.dm, ticks, lbl,
                    self.pm, n=3000, timeframe_key=tf_key,
                )
                if r:
                    results[tf_key] = r
            except Exception:
                pass
        tod_m = self.tod.multiplier(sym)
        lines = [
            f"*{fname}* [{profile.name}] — "
            f"Probability Matrix v4.5\n",
            f"Price: `{S0:.6f}` | Ticks: `{n:,}` | "
            f"ToD: x{tod_m:.3f}\n",
            f"{'TF':<8}|{'P(up)':>7}|{'P(+)':>7}|"
            f"{'P(-)':>7}|{'Edge':>6}|{'RR':>7}|"
            f"{'EV':>7}|{'Q':>3}|{'Ens':>6}",
            "─" * 72,
        ]
        for tf_key, r in results.items():
            rr_str = (f"1:{r.trade_setup.rr_ratio:.1f}"
                      if r.trade_setup else "—")
            q_str  = (
                f"{r.trade_setup.quality.stars}/5"
                if r.trade_setup and r.trade_setup.quality
                else "—"
            )
            ev_str = (f"{r.trade_setup.expected_value:+.3f}"
                      if r.trade_setup else "—")
            ens_str= f"{r.ens_agreement:.2f}"
            se_i   = _se(r.signal)
            lines.append(
                f"{TIMEFRAMES[tf_key]:<8}|"
                f"{r.prob_up_horizon:>7.1%}|"
                f"{r.prob_hit_up:>7.1%}|"
                f"{r.prob_hit_down:>7.1%}|"
                f"{r.edge_score:>6.3f}|"
                f"{rr_str:>7}|{ev_str:>7}|"
                f"{se_i}{q_str:>2}|{ens_str:>6}"
            )
        if results:
            best_tf = max(
                results, key=lambda k: results[k].edge_score
            )
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
                q_s = (
                    f" Quality:{_stars(ts.quality.stars)}"
                    if ts.quality else ""
                )
                lines.append(
                    f"Setup: *{ts.direction}* | "
                    f"RR *1:{ts.rr_ratio:.2f}* | "
                    f"P *{ts.prob_target:.2%}* | "
                    f"EV `{ts.expected_value:+.4f}%`{q_s}"
                )
        lines += ["", RISK_MSG]
        await _safe_send_message(
            ctx.bot, cid, "\n".join(lines)
        )

    async def _send_parameters(self, cid: int, ctx, sym: str):
        fname   = _friendly(sym)
        lr      = self.dm.log_returns(sym)
        n       = self.dm.n(sym)
        cat     = _cat(sym)
        rate    = self.dm.observed_tick_rate(sym)
        profile = effective_profile(sym)
        S0      = self.dm.last(sym) or 1.0
        tod_m   = self.tod.multiplier(sym)
        lines   = [
            f"*Parameters: {fname}* [{profile.name}]\n",
            f"Category: `{cat.upper()}` | "
            f"Ticks: `{n:,}` | Rate: `{rate:.2f}t/s` | "
            f"ToD: x{tod_m:.3f}\n",
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
                    f"*GBM | Regime:{rg}*",
                    f"EWMA sig: `{p.sigma_ewma:.9f}`",
                    f"Ann vol:  `{a:.5f}%`",
                    f"KS pval:  `{p.ks_pvalue:.5f}`"
                    + (" normal" if p.ks_pvalue > 0.05
                       else " fat-tailed"),
                ]
                if p.advertised_vol:
                    lines.append(
                        f"Deviation: `{d:+.2f}%` from target"
                    )
            else:
                jp  = self.jm.fit_unbiased(sym, lr)
                et  = (1 / jp.lam_posterior
                       if jp.lam_posterior > 1e-10
                       else float("inf"))
                ef  = EXPECTED_JUMP_FREQ.get(sym, 0)
                pp  = self.jm.prob_in_n(jp, 300)
                prices_arr = self.dm.prices(sym)
                gp_tmp     = self.vm.fit(sym, lr)
                adv        = ADVERTISED_VOL.get(
                    sym, gp_tmp.sigma_ewma
                )
                vol_dev    = (
                    (gp_tmp.sigma_ewma - adv) / max(adv, 1e-10)
                )
                sup = self.ssa.analyse(
                    sym=sym, jp=jp,
                    raw_confidence=0.97,
                    prices=prices_arr,
                    current_tick=self.dm.tick_count(sym),
                    vol_deviation=vol_dev,
                    lr=lr,
                )
                lines += [
                    "*Jump-Diffusion v4.5 (Bayesian):*",
                    f"lambda post: `{jp.lam_posterior:.7f}` "
                    f"(1/{et:.0f}t)",
                    f"theory lam:  `{jp.expected_lam:.7f}` "
                    f"(1/{ef}t)",
                    f"lambda dev:  `{jp.lam_deviation:+.1f}%`",
                    f"Hazard:      `{jp.hazard_intensity:.0%}`",
                    f"P(spike/300t): `{pp:.2%}`",
                    f"\n*Current Spike Quality Indicators:*",
                    f"  Stars:          `{sup.stars}/5`",
                    f"  Drought:        `{sup.ticks_since_spike}t`",
                    f"  Activity ratio: `{sup.activity_ratio:.2f}`",
                    f"  Vol deviation:  `{vol_dev*100:+.1f}%`",
                    f"  Range ratio:    `{sup.range_ratio:.2f}`",
                    f"  In cooling:     `{'Yes' if sup.in_cooling else 'No'}`",
                ]
        lines += ["", RISK_MSG]
        await _safe_send_message(
            ctx.bot, cid, "\n".join(lines)
        )

    async def _send_results(self, cid: int, ctx, sym: str):
        fname   = _friendly(sym)
        profile = effective_profile(sym)
        n, wr, ae = self.pm.stats(sym)
        tod_m   = self.tod.multiplier(sym)
        if n == 0:
            await ctx.bot.send_message(
                cid,
                f"*Results — {fname}* [{profile.name}]\n\n"
                f"No resolved predictions yet.\n"
                f"ToD: x{tod_m:.3f}",
                parse_mode=ParseMode.MARKDOWN,
            )
            return
        wins   = sum(
            1 for r in self.pm.records
            if r.symbol == sym and r.resolved and r.correct
        )
        losses = n - wins
        icon   = (
            "🟢" if wr >= 0.55
            else ("🔴" if wr < 0.45 else "🟡")
        )
        msg = (
            f"*Results — {fname}* [{profile.name}]\n\n"
            f"*Accuracy:* {icon} *{wr:.0%}*\n"
            f"W:{wins} L:{losses} Total:{n}\n"
            f"Avg edge: {ae:.3f} | ToD: x{tod_m:.3f}"
        )
        await _safe_send_message(ctx.bot, cid, msg)

    def _caption(
        self, fname: str, sym: str, mc: MCResult,
        gp: Optional[GBMParams], jp: Optional[JumpParams],
        note: str, risk_pct: float,
    ) -> str:
        S0      = mc.S0
        n       = self.dm.n(sym)
        se_i    = _se(mc.signal)
        profile = effective_profile(sym)
        L = [
            f"*{fname}* [{profile.name}] — {mc.horizon_label}\n",
            f"Price: `{_safe_md(S0,'.6f')}` | "
            f"Ticks: `{n:,}` | "
            f"ToD: x{mc.tod_multiplier:.3f} | "
            f"Ens: {mc.ens_agreement:.2f}({mc.ens_n_runs}r)",
        ]
        if mc.trade_setup:
            ts  = mc.trade_setup
            di  = "📈" if ts.direction == "BUY" else "📉"
            si  = {
                "STRONG": "🟢", "MODERATE": "🟡", "WEAK": "🔴",
            }.get(ts.signal_strength, "⬜")
            sg  = "+" if ts.direction == "BUY" else "-"
            sgs = "-" if ts.direction == "BUY" else "+"
            q_str = (
                f" {_stars(ts.quality.stars)}"
                if ts.quality else ""
            )
            L += [
                f"\n{si} *{ts.direction}* {di} "
                f"[{ts.signal_strength}]{q_str} "
                f"[{ts.signal_source}]\n",
                f"Entry:  `{_safe_md(ts.entry,'.6f')}`",
                f"Target: `{_safe_md(ts.target,'.6f')}` "
                f"({sg}{ts.target_pct:.3f}%)",
                f"Stop:   `{_safe_md(ts.invalidation,'.6f')}` "
                f"({sgs}{ts.stop_pct:.3f}%)",
                f"R/R: *1:{ts.rr_ratio:.2f}* | "
                f"P(tgt): `{ts.prob_target:.2%}`",
                f"EV: `{ts.expected_value:+.4f}%`",
            ]
        else:
            L += [
                f"\n{se_i} *{mc.signal.upper()}*",
                f"Edge:`{_safe_md(mc.edge_score,'.4f')}` "
                f"CI:`{_safe_md(mc.ci_width_pct*100,'.4f')}%`",
                "No trade setup — quality gates not met.",
            ]
        if mc.vol_regime and mc.vol_regime.signal != "neutral":
            vr = mc.vol_regime
            L.append(
                f"VolReg: `{vr.signal}` "
                f"z=`{vr.normalized_momentum:+.3f}`"
            )
        elif mc.drift_signal:
            ds    = mc.drift_signal
            sig_i = "✅" if ds.direction != "neutral" else "❌"
            L.append(
                f"Drift: `{ds.direction}` "
                f"t=`{ds.tstat:+.3f}` {sig_i}"
            )
        if mc.prob_jump is not None:
            L.append(
                f"P(spike/{mc.horizon}t): "
                f"`{mc.prob_jump:.2%}`"
            )
        L += ["", note, RISK_MSG]
        return "\n".join(line for line in L if line is not None)

    def _full_analysis(
        self, fname: str, sym: str, mc: MCResult,
        gp: Optional[GBMParams], jp: Optional[JumpParams],
        note: str, risk_pct: float, hlabel: str,
    ) -> str:
        cat   = _cat(sym)
        parts = []
        if mc.trade_setup:
            parts.append(TradeSetupBuilder.format_setup(
                mc.trade_setup, fname,
                mc.drift_signal, mc.vol_regime,
            ))
        try:
            if cat in ("vol", "step") and gp:
                if cat == "step":
                    parts.append(
                        self.narr.step_context(
                            sym, gp, mc, hlabel
                        )
                    )
                else:
                    parts.append(
                        self.narr.vol_context(
                            sym, gp, mc, hlabel
                        )
                    )
            elif jp:
                parts.append(
                    self.narr.jump_context(
                        sym, jp, mc, hlabel
                    )
                )
        except Exception:
            pass
        parts.append(
            self.narr.risk_reward_block(mc, risk_pct)
        )
        parts.append(f"\n{note}")
        parts.append(RISK_MSG)
        return "\n\n".join(p for p in parts if p)

    # ------------------------------------------------------------------
    # Alert loop
    # ------------------------------------------------------------------
    async def _alert_loop(self, bot):
        self._bot_ref = bot
        log.info(f"Alert scanner v4.5: warmup={ALERT_WARMUP}s")
        await asyncio.sleep(ALERT_WARMUP)
        await self.dm._ready.wait()
        log.info(
            f"Alert scanner v4.5 active. "
            f"Spike gate: raw conf≥{SPIKE_MIN_CONFIDENCE:.0%} "
            f"(quality shown in alert, never blocks). "
            f"Bootstrap N={N_BOOTSTRAP_RUNS} "
            f"Ens≥{ENSEMBLE_AGREEMENT_FLOOR:.2f}"
        )
        scan_count = 0
        syms       = list(ALL_SYMBOLS.items())
        while True:
            try:
                await asyncio.sleep(20)
                if not self.dm._init_done:
                    continue
                enabled = [
                    c for c in self._chats
                    if self._state(c).alerts_enabled
                ]
                if not enabled:
                    continue
                fn, ds  = syms[scan_count % len(syms)]
                scan_count += 1
                profile = effective_profile(ds)

                # Spike check — quality attached, never blocks
                if profile.spike_enabled:
                    sa = self.ae.check_spike(
                        ds, self.dm, self.jm,
                        self.se, self.ssa, self.vm,
                    )
                    if sa:
                        msg = self.se.format_standalone(sa)
                        for cid in enabled:
                            try:
                                await _safe_send_message(
                                    bot, cid, msg
                                )
                                await asyncio.sleep(0.1)
                            except Exception:
                                pass

                # Trade check
                res = self.ae.check_trade(
                    ds, self.dm, 300, "300 Ticks"
                )
                if not res:
                    continue
                _, _, mc = res

                spike_sa = None
                if profile.spike_enabled:
                    lr = self.dm.log_returns(ds)
                    if len(lr) >= MIN_TICKS:
                        jp = self.jm.fit_unbiased(ds, lr)
                        tr = self.dm.observed_tick_rate(ds)
                        spike_sa = self.se.assess(
                            ds, jp, 300, tr,
                            self.tod.multiplier(ds),
                        )
                validated, tv, _ = self.cr.resolve(
                    ds, mc, spike_sa, mc.trade_setup
                )
                if not tv or validated is None:
                    continue
                ts  = validated
                msg = TradeSetupBuilder.format_setup(
                    ts, fn, mc.drift_signal, mc.vol_regime
                )
                for cid in enabled:
                    try:
                        await _safe_send_message(bot, cid, msg)
                        await asyncio.sleep(0.1)
                    except Exception:
                        pass
            except asyncio.CancelledError:
                log.info("Alert scanner v4.5 stopped.")
                break
            except Exception as e:
                log.warning(f"Alert loop: {e}")
                await asyncio.sleep(15)

    # ------------------------------------------------------------------
    async def _post_init(self, app: Application):
        await app.bot.set_my_commands([
            BotCommand("start",      "Main menu"),
            BotCommand("spike",      "Spike check — /spike Boom1000"),
            BotCommand("drift",      "Drift/Regime — /drift V75"),
            BotCommand("tod",        "Time-of-Day — /tod V75"),
            BotCommand("prob",       "Prob matrix — /prob V100"),
            BotCommand("parameters", "Params — /parameters V75"),
            BotCommand("results",    "Results — /results V75"),
            BotCommand("watch",      "Watch — /watch V75"),
            BotCommand("unwatch",    "Unwatch"),
            BotCommand("status",     "Data status"),
            BotCommand("v75",        "Quick V75"),
            BotCommand("v100",       "Quick V100"),
            BotCommand("boom1000",   "Quick Boom1000"),
            BotCommand("crash500",   "Quick Crash500"),
            BotCommand("jump50",     "Quick Jump50"),
            BotCommand("step",       "Quick Step"),
        ])
        # Pass ssa so backfill initialises drought tracking
        self._ws_task = asyncio.ensure_future(
            self.dm.run(ssa=self.ssa)
        )
        self._al_task = asyncio.ensure_future(
            self._alert_loop(app.bot)
        )
        log.info("SQE v4.5 background tasks launched OK")

    async def _post_stop(self, app: Application):
        log.info("Stopping SQE v4.5 ...")
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
        log.info("SQE v4.5 shutdown — state saved OK")

    async def _track_msg(self, update: Update,
                         ctx: ContextTypes.DEFAULT_TYPE):
        self._track(update)

    def run(self):
        if not TELEGRAM_TOKEN:
            log.error("Set TELEGRAM_TOKEN environment variable!")
            raise SystemExit(1)
        app = (
            Application.builder()
            .token(TELEGRAM_TOKEN)
            .post_init(self._post_init)
            .post_stop(self._post_stop)
            .build()
        )
        for cmd in [
            "v10","v25","v50","v75","v100","v250",
            "boom300","boom500","boom600","boom900","boom1000",
            "crash300","crash500","crash600","crash900","crash1000",
            "jump10","jump25","jump50","jump75","jump100",
            "step","stepindex",
        ]:
            app.add_handler(
                CommandHandler(cmd, self.cmd_quick_sym)
            )
        app.add_handler(CommandHandler("start",      self.cmd_start))
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
            filters.TEXT & ~filters.COMMAND, self._track_msg
        ))
        log.info("SQE v4.5 — Starting polling ...")
        app.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True,
        )

# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    log.info("=" * 66)
    log.info("  SYNTHETIC QUANT ELITE v4.5 — PRODUCTION")
    log.info("=" * 66)
    log.info("  v4.5 Philosophy:")
    log.info("  Spike alerts: ALWAYS fire if raw_conf >= 0.95")
    log.info("  Quality stars: ALWAYS shown in message")
    log.info("  Stars NEVER block — they INFORM the user")
    log.info("  Mirrors VolSignalQuality pattern exactly")
    log.info("=" * 66)
    log.info("  SpikeSuppressionAnalyser (5 factors, informational):")
    log.info(
        f"  1. Drought: 2x={DROUGHT_DISCOUNT_2X:.0%} "
        f"3x={DROUGHT_DISCOUNT_3X:.0%} "
        f"5x={DROUGHT_DISCOUNT_5X:.0%} display discount"
    )
    log.info(
        f"  2. Vol compression: "
        f"mild={VOL_COMPRESSION_DISCOUNT_MILD:.0%} "
        f"strong={VOL_COMPRESSION_DISCOUNT_STRONG:.0%}"
    )
    log.info(
        f"  3. Activity ratio: "
        f"quiet={ACTIVITY_DISCOUNT_QUIET:.0%} "
        f"very_quiet={ACTIVITY_DISCOUNT_VERY_QUIET:.0%}"
    )
    log.info(
        f"  4. Post-spike cooling: "
        f"{POST_SPIKE_COOLING_FRACTION:.0%} of interval "
        f"(soft warning, not zero)"
    )
    log.info(
        f"  5. Range compression: "
        f"ratio<{RANGE_COMPRESSION_RATIO:.2f} → warning"
    )
    log.info("  Drought initialised from backfill on startup")
    log.info("=" * 66)
    log.info("  VolSignalQuality (5 factors, informational):")
    log.info("  1. Regime consistency (3 windows)")
    log.info(
        f"  2. Momentum persistence "
        f"(≥{PERSISTENCE_MIN_AGREEMENT:.0%} of snapshots)"
    )
    log.info(
        f"  3. Drift/noise ratio (min {DRIFT_NOISE_MIN_RATIO:.2f})"
    )
    log.info("  4. Regime support (RANGING = caution)")
    log.info("  5. Cross-TF agreement (3 windows)")
    log.info("=" * 66)
    log.info(
        f"  Global gates: "
        f"P≥{MIN_PROB_TARGET:.0%} EV>0 "
        f"RR≥{MIN_RR_GLOBAL:.1f} "
        f"Ens≥{ENSEMBLE_AGREEMENT_FLOOR:.2f} "
        f"Bootstrap N={N_BOOTSTRAP_RUNS}"
    )
    log.info(
        f"  Spike gate: raw_conf≥{SPIKE_MIN_CONFIDENCE:.0%} ONLY"
    )
    log.info("=" * 66)
    if not TELEGRAM_TOKEN:
        log.error("Set TELEGRAM_TOKEN environment variable!")
        raise SystemExit(1)
    log.info(f"Telegram: ...{TELEGRAM_TOKEN[-8:]}")
    log.info(f"Deriv:    app_id={DERIV_APP_ID}")
    log.info(f"Persist:  {PERSIST_FILE}")
    log.info(f"Patterns: {PATTERN_FILE}")
    log.info("=" * 66)
    try:
        bot = SQEBotV45()
        bot.run()
    except KeyboardInterrupt:
        log.info("Interrupted — shutting down.")
    except Exception as e:
        log.critical(f"Fatal: {e}", exc_info=True)
        raise SystemExit(1)
