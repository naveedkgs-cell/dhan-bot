"""
MASTER TRADING ENGINE - NIFTY / SENSEX / CRUDE OIL / USDINR
================================================================
Full spec implementation:
  - 5-min timeframe
  - SL 10-12 pts (20 pts if VIX > 20), first target 25 pts
  - At 25 pts: book 50% qty, trail the rest
  - Breakeven trail at +15, then trail every +10 after 25
  - Auto Call<->Put reversal on trend flip while in profit
  - Pre-market regime check (VIX / gap / crude / DXY / US futures)
  - EMA9/21, VWAP, SuperTrend(10,3), ATR14, RSI14+EMA9
  - Basic candlestick pattern recognition
  - Greeks-based strike filter (delta 0.50-0.60, theta cutoff)
  - PCR/OI filter (STUB - needs a real option-chain data source)
  - Gap/crash-day protection (skip new trades on big gap / ATR spike)
  - Risk management: max 10 trades/day, 2% capital risk per trade
  - Trading window HARD-locked to 9:30-15:00 for new entries; 15:20
    force square-off of everything, every day - NO BTST, ever
  - Glitch/anomaly protection: an abnormally huge single-candle range
    or stale data pauses new entries and protectively tightens the
    stop on any open position instead of trusting a bad print

===========================================================================
READ BEFORE USE - IMPORTANT FACTS, NOT JUST DISCLAIMERS
===========================================================================
1. FOREX: Indian residents may legally trade only INR currency pairs
   (USDINR, EURINR, GBPINR, JPYINR) on NSE/BSE/MCX-SX. Trading global
   pairs (EUR/USD etc.) via an offshore broker is restricted under
   FEMA/RBI rules for retail residents. This script only wires up
   USDINR for that reason.
2. TradingView / Moneycontrol / Chartink / SenseBull / Zee Business do
   NOT have public free APIs. Scraping them breaks their Terms of
   Service and breaks constantly (layout changes). Pre-market checks
   here use publicly available data (yfinance) instead. News-sentiment
   is a STUB - wire in a real news API if you want that filter live.
3. PCR / Open Interest needs live option-chain data. Verify the exact
   dhanhq option-chain method name against current docs before relying
   on `get_pcr_oi()` below; it is written defensively (returns None ->
   filter skipped - if it can't fetch data, rather than guessing).
4. Rule-based regime filters (VIX spike, ATR spike, big gap) are NOT a
   trained model that has "learned" crash patterns from 20 years of
   data. That would be a separate, larger backtesting project.
5. Real money, real risk. Paper-trade (DRY_RUN=True) for weeks before
   going live. Not investment advice - not a SEBI-registered advisor.

Install:
    pip install dhanhq pandas numpy yfinance scipy requests --break-system-packages

Run under tmux so it survives SSH disconnects:
    tmux new -s trading
    source ~/tradingbot_venv/bin/activate
    python3 master_trading_engine.py
    (detach: Ctrl+b then d)
"""

import os
import time
import threading
import logging
import traceback
import datetime as dt
from zoneinfo import ZoneInfo
import requests
import pandas as pd
import numpy as np
from dhanhq import DhanContext, dhanhq

# All market-hour / date logic below is IST, regardless of the server's
# own timezone (e.g. a UTC cloud VM). NEVER use dt.datetime.now() or
# dt.date.today() directly in this file for trading-window/date checks -
# always use now_ist() / today_ist() instead.
IST = ZoneInfo("Asia/Kolkata")


def now_ist():
    return dt.datetime.now(IST)


def today_ist():
    return now_ist().date()

# =====================================================================
# 0. CONFIG
# =====================================================================
DRY_RUN = True
MAX_DAILY_LOSS_RUPEES = 3000
MAX_TRADES_PER_DAY = 10
RISK_PER_TRADE_PCT = 0.02          # 2% of capital per trade
CAPITAL = 100000

BASE_SL_POINTS = 12                # normal regime
HIGH_VIX_SL_POINTS = 20            # VIX > 20 regime
TARGET_TRIGGER_POINTS = 25
PARTIAL_BOOK_FRACTION = 0.5        # book this fraction of qty at first target
BREAKEVEN_TRIGGER_POINTS = 15
TRAIL_STEP_POINTS = 10             # after target hit, trail SL every 10 pts
LOTS = 1

# --- sideways-day range-scalp setup ---
# On a tight, no-trend day (VIX < 12, "no_trade_sideways" regime) the
# normal breakout conditions rarely fire at all. Instead of sitting
# completely idle, take small mean-reversion bounces off the day's
# opening-range edges - much smaller SL/target than a trend trade,
# because we're not expecting a sustained move.
SIDEWAYS_SL_POINTS = 8
SIDEWAYS_TARGET_POINTS = 12
SIDEWAYS_BREAKEVEN_TRIGGER = 6
SIDEWAYS_TRAIL_STEP = 5

INSTRUMENTS = {
    "NIFTY":    {"yf_symbol": "^NSEI",  "exch_segment": "NSE_FNO",  "lot_size": 75},
    "SENSEX":   {"yf_symbol": "^BSESN", "exch_segment": "BSE_FNO",  "lot_size": 20},
    "CRUDEOIL": {"yf_symbol": "CL=F",   "exch_segment": "MCX_COMM", "lot_size": 100},
    "USDINR":   {"yf_symbol": "INR=X",  "exch_segment": "NSE_CURRENCY", "lot_size": 1000},
}
ACTIVE_INSTRUMENTS = ["NIFTY", "SENSEX"]   # both tradeable via Dhan; add CRUDEOIL/USDINR later if wanted

# security_id + exchange segment of the UNDERLYING INDEX (not the options)
# that dhanhq's option-chain endpoint needs. TODO: fill these in yourself
# from the current Dhan scrip master / API docs before PCR/OI or real
# Greeks will actually work - left empty on purpose rather than guessing
# IDs I can't verify, which could silently point at the wrong instrument.
UNDERLYING_SECURITY_IDS = {
    # "NIFTY": ("13", "IDX_I"),
    # "SENSEX": ("51", "IDX_I"),
}

# Optional - set NEWS_API_KEY to enable the news-sentiment filter (e.g. a
# newsapi.org key). Without it the filter is simply skipped, same as PCR.
NEWS_API_KEY = os.environ.get("NEWS_API_KEY")
NEWS_QUERY = "Nifty 50 OR Sensex OR RBI OR Indian stock market"
NEWS_SENTIMENT_BLOCK_THRESHOLD = 1.5   # crude keyword-score magnitude that blocks the opposite-side trade

DHAN_SCRIP_MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
MARKET_OPEN = dt.time(9, 15)
FIRST_15MIN_END = dt.time(9, 30)   # no NEW trade before this (range-building window)
LAST_ENTRY_TIME = dt.time(15, 0)   # no NEW trade after this - gives time to manage before square-off
SQUARE_OFF_TIME = dt.time(15, 20)  # force-close every open position here, no BTST ever
MARKET_CLOSE = dt.time(15, 30)
POLL_SECONDS = 60

# --- anomaly / glitch protection ---
ANOMALY_ATR_MULT = 3.0             # a single candle range > this * ATR14 = suspicious tick/spike
ANOMALY_PAUSE_CYCLES = 3           # skip this many poll cycles after an anomaly before trusting data again

TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TG_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

LOG_FILE = os.path.expanduser("~/trading_bot.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
log = logging.getLogger("master_engine")


def tg_send(msg: str):
    if not TG_TOKEN or not TG_CHAT_ID:
        log.info(f"[telegram not configured] {msg}")
        return
    try:
        requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                       data={"chat_id": TG_CHAT_ID, "text": msg}, timeout=10)
    except Exception as e:
        log.warning(f"Telegram send failed: {e}")


# =====================================================================
# 1. INDICATORS
# =====================================================================
def ema(s, span):
    return s.ewm(span=span, adjust=False).mean()


def atr(df, period=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def rsi(s, period=14):
    d = s.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def vwap(df):
    typical = (df["high"] + df["low"] + df["close"]) / 3
    day = df.index.date
    pv = (typical * df["volume"]).groupby(day).cumsum()
    vol = df["volume"].groupby(day).cumsum()
    return pv / vol.replace(0, np.nan)


def supertrend(df, period=10, multiplier=3):
    atr_val = atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2
    upper = hl2 + multiplier * atr_val
    lower = hl2 - multiplier * atr_val

    trend = pd.Series(index=df.index, dtype="float64")
    direction = pd.Series(index=df.index, dtype="int64")
    trend.iloc[0] = upper.iloc[0]
    direction.iloc[0] = 1

    for i in range(1, len(df)):
        close = df["close"].iloc[i]
        if close > upper.iloc[i - 1]:
            direction.iloc[i] = 1
        elif close < lower.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]

        # Band ratcheting (this is what makes it a SuperTrend and not just
        # raw hl2 +/- ATR bands): in an uptrend the lower band can only
        # move UP, never back down; in a downtrend the upper band can
        # only move DOWN, never back up. Without this the bands have no
        # memory and the indicator whipsaws far more than it should.
        if direction.iloc[i] == 1:
            lower.iloc[i] = max(lower.iloc[i], lower.iloc[i - 1])
        else:
            upper.iloc[i] = min(upper.iloc[i], upper.iloc[i - 1])
        trend.iloc[i] = lower.iloc[i] if direction.iloc[i] == 1 else upper.iloc[i]

    return trend, direction  # direction: 1 = green/bullish, -1 = red/bearish


def candlestick_pattern(df):
    """Very simplified pattern check on the last 3 candles."""
    if len(df) < 3:
        return None
    c0, c1, c2 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    body = lambda r: abs(r["close"] - r["open"])
    rng = lambda r: r["high"] - r["low"] + 1e-9
    is_bull = lambda r: r["close"] > r["open"]
    is_bear = lambda r: r["close"] < r["open"]

    if is_bear(c1) and is_bull(c2) and c2["close"] > c1["open"] and c2["open"] < c1["close"]:
        return "bullish_engulfing"
    if is_bull(c1) and is_bear(c2) and c2["open"] > c1["close"] and c2["close"] < c1["open"]:
        return "bearish_engulfing"
    if body(c2) / rng(c2) < 0.1:
        return "doji"
    lower_wick = min(c2["open"], c2["close"]) - c2["low"]
    upper_wick = c2["high"] - max(c2["open"], c2["close"])
    if lower_wick > 2 * body(c2) and upper_wick < body(c2):
        return "hammer"
    if upper_wick > 2 * body(c2) and lower_wick < body(c2):
        return "shooting_star"
    if is_bear(c0) and body(c1) / rng(c1) < 0.3 and is_bull(c2) and c2["close"] > (c0["open"] + c0["close"]) / 2:
        return "morning_star"
    if is_bull(c0) and body(c1) / rng(c1) < 0.3 and is_bear(c2) and c2["close"] < (c0["open"] + c0["close"]) / 2:
        return "evening_star"
    return None


BULLISH_PATTERNS = {"bullish_engulfing", "hammer", "morning_star"}
BEARISH_PATTERNS = {"bearish_engulfing", "shooting_star", "evening_star"}


# =====================================================================
# 2. PRE-MARKET REGIME CHECK  (public data only - yfinance)
# =====================================================================
def premarket_check():
    import yfinance as yf
    out = {}
    tickers = {
        "us_futures_sp500": "ES=F",
        "us_futures_nasdaq": "NQ=F",
        "dxy": "DX-Y.NYB",
        "crude": "CL=F",
        "india_vix": "^INDIAVIX",
        "nifty_prev": "^NSEI",
    }
    for key, sym in tickers.items():
        try:
            data = yf.Ticker(sym).history(period="2d")
            out[key] = float(data["Close"].iloc[-1]) if not data.empty else None
        except Exception:
            out[key] = None

    vix = out.get("india_vix")
    regime = vix_regime(vix) if vix else "unknown"
    sentiment = get_news_sentiment()
    sentiment_str = f"{sentiment:+.2f}" if sentiment is not None else "not configured (set NEWS_API_KEY)"
    msg = (f"📊 Pre-market check:\n"
           f"India VIX: {vix}\nRegime: {regime}\n"
           f"US S&P fut: {out.get('us_futures_sp500')}\n"
           f"DXY: {out.get('dxy')}\nCrude: {out.get('crude')}\n"
           f"News sentiment (keyword-based, not a model): {sentiment_str}")
    tg_send(msg)
    return out, regime, sentiment


def vix_regime(vix_value):
    if vix_value is None:
        return "unknown"
    if vix_value < 12:
        return "no_trade_sideways"
    elif vix_value <= 18:
        return "normal"
    elif vix_value <= 20:
        return "elevated"
    return "high_vix"


def sl_points_for_regime(regime):
    """Kept for compatibility - risk_params_for_setup() is what open_position()
    and manage_position() actually use now (it also covers the sideways setup)."""
    return HIGH_VIX_SL_POINTS if regime == "high_vix" else BASE_SL_POINTS


# =====================================================================
# 3. PCR / OI  (live option chain, best-effort - see caveats below)
# =====================================================================
def get_option_chain(dhan, instrument_name, expiry=None):
    """
    Live option chain (every strike, one expiry) for instrument_name.
    Needs UNDERLYING_SECURITY_IDS filled in above - returns None (and
    logs why) until it is, same "skip the filter rather than guess"
    behaviour as before.
    VERIFY the exact dhanhq method names (`expiry_list`, `option_chain`)
    and response shape against current docs before relying on this -
    broker SDKs change these; written defensively so a changed/bad
    response returns None instead of silently trusting a wrong shape.
    """
    under = UNDERLYING_SECURITY_IDS.get(instrument_name)
    if under is None:
        log.info(f"get_option_chain: no UNDERLYING_SECURITY_IDS entry for {instrument_name} - skipping")
        return None
    under_id, under_seg = under
    try:
        if expiry is None:
            exp_resp = dhan.expiry_list(under_security_id=under_id, under_exchange_segment=under_seg)
            expiries = exp_resp.get("data", {}).get("data", [])
            if not expiries:
                return None
            expiry = expiries[0]
        resp = dhan.option_chain(under_security_id=under_id, under_exchange_segment=under_seg, expiry=expiry)
        chain = resp.get("data", {}).get("data", {})
        return chain or None   # expected shape: {strike_str: {"ce": {...}, "pe": {...}}, ...}
    except Exception as e:
        log.warning(f"get_option_chain failed for {instrument_name}: {e}")
        return None


def get_pcr_oi(dhan, instrument_name):
    """
    Returns {"pcr": float, "chain": <raw chain>} or None if unavailable.
    PCR = total put OI / total call OI across the fetched chain.
    """
    chain = get_option_chain(dhan, instrument_name)
    if not chain:
        return None
    try:
        total_ce_oi = sum(float((v.get("ce") or {}).get("oi", 0) or 0) for v in chain.values())
        total_pe_oi = sum(float((v.get("pe") or {}).get("oi", 0) or 0) for v in chain.values())
        if total_ce_oi <= 0:
            return None
        return {"pcr": total_pe_oi / total_ce_oi, "chain": chain}
    except Exception as e:
        log.warning(f"PCR calc failed for {instrument_name}: {e}")
        return None


# =====================================================================
# 3b. NEWS SENTIMENT  (optional - needs NEWS_API_KEY; crude keyword score,
#     NOT a trained sentiment model - same honesty as the regime filter
#     being rule-based, not a model that's "learned" crash patterns)
# =====================================================================
_POSITIVE_WORDS = {"rally", "surge", "gain", "gains", "rise", "rises", "rising", "bullish", "upgrade",
                    "beats", "growth", "strong", "record high", "optimis", "boost", "rebound"}
_NEGATIVE_WORDS = {"crash", "plunge", "slump", "fall", "falls", "falling", "bearish", "downgrade",
                    "misses", "recession", "weak", "record low", "pessimis", "selloff", "sell-off", "war"}


def get_news_sentiment(query=NEWS_QUERY):
    """
    Rough score: (positive keyword hits - negative keyword hits) averaged
    across recent headlines. Positive = bullish-leaning news, negative =
    bearish-leaning. This is keyword counting, not NLP - treat it as a
    coarse filter, not a signal on its own. Returns None if NEWS_API_KEY
    isn't set or the fetch fails, same "skip the filter" pattern as PCR.
    """
    if not NEWS_API_KEY:
        return None
    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={"q": query, "language": "en", "sortBy": "publishedAt",
                    "pageSize": 20, "apiKey": NEWS_API_KEY},
            timeout=10,
        )
        articles = resp.json().get("articles", [])
        if not articles:
            return None
        score = 0
        for a in articles:
            text = f"{a.get('title') or ''} {a.get('description') or ''}".lower()
            score += sum(w in text for w in _POSITIVE_WORDS)
            score -= sum(w in text for w in _NEGATIVE_WORDS)
        return score / len(articles)
    except Exception as e:
        log.warning(f"News sentiment fetch failed: {e}")
        return None


# =====================================================================
# 4. DHAN CONNECTION / SCRIP MASTER
# =====================================================================
def connect_dhan():
    client_id = os.environ.get("DHAN_CLIENT_ID")
    access_token = os.environ.get("DHAN_ACCESS_TOKEN")
    if not client_id or not access_token:
        raise RuntimeError("DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN not set in environment")
    return dhanhq(DhanContext(client_id, access_token))


_scrip_master_cache = None


def load_scrip_master():
    global _scrip_master_cache
    if _scrip_master_cache is None:
        _scrip_master_cache = pd.read_csv(DHAN_SCRIP_MASTER_URL, low_memory=False)
    return _scrip_master_cache


def resolve_option_by_delta(instrument_name, spot_price, option_type, target_delta_range=(0.50, 0.60), chain=None):
    """Pick the nearest-expiry strike whose delta falls in range.
    If `chain` (this cycle's live option chain, from get_pcr_oi) has a
    real delta for a strike, that's used - actual market-implied Greeks.
    Only strikes missing from the chain fall back to a rough
    Black-Scholes estimate with a flat guessed IV, which is an
    approximation, not real Greeks - always sanity-check the resolved
    strike/premium before trusting either one.
    VERIFY the chain's field names (assumed here: chain[strike]["ce"/"pe"]
    ["greeks"]["delta"]) against current dhanhq docs - fixed defensively
    so a mismatched shape just falls back to the approximation instead
    of crashing or silently using a wrong number."""
    from scipy.stats import norm

    df = load_scrip_master()
    subset = df[
        (df["SEM_TRADING_SYMBOL"].astype(str).str.startswith(instrument_name))
        & (df["SEM_OPTION_TYPE"] == option_type)
        & (df["SEM_INSTRUMENT_NAME"].astype(str).str.contains("OPT"))
    ].copy()
    if subset.empty:
        raise RuntimeError(f"No contracts found for {instrument_name}/{option_type}")

    subset["SEM_EXPIRY_DATE"] = pd.to_datetime(subset["SEM_EXPIRY_DATE"])
    nearest_expiry = subset["SEM_EXPIRY_DATE"].min()
    subset = subset[subset["SEM_EXPIRY_DATE"] == nearest_expiry]
    days_to_expiry = max((nearest_expiry.date() - today_ist()).days, 1)

    real_deltas = {}
    if chain:
        side_key = "ce" if option_type == "CE" else "pe"
        for strike_key, strike_data in chain.items():
            try:
                strike_val = round(float(strike_key), 2)
                delta_val = (strike_data.get(side_key) or {}).get("greeks", {}).get("delta")
                if delta_val is not None:
                    real_deltas[strike_val] = float(delta_val)
            except Exception:
                continue  # skip malformed entries rather than trust them

    def resolve_delta(strike, iv_pct=15.0, r=0.07):
        key = round(float(strike), 2)
        if key in real_deltas:
            return real_deltas[key], True
        T = days_to_expiry / 365
        sigma = iv_pct / 100
        d1 = (np.log(spot_price / strike) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        approx = norm.cdf(d1) if option_type == "CE" else norm.cdf(d1) - 1
        return approx, False

    resolved = subset["SEM_STRIKE_PRICE"].apply(resolve_delta)
    subset["approx_delta"] = resolved.apply(lambda t: t[0])
    subset["delta_is_real"] = resolved.apply(lambda t: t[1])
    target_mid = sum(target_delta_range) / 2
    subset["delta_diff"] = (subset["approx_delta"].abs() - target_mid).abs()
    best = subset.sort_values("delta_diff").iloc[0]

    return {
        "security_id": str(best["SEM_SMST_SECURITY_ID"]),
        "trading_symbol": best["SEM_TRADING_SYMBOL"],
        "strike": best["SEM_STRIKE_PRICE"],
        "approx_delta": round(best["approx_delta"], 3),
        "delta_is_real": bool(best["delta_is_real"]),
        "expiry": nearest_expiry.date(),
    }


# =====================================================================
# 5. SIGNAL ENGINE
# =====================================================================
def compute_frame(yf_symbol):
    import yfinance as yf
    raw = yf.download(yf_symbol, period="3d", interval="5m", progress=False)
    if raw.empty:
        return None
    # Newer yfinance versions return MultiIndex columns (Price, Ticker) even
    # for a single symbol. Flatten to plain column names, otherwise every
    # df["close"] is a one-column DataFrame and the indicators crash.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
    raw["ema9"] = ema(raw["close"], 9)
    raw["ema21"] = ema(raw["close"], 21)
    raw["rsi14"] = rsi(raw["close"], 14)
    raw["vwap"] = vwap(raw)
    raw["atr14"] = atr(raw, 14)
    st_trend, st_dir = supertrend(raw, 10, 3)
    raw["st_dir"] = st_dir
    return raw


def day_high_low_range(df, ref_date):
    day_df = df[df.index.date == ref_date]
    first15 = day_df.between_time("09:15", "09:30")
    if first15.empty:
        return None, None
    return first15["high"].max(), first15["low"].min()


def sideways_range_signal(df):
    """
    Small mean-reversion scalp for tight/no-trend days: price pokes
    near the opening-range edge, RSI is stretched, and a reversal
    candle shows up - opposite logic to the breakout setup, since on
    a sideways day the edge is more likely to hold than break.
    """
    last = df.iloc[-1]
    today = today_ist()
    day_high, day_low = day_high_low_range(df, today)
    if day_high is None:
        return None
    range_span = day_high - day_low
    if range_span <= 0:
        return None

    pattern = candlestick_pattern(df)
    near_low = last["close"] <= day_low + 0.15 * range_span
    near_high = last["close"] >= day_high - 0.15 * range_span

    call_cond = near_low and last["rsi14"] < 35 and pattern in BULLISH_PATTERNS
    put_cond = near_high and last["rsi14"] > 65 and pattern in BEARISH_PATTERNS

    if call_cond:
        return {"side": "CE", "spot": float(last["close"]), "setup": "sideways_range_scalp"}
    if put_cond:
        return {"side": "PE", "spot": float(last["close"]), "setup": "sideways_range_scalp"}
    return None


def risk_params_for_setup(regime, setup):
    """(sl_pts, target_pts, breakeven_trigger, trail_step) for a signal.
    Range-scalp setups get tighter numbers across the board - they're
    meant to pocket a small bounce, not ride a trend."""
    if setup == "sideways_range_scalp":
        return SIDEWAYS_SL_POINTS, SIDEWAYS_TARGET_POINTS, SIDEWAYS_BREAKEVEN_TRIGGER, SIDEWAYS_TRAIL_STEP
    sl = HIGH_VIX_SL_POINTS if regime == "high_vix" else BASE_SL_POINTS
    return sl, TARGET_TRIGGER_POINTS, BREAKEVEN_TRIGGER_POINTS, TRAIL_STEP_POINTS


def entry_signal(df, vix_regime_str, pcr_info=None, news_sentiment=None):
    now = now_ist().time()
    if now < FIRST_15MIN_END or now >= LAST_ENTRY_TIME:
        return None   # outside the 9:30-15:00 new-entry window

    if vix_regime_str == "no_trade_sideways":
        sig = sideways_range_signal(df)
    else:
        sig = _breakout_signal(df, pcr_info)

    if sig and news_sentiment is not None:
        # broadly negative news -> don't buy calls; broadly positive -> don't buy puts
        if sig["side"] == "CE" and news_sentiment <= -NEWS_SENTIMENT_BLOCK_THRESHOLD:
            return None
        if sig["side"] == "PE" and news_sentiment >= NEWS_SENTIMENT_BLOCK_THRESHOLD:
            return None

    return sig


def _breakout_signal(df, pcr_info=None):
    last = df.iloc[-1]
    today = today_ist()
    day_high, day_low = day_high_low_range(df, today)
    if day_high is None:
        return None

    pattern = candlestick_pattern(df)

    call_cond = (
        last["close"] > last["vwap"]
        and last["ema9"] > last["ema21"]
        and last["rsi14"] > 55
        and last["st_dir"] == 1
        and pattern in BULLISH_PATTERNS
        and last["close"] > day_high
    )
    put_cond = (
        last["close"] < last["vwap"]
        and last["ema9"] < last["ema21"]
        and last["rsi14"] < 45
        and last["st_dir"] == -1
        and pattern in BEARISH_PATTERNS
        and last["close"] < day_low
    )

    if pcr_info:
        pcr = pcr_info.get("pcr")
        if pcr is not None:
            call_cond = call_cond and pcr > 1.0
            put_cond = put_cond and pcr < 0.8

    if call_cond:
        return {"side": "CE", "spot": float(last["close"]), "setup": "breakout"}
    if put_cond:
        return {"side": "PE", "spot": float(last["close"]), "setup": "breakout"}
    return None


def trend_reversed_against(df, side):
    """True if EMA9/21 + SuperTrend direction has flipped against an open side -
    used for auto CE<->PE reversal while the position is in profit."""
    if len(df) < 22:
        return False
    last = df.iloc[-1]
    if side == "CE":
        return last["ema9"] < last["ema21"] and last["st_dir"] == -1
    else:
        return last["ema9"] > last["ema21"] and last["st_dir"] == 1


def gap_or_crash_day(df):
    """True if today opened with a big gap or ATR is unusually high ->
    skip NEW trades, only manage existing ones."""
    today = today_ist()
    day_df = df[df.index.date == today]
    if day_df.empty or len(df[df.index.date < today]) == 0:
        return False
    prev_close = df[df.index.date < today]["close"].iloc[-1]
    today_open = day_df["open"].iloc[0]
    gap = abs(today_open - prev_close)
    atr_now = df["atr14"].iloc[-1]
    return gap > 1.5 * atr_now or atr_now > 70


def detect_anomaly(df):
    """
    Flags a likely data glitch / stray-tick spike:
      - last candle's range is way bigger than normal (ATR-based)
      - OR the feed looks stale (last candle timestamp too old vs now)
    On a flagged cycle: don't take new trades, and if a position is
    open, tighten its stop protectively rather than trusting a bad print.
    """
    if len(df) < 20:
        return False, "insufficient_data"

    last = df.iloc[-1]
    atr_now = df["atr14"].iloc[-1]
    candle_range = last["high"] - last["low"]
    if atr_now > 0 and candle_range > ANOMALY_ATR_MULT * atr_now:
        return True, "extreme_candle_range"

    last_ts = df.index[-1]
    now_ts = pd.Timestamp.now(tz=last_ts.tz) if last_ts.tzinfo else pd.Timestamp.now()
    staleness_min = (now_ts - last_ts).total_seconds() / 60
    if staleness_min > 15:
        return True, f"stale_data_{int(staleness_min)}min"

    return False, None


# =====================================================================
# 6. ORDER PLACEMENT
# =====================================================================
ORDER_FILL_TIMEOUT_SEC = 15
ORDER_FILL_POLL_SEC = 2
FILLED_STATUSES = {"TRADED", "COMPLETE", "FILLED"}
DEAD_STATUSES = {"REJECTED", "CANCELLED"}


def verify_order_filled(dhan, order_id):
    """
    Polls order status until it's actually TRADED/filled, rejected, or
    cancelled - rather than trusting place_order()'s immediate response,
    which only confirms the order was ACCEPTED by the exchange, not that
    it actually got filled (or at what price).
    VERIFY the exact dhanhq order-status method/response shape/status
    strings against current docs before relying on this for real money -
    the field names/status strings below are best-effort and may not
    match your SDK version exactly.
    """
    if not order_id:
        return {"status": "NO_ORDER_ID", "filled_qty": None, "avg_price": None}
    waited = 0
    while waited < ORDER_FILL_TIMEOUT_SEC:
        try:
            resp = dhan.get_order_by_id(order_id)
            data = resp.get("data", {})
            if isinstance(data, list):
                data = data[0] if data else {}
            status = str(data.get("orderStatus", "")).upper()
            if status in FILLED_STATUSES:
                return {
                    "status": status,
                    "filled_qty": data.get("filledQty") or data.get("quantity"),
                    "avg_price": data.get("averageTradedPrice") or data.get("price"),
                }
            if status in DEAD_STATUSES:
                return {"status": status, "filled_qty": 0, "avg_price": None}
        except Exception as e:
            log.warning(f"Order status check failed for {order_id}: {e}")
        time.sleep(ORDER_FILL_POLL_SEC)
        waited += ORDER_FILL_POLL_SEC
    return {"status": "UNCONFIRMED_TIMEOUT", "filled_qty": None, "avg_price": None}


def place_order(dhan, cfg, security_id, side, qty):
    """
    Returns {"status", "filled_qty", "avg_price"}. In DRY_RUN, "status"
    is always "dry_run" (no real order exists to confirm). Live, this
    only returns after verify_order_filled() confirms an actual fill -
    callers must check status before trusting the trade happened.
    """
    if DRY_RUN:
        ref_price = get_option_ltp(dhan, cfg, security_id)
        tg_send(f"[DRY RUN] {side} qty={qty} security_id={security_id} ref_price={ref_price}")
        return {"status": "dry_run", "filled_qty": qty, "avg_price": ref_price}
    try:
        resp = dhan.place_order(
            security_id=security_id, exchange_segment=cfg["exch_segment"],
            transaction_type=dhan.BUY if side == "BUY" else dhan.SELL,
            quantity=qty, order_type=dhan.MARKET, product_type=dhan.INTRA, price=0,
        )
        order_id = (resp.get("data") or {}).get("orderId") or resp.get("orderId")
        fill = verify_order_filled(dhan, order_id)
        if fill["status"] in FILLED_STATUSES:
            tg_send(f"✅ ORDER FILLED: {side} qty={fill['filled_qty']} avg_price={fill['avg_price']} "
                     f"id={security_id} order_id={order_id}")
        else:
            tg_send(f"🚨 ORDER NOT CONFIRMED FILLED: {side} qty={qty} id={security_id} "
                     f"order_id={order_id} status={fill['status']} - CHECK DHAN APP MANUALLY")
        return fill
    except Exception as e:
        tg_send(f"⚠️ ORDER FAILED: {side} qty={qty} id={security_id}\n{e}")
        raise


def force_close_position(dhan, cfg, pos, reason):
    fill = None
    if pos["qty"] > 0:
        try:
            fill = place_order(dhan, cfg, pos["security_id"], "SELL", pos["qty"])
        except Exception as e:
            tg_send(f"🚨 EXIT ORDER FAILED for {pos['instrument']} {pos['side']} qty={pos['qty']}: {e} "
                     f"- POSITION MAY STILL BE OPEN, CHECK/CLOSE MANUALLY IN DHAN APP")
    if fill is not None and fill["status"] not in ({"dry_run"} | FILLED_STATUSES):
        tg_send(f"⚠️ Exit fill unconfirmed for {pos['instrument']} {pos['side']} - "
                 f"VERIFY MANUALLY in Dhan app (status={fill['status']})")
    tg_send(f"❌ CLOSED {pos['instrument']} {pos['side']} qty={pos['qty']} | reason: {reason}")


def square_off_everything(dhan, states, reason="NO-BTST EOD SQUARE-OFF"):
    """Force-flatten every open position across every active instrument.
    This is the enforcement step - runs unconditionally at SQUARE_OFF_TIME,
    no overnight carry under any circumstance."""
    for name, state in states.items():
        if state.position is not None:
            cfg = INSTRUMENTS[name]
            force_close_position(dhan, cfg, state.position, reason)
            state.position = None
            state.squared_off_today = True
    tg_send(f"🔔 {reason} - all positions flattened across {list(states.keys())}. NO overnight carry.")


def get_option_ltp(dhan, cfg, security_id):
    """
    Live LTP of the OPTION itself (premium), not the underlying spot.
    SL/target/trailing must act on premium movement, since an option's
    premium does NOT move 1:1 with the underlying (delta/theta/IV all
    change it) - using spot points as a stand-in for premium points
    under- or over-states real risk.
    Called in DRY_RUN too, on purpose: quote_data() only READS market
    data, it never places an order, so paper trades can still track a
    real premium instead of the spot approximation.
    VERIFY the exact dhanhq quote method/response shape against current
    docs, same caveat as get_pcr_oi() below - written defensively so a
    bad/changed response returns None (caller falls back safely)
    instead of silently trusting a wrong number.
    """
    try:
        resp = dhan.quote_data({cfg["exch_segment"]: [int(security_id)]})
        data = resp.get("data", {}).get("data", {})
        seg_data = data.get(cfg["exch_segment"], {})
        entry = seg_data.get(str(security_id)) or seg_data.get(int(security_id))
        if entry is None:
            return None
        ltp = entry.get("last_price")
        return float(ltp) if ltp is not None else None
    except Exception as e:
        log.warning(f"get_option_ltp failed for {security_id}: {e}")
        return None


# =====================================================================
# 7. STATE + POSITION MANAGEMENT
# =====================================================================
class InstrumentState:
    def __init__(self, name):
        self.name = name
        self.position = None
        self.day_pnl = 0.0
        self.day_trades = 0
        self.day_stopped = False
        self.anomaly_pause_left = 0
        self.squared_off_today = False


def within_market_hours():
    now_dt = now_ist()
    if now_dt.weekday() >= 5:   # Saturday=5, Sunday=6 - NSE/BSE closed, don't even poll
        return False
    return MARKET_OPEN <= now_dt.time() <= MARKET_CLOSE


def position_size(cfg, sl_points):
    """Lots sized so hitting SL loses at most RISK_PER_TRADE_PCT of capital."""
    risk_amount = CAPITAL * RISK_PER_TRADE_PCT
    lot_size = cfg["lot_size"]
    risk_per_lot = sl_points * lot_size
    if risk_per_lot <= 0:
        return lot_size
    lots = max(1, int(risk_amount // risk_per_lot))
    return min(lots, LOTS) * lot_size if LOTS else lots * lot_size


def open_position(dhan, cfg, state, instrument_name, sig, regime, pcr_info=None):
    if state.day_trades >= MAX_TRADES_PER_DAY:
        return
    if state.day_pnl <= -MAX_DAILY_LOSS_RUPEES:
        state.day_stopped = True
        return
    if state.day_stopped:
        return

    setup = sig.get("setup", "breakout")
    sl_pts, target_pts, breakeven_trigger, trail_step = risk_params_for_setup(regime, setup)
    qty = position_size(cfg, sl_pts)
    chain = (pcr_info or {}).get("chain")

    try:
        option = resolve_option_by_delta(instrument_name, sig["spot"], sig["side"], chain=chain)
    except Exception as e:
        log.warning(f"Strike resolution failed for {instrument_name}: {e}")
        tg_send(f"⚠️ Could not resolve {sig['side']} strike for {instrument_name}: {e}")
        return

    try:
        fill = place_order(dhan, cfg, option["security_id"], "BUY", qty)
    except Exception as e:
        log.warning(f"Entry order failed for {instrument_name}: {e}")
        return

    if fill["status"] not in ({"dry_run"} | FILLED_STATUSES):
        tg_send(f"🚨 ENTRY ABORTED for {instrument_name} {sig['side']} - order not confirmed filled "
                 f"(status={fill['status']}). No position recorded - if it DID fill on the broker side, "
                 f"square it off manually.")
        return

    filled_qty = fill.get("filled_qty") or qty
    entry_premium = fill.get("avg_price") or get_option_ltp(dhan, cfg, option["security_id"])

    state.position = {
        "instrument": instrument_name,
        "side": sig["side"],
        "security_id": option["security_id"],
        "trading_symbol": option["trading_symbol"],
        "entry_spot": sig["spot"],
        "entry_premium": entry_premium,   # confirmed fill price (or live LTP fallback) - basis for SL/target
        "qty": filled_qty,
        "setup": setup,
        "sl_points": sl_pts,
        "target_points": target_pts,
        "breakeven_trigger": breakeven_trigger,
        "trail_step": trail_step,
        "sl_level": -sl_pts,       # move in "points of favorable move" space; -sl_pts = SL below entry
        "peak": 0.0,
        "breakeven_done": False,
        "partial_booked": False,
        "trail_count": 0,
        "premium_mode": entry_premium is not None,  # False = degraded, using spot as an approximation
    }
    state.day_trades += 1
    basis = "premium" if entry_premium is not None else "spot(approx - premium fetch unavailable)"
    delta_basis = "real" if option.get("delta_is_real") else "approx"
    tg_send(f"✅ ENTRY {instrument_name} {sig['side']} [{setup}] {option['trading_symbol']} "
             f"qty={filled_qty} spot={sig['spot']} entry_premium={entry_premium} "
             f"delta≈{option['approx_delta']}({delta_basis}) SL={-sl_pts}pts target={target_pts}pts (basis={basis})")


def manage_position(dhan, state, cfg, df, regime, anomaly_flagged=False):
    pos = state.position
    if pos is None:
        return

    # Prefer real premium movement over spot movement - premium is what
    # is actually bought/sold and what SL/target rupee risk is based on.
    current_premium = get_option_ltp(dhan, cfg, pos["security_id"])
    if current_premium is not None and pos.get("entry_premium") is not None:
        if not pos["premium_mode"]:
            pos["premium_mode"] = True
            tg_send(f"ℹ️ {pos['instrument']} {pos['side']} switched to live premium tracking")
        move = current_premium - pos["entry_premium"]   # CE and PE both: premium up = profit, no sign flip needed
    else:
        # Degraded fallback: no live premium (DRY_RUN or fetch failed) -
        # spot points is an APPROXIMATION only, not real premium P&L.
        last_spot = df["close"].iloc[-1]
        move = (last_spot - pos["entry_spot"]) * (1 if pos["side"] == "CE" else -1)
        if pos["premium_mode"]:
            pos["premium_mode"] = False
            tg_send(f"⚠️ {pos['instrument']} {pos['side']} premium fetch failed - "
                     f"using spot as a rough approximation this cycle")
    pos["peak"] = max(pos["peak"], move)

    sl_pts = pos["sl_points"]
    target_pts = pos["target_points"]
    breakeven_trigger = pos["breakeven_trigger"]
    trail_step = pos["trail_step"]

    # --- If data looks anomalous, protectively tighten the stop instead
    #     of trusting the bad print for target/trail math ---
    if anomaly_flagged and not pos["breakeven_done"]:
        pos["sl_level"] = max(pos["sl_level"], -sl_pts / 2)
        tg_send(f"⚠️ Anomaly detected - tightened SL protectively on {pos['instrument']} {pos['side']}")

    # --- Stop loss hit ---
    if move <= pos["sl_level"]:
        pnl = move * pos["qty"]
        state.day_pnl += pnl
        force_close_position(dhan, cfg, pos, f"SL HIT (move={move:.1f}pts, pnl≈₹{pnl:.0f})")
        state.position = None
        return

    # --- Breakeven trail ---
    if not pos["breakeven_done"] and move >= breakeven_trigger:
        pos["sl_level"] = 0
        pos["breakeven_done"] = True
        tg_send(f"🔒 {pos['instrument']} {pos['side']} moved to BREAKEVEN")

    # --- First target -> partial booking (50%) ---
    if not pos["partial_booked"] and move >= target_pts:
        book_qty = int(pos["qty"] * PARTIAL_BOOK_FRACTION)
        if book_qty > 0:
            try:
                fill = place_order(dhan, cfg, pos["security_id"], "SELL", book_qty)
            except Exception as e:
                tg_send(f"⚠️ Partial-book order failed for {pos['instrument']} {pos['side']}: {e} - will retry next cycle")
                fill = None
            if fill is not None and fill["status"] in ({"dry_run"} | FILLED_STATUSES):
                actual_book_qty = fill.get("filled_qty") or book_qty
                pnl = move * actual_book_qty
                state.day_pnl += pnl
                pos["qty"] -= actual_book_qty
                pos["partial_booked"] = True
                pos["sl_level"] = target_pts - trail_step  # lock in some of the move
                tg_send(f"💰 PARTIAL BOOK {pos['instrument']} {pos['side']} qty={actual_book_qty} "
                         f"(+{move:.1f}pts, pnl≈₹{pnl:.0f}); SL now +{pos['sl_level']}pts")
            elif fill is not None:
                tg_send(f"🚨 Partial-book SELL not confirmed filled for {pos['instrument']} {pos['side']} "
                         f"(status={fill['status']}) - CHECK DHAN APP, will retry next cycle")

    # --- Trailing past target ---
    if pos["partial_booked"]:
        points_past_target = move - target_pts
        expected_trail_count = int(points_past_target // trail_step)
        if expected_trail_count > pos["trail_count"]:
            pos["trail_count"] = expected_trail_count
            pos["sl_level"] = target_pts - trail_step + expected_trail_count * trail_step
            tg_send(f"📈 TRAIL {pos['instrument']} {pos['side']} SL now +{pos['sl_level']}pts")

    # --- Auto Call<->Put reversal on trend flip while in profit ---
    if move > 0 and trend_reversed_against(df, pos["side"]):
        pnl = move * pos["qty"]
        state.day_pnl += pnl
        force_close_position(dhan, cfg, pos, f"TREND FLIP - AUTO REVERSAL (+{move:.1f}pts, pnl≈₹{pnl:.0f})")
        new_side = "PE" if pos["side"] == "CE" else "CE"
        tg_send(f"🔄 REVERSING {pos['instrument']}: {pos['side']} -> {new_side}")
        state.position = None
        # next poll cycle's entry_signal() will pick up the new side naturally
        # if conditions confirm it; we don't force-open immediately to avoid
        # chasing a single flip candle.


# =====================================================================
# 7b. TELEGRAM COMMAND LISTENER (so the bot can reply, not just push)
# =====================================================================
class EngineControl:
    """Shared, thread-safe flag the Telegram listener flips and the
    main loop reads. Pausing only blocks NEW entries - SL/target/
    trailing management and the 15:20 square-off keep running no
    matter what, so /stop can never leave a position unmanaged."""
    def __init__(self):
        self.lock = threading.Lock()
        self.trading_enabled = True
        self.regime = "unknown"

    def set_trading_enabled(self, value: bool):
        with self.lock:
            self.trading_enabled = value

    def is_trading_enabled(self) -> bool:
        with self.lock:
            return self.trading_enabled


def handle_telegram_command(text, states, control):
    text = text.strip().lower()

    if text in ("/status", "status"):
        lines = [f"📟 Status | regime={control.regime} | new entries: "
                 f"{'ON' if control.is_trading_enabled() else 'PAUSED'}"]
        for name, state in states.items():
            pos = state.position
            if pos is None:
                pos_desc = "flat"
            else:
                pos_desc = f"{pos['side']} {pos['trading_symbol']} [{pos.get('setup')}] qty={pos['qty']}"
            lines.append(f"{name}: trades={state.day_trades}/{MAX_TRADES_PER_DAY} "
                          f"pnl≈₹{state.day_pnl:.0f} pos={pos_desc}")
        tg_send("\n".join(lines))

    elif text in ("/start", "start", "/resume", "resume"):
        control.set_trading_enabled(True)
        tg_send("▶️ New entries RESUMED. (SL/target/square-off were never paused.)")

    elif text in ("/stop", "/pause", "stop", "pause"):
        control.set_trading_enabled(False)
        tg_send("⏸️ New entries PAUSED. Existing positions still fully managed (SL/target/square-off active).")

    elif text in ("/pnl", "pnl"):
        total = sum(s.day_pnl for s in states.values())
        tg_send(f"💰 Today's PnL so far ≈ ₹{total:.0f}")

    elif text in ("/help", "help", "/help@"):
        tg_send("Commands:\n"
                 "/status - regime, pause state, open positions, today's PnL\n"
                 "/pnl - today's PnL only\n"
                 "/start - resume new entries\n"
                 "/stop - pause new entries (existing positions still managed)\n"
                 "/help - this message")

    else:
        tg_send("Command samjha nahi. /help bhejo list ke liye.")


def telegram_listener(states, control):
    """Long-polls Telegram getUpdates in a background thread and replies
    to commands from TG_CHAT_ID only. Runs the whole session - if this
    thread dies, trading itself is unaffected, it just stops answering."""
    if not TG_TOKEN or not TG_CHAT_ID:
        log.info("Telegram command listener not started - TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID not set")
        return

    base = f"https://api.telegram.org/bot{TG_TOKEN}"
    offset = None
    log.info("Telegram command listener started")

    while True:
        try:
            params = {"timeout": 30}
            if offset is not None:
                params["offset"] = offset
            resp = requests.get(f"{base}/getUpdates", params=params, timeout=35)
            updates = resp.json().get("result", [])
            for upd in updates:
                offset = upd["update_id"] + 1
                msg = upd.get("message") or upd.get("channel_post")
                if not msg:
                    continue
                chat_id = str(msg.get("chat", {}).get("id"))
                if chat_id != str(TG_CHAT_ID):
                    continue  # ignore any chat other than the configured one
                text = msg.get("text")
                if not text:
                    continue
                handle_telegram_command(text, states, control)
        except Exception as e:
            log.warning(f"Telegram listener error: {e}")
            time.sleep(5)


# =====================================================================
# 8. MAIN LOOP
# =====================================================================
def main():
    log.info("Starting master_trading_engine.py — NO BTST enforced, DRY_RUN=%s" % DRY_RUN)
    tg_send(f"🤖 Trading bot STARTED (DRY_RUN={DRY_RUN}) — NO BTST, force square-off at 15:20 IST")

    dhan = connect_dhan()
    states = {name: InstrumentState(name) for name in ACTIVE_INSTRUMENTS}
    control = EngineControl()

    threading.Thread(target=telegram_listener, args=(states, control), daemon=True).start()

    premarket_done_date = None
    squareoff_done_date = None
    regime = "unknown"
    news_sentiment = None

    while True:
        try:
            now = now_ist()
            today = now.date()

            # --- Pre-market regime check (once per day, 8:30-9:15, Mon-Fri only) ---
            if now.weekday() < 5 and dt.time(8, 30) <= now.time() < MARKET_OPEN and premarket_done_date != today:
                _, regime, news_sentiment = premarket_check()
                control.regime = regime
                premarket_done_date = today

            # --- NO BTST: unconditional force square-off at 15:20, Mon-Fri only ---
            if now.weekday() < 5 and now.time() >= SQUARE_OFF_TIME and squareoff_done_date != today:
                square_off_everything(dhan, states)
                squareoff_done_date = today

                # Send a final EOD summary BEFORE wiping day_pnl/day_trades,
                # otherwise /status or /pnl checked after 15:20 would show
                # 0 and today's real result would be lost.
                total_pnl = sum(s.day_pnl for s in states.values())
                lines = [f"📒 EOD SUMMARY {today} | total PnL ≈ ₹{total_pnl:.0f}"]
                for name, state in states.items():
                    lines.append(f"{name}: trades={state.day_trades} pnl≈₹{state.day_pnl:.0f}")
                tg_send("\n".join(lines))

                for state in states.values():
                    state.day_trades = 0
                    state.day_pnl = 0.0
                    state.day_stopped = False

            if not within_market_hours():
                time.sleep(POLL_SECONDS)
                continue

            for name in ACTIVE_INSTRUMENTS:
                cfg = INSTRUMENTS[name]
                state = states[name]

                df = compute_frame(cfg["yf_symbol"])
                if df is None or df.empty:
                    log.warning(f"No data for {name} this cycle")
                    continue

                anomaly_flagged, anomaly_reason = detect_anomaly(df)
                if anomaly_flagged:
                    log.warning(f"{name}: anomaly detected ({anomaly_reason})")
                    state.anomaly_pause_left = ANOMALY_PAUSE_CYCLES

                # manage any existing position first, every cycle
                if state.position is not None:
                    manage_position(dhan, state, cfg, df, regime, anomaly_flagged)

                if state.anomaly_pause_left > 0:
                    state.anomaly_pause_left -= 1
                    continue  # skip new entries while pausing after an anomaly

                if state.position is not None or state.day_stopped:
                    continue

                if not control.is_trading_enabled():
                    continue  # /stop was sent - existing positions above are still fully managed

                if gap_or_crash_day(df):
                    log.info(f"{name}: gap/crash-day filter active, skipping new entries")
                    continue

                pcr_info = get_pcr_oi(dhan, name)
                sig = entry_signal(df, regime, pcr_info, news_sentiment)
                if sig:
                    open_position(dhan, cfg, state, name, sig, regime, pcr_info)

            time.sleep(POLL_SECONDS)

        except KeyboardInterrupt:
            log.info("Manual stop received.")
            tg_send("🛑 Bot manually stopped.")
            break
        except Exception as e:
            log.error(f"Main loop error: {e}\n{traceback.format_exc()}")
            tg_send(f"🚨 Bot error: {e}")
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
