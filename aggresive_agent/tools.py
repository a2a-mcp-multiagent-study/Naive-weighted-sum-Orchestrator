from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta, timezone
from dateutil import parser
from urllib.parse import quote

import math
import statistics

import feedparser
import pandas as pd
import yfinance as yf 
from google.adk.tools import FunctionTool
from transformers import pipeline


# ---------------------------
# Utilities
# ---------------------------

TZ_KST = timezone(timedelta(hours=9))
sentiment_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

def now_iso() -> str:
    return datetime.now(TZ_KST).isoformat(timespec="seconds")


def safe_parse_date(date_str: str) -> datetime:
    """
    ISO, YYYY-MM-DD, YYYY/MM/DD 등 다양한 입력을 허용해 UTC datetime으로 변환.
    """
    if not date_str:
        raise ValueError("Invalid date string (empty)")
    s = date_str.strip()
    # "Z"는 UTC로 해석
    if s.endswith("Z"):
        return parser.isoparse(s).astimezone(timezone.utc)
    try:
        dt = parser.isoparse(s)
    except Exception:
        dt = parser.parse(s)
    # timezone이 없으면 UTC로 가정
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt

def ensure_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def ema(values: List[float], period: int) -> List[float]:
    if not values or period <= 1:
        return values or []
    k = 2 / (period + 1)
    out = []
    prev = values[0]
    out.append(prev)
    for v in values[1:]:
        prev = v * k + prev * (1 - k)
        out.append(prev)
    return out

def sma(values: List[float], period: int) -> List[Optional[float]]:
    out = []
    buf: List[float] = []
    for v in values:
        buf.append(v)
        if len(buf) > period:
            buf.pop(0)
        out.append(sum(buf) / len(buf) if len(buf) == period else None)
    return out

def rsi(values: List[float], period: int = 14) -> List[Optional[float]]:
    if len(values) < period + 1:
        return [None] * len(values)
    gains, losses = [], []
    for i in range(1, period + 1):
        diff = values[i] - values[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    rs = avg_gain / avg_loss if avg_loss != 0 else float("inf")
    out = [None] * period + [100 - (100 / (1 + rs))]
    for i in range(period + 1, len(values)):
        diff = values[i] - values[i - 1]
        gain = max(diff, 0)
        loss = max(-diff, 0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else float("inf")
        out.append(100 - (100 / (1 + rs)))
    return out

def atr(ohlc: List[Dict[str, float]], period: int = 14) -> List[Optional[float]]:
    if len(ohlc) < period + 1:
        return [None] * len(ohlc)
    trs = []
    prev_close = ohlc[0]["close"]
    for i, bar in enumerate(ohlc):
        tr = max(
            bar["high"] - bar["low"],
            abs(bar["high"] - prev_close),
            abs(bar["low"] - prev_close),
        )
        trs.append(tr)
        prev_close = bar["close"]
    # Wilder's smoothing
    out = [None] * len(ohlc)
    first = sum(trs[1:period+1]) / period  # use first full window
    out[period] = first
    for i in range(period + 1, len(trs)):
        out[i] = (out[i - 1] * (period - 1) + trs[i]) / period
    return out

def pct_change(values: List[float]) -> List[Optional[float]]:
    out: List[Optional[float]] = [None]
    for i in range(1, len(values)):
        prev = values[i - 1]
        out.append((values[i] - prev) / prev if prev else None)
    return out

def safe_std(values: List[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0

# ---------------------------
# 1) MarketData (mock with realistic series; plug real providers)
# ---------------------------

def get_market_data(
    symbol: str,
    start: str,
    end: str,
    interval: str = "1d",
) -> Dict[str, Any]:
    """
    yfinance 실데이터 기반 OHLCV.
    - end를 현재시각으로 클램프
    - end는 exclusive 경향 → +1일 버퍼
    - download 실패 시 history(period=...)로 폴백
    - 반환 ts는 KST
    """
    # 1) interval 정규화
    interval_map = {
        "1d": "1d",
        "1h": "1h",
        "5m": "5m",
        "1m": "1m",   # ← 공백 제거(중요)
    }
    yf_interval = interval_map.get(interval, "1d")

    # 2) 날짜 파싱 + end 클램프
    start_dt = safe_parse_date(start)
    end_dt = safe_parse_date(end)
    now_utc = datetime.now(timezone.utc)
    if end_dt > now_utc:
        end_dt = now_utc
    if end_dt <= start_dt:
        # 최소 7일 윈도우 보장
        start_dt = end_dt - timedelta(days=7)

    # 3) download 시도
    df = yf.download(
        tickers=symbol,
        start=start_dt,
        end=end_dt + timedelta(days=1),  # end exclusive 보정
        interval=yf_interval,
        auto_adjust=False,
        threads=False,
        progress=False,
    )
    
    # 4) 실패/빈응답 시 history(period=...)로 폴백
    if df is None or df.empty:
        # 기간 길이에 맞는 period 추정
        span = end_dt - start_dt
        if span.days <= 7:
            period = "7d"
        elif span.days <= 30:
            period = "1mo"
        elif span.days <= 90:
            period = "3mo"
        elif span.days <= 365:
            period = "1y"
        else:
            period = "max"

        try:
            t = yf.Ticker(symbol)
            df = t.history(period=period, interval=yf_interval, auto_adjust=False)
        except Exception:
            df = pd.DataFrame()

        # history는 전체 기간을 주므로 start/end로 한 번 더 슬라이싱
        if df is not None and not df.empty:
            # 인덱스가 tz-naive일 수 있으니 일단 UTC 가정
            idx = df.index
            if isinstance(idx, pd.DatetimeIndex):
                # UTC 기준으로 자르고 나중에 KST 변환
                mask = (idx >= start_dt) & (idx <= end_dt + timedelta(days=1))
                df = df.loc[mask]

    # 5) 최종 체크
    if df is None or df.empty:
        return {
            "timestamp": now_iso(),
            "symbol": symbol.upper(),
            "interval": yf_interval,
            "bars": [],
            "note": f"empty after download/history. start={start_dt.isoformat()}, end={end_dt.isoformat()}",
            "disclaimer": "실데이터 조회 실패(티커/기간/인터벌/야후 가용성). 매매 실행 없음.",
        }

    # 6) 컬럼 정규화
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join(col).strip() for col in df.columns.values]
    
    rename_map = {
        "Open " + symbol: "open",
        "High " + symbol: "high",
        "Low " + symbol: "low",
        "Close " + symbol: "close",
        "Adj Close " + symbol: "adj_close",
        "Volume " + symbol: "volume",
    }
    for k in list(df.columns):
        if k in rename_map and rename_map[k] not in df.columns:
            df.rename(columns={k: rename_map[k]}, inplace=True)
    # 7) 레코드 변환(KST) + NaN 필터
    bars: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        try:
            ts = pd.to_datetime(idx)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            else:
                ts = ts.tz_convert(timezone.utc)

            ts_kst = ts.astimezone(TZ_KST)
            o = float(row.get("open", float("nan")))
            h = float(row.get("high", float("nan")))
            l = float(row.get("low", float("nan")))
            c = float(row.get("close", float("nan")))
            
            if any(pd.isna([o, h, l, c])):
                continue
            v = row.get("volume", 0)
            v = 0 if pd.isna(v) else int(v)
            bars.append({"ts": ts_kst, "open": o, "high": h, "low": l, "close": c, "volume": v})
        except Exception:
            continue

    if not bars:
        # 디버깅용 노트 포함
        return {
            "timestamp": now_iso(),
            "symbol": symbol.upper(),
            "interval": yf_interval,
            "bars": [],
            "note": f"rows existed but filtered as NaN. columns={list(df.columns)} head=\n{df.head(3)}",
            "disclaimer": "정규화/NaN 필터 이후 데이터 없음. 매매 실행 없음.",
        }

    return {
        "timestamp": now_iso(),
        "symbol": symbol.upper(),
        "interval": yf_interval,
        "bars": bars,
        "note": f"ok start={start_dt.isoformat()} end={end_dt.isoformat()} rows={len(bars)}",
        "disclaimer": "yfinance 실데이터 기반. 매매 실행 없음.",
    }
    
# ---------------------------
# 2) Fundamentals (mock; plug SEC/DART/Finnhub etc.)
# ---------------------------

def get_fundamentals(symbol: str, period: str = "annual") -> Dict[str, Any]:
    """
    yfinance로 재무/밸류에이션 스냅샷을 구성.
    가용 데이터가 없는 항목은 None 처리.
    period: 'annual' | 'quarterly' (yfinance 테이블 선택에 활용)
    """
    t = yf.Ticker(symbol)
    snap: Dict[str, Any] = {
        "revenue": None,
        "net_income": None,
        "eps": None,
        "roe": None,
        "debt_to_equity": None,
        "trailing_pe": None,
        "pbr": None,
        "analyst_consensus": None,  # yfinance에서 직접 제공 안함 → None
    }

    # price/ratios
    try:
        fi = getattr(t, "fast_info", None)
        if fi:
            snap["trailing_pe"] = getattr(fi, "pe_ratio", None)
            # pbr: price_to_book (없으면 None)
            if hasattr(fi, "price_to_book"):
                snap["pbr"] = getattr(fi, "price_to_book", None)
    except Exception:
        pass

    # 발행주식수, 시가 등
    shares_out = None
    try:
        if fi:
            last_price = getattr(fi, "last_price", None)
            shares_out = getattr(fi, "shares", None)
    except Exception:
        pass

    # 재무제표 (연간/분기)
    try:
        if period == "quarterly":
            inc = t.quarterly_income_stmt
            bal = t.quarterly_balance_sheet
        else:
            inc = t.income_stmt
            bal = t.balance_sheet

        # 최신 열(가장 최근 분기/연도) 추출
        if inc is not None and not inc.empty:
            latest_col = inc.columns[0]
            rev = inc.get("Total Revenue", pd.Series(dtype=float))
            ni = inc.get("Net Income", pd.Series(dtype=float))
            snap["revenue"] = None if rev is None or rev.empty else int(rev.get(latest_col, None)) if not pd.isna(rev.get(latest_col, None)) else None
            snap["net_income"] = None if ni is None or ni.empty else int(ni.get(latest_col, None)) if not pd.isna(ni.get(latest_col, None)) else None

        if bal is not None and not bal.empty:
            latest_col_b = bal.columns[0]
            total_eq = bal.get("Total Stockholder Equity", pd.Series(dtype=float))
            total_debt = bal.get("Total Debt", pd.Series(dtype=float))
            eq_val = None if total_eq is None or total_eq.empty else float(total_eq.get(latest_col_b, None))
            debt_val = None if total_debt is None or total_debt.empty else float(total_debt.get(latest_col_b, None))
            # ROE
            if snap["net_income"] is not None and eq_val and eq_val != 0:
                snap["roe"] = round(float(snap["net_income"]) / eq_val, 4)
            # D/E
            if debt_val is not None and eq_val and eq_val != 0:
                snap["debt_to_equity"] = round(debt_val / eq_val, 4)
    except Exception:
        pass

    # EPS (TTM 또는 간이 계산)
    try:
        # yfinance가 EPS TTM을 직접 주지 않으면 net_income / shares_out으로 근사
        if snap["net_income"] is not None and shares_out:
            snap["eps"] = round(float(snap["net_income"]) / float(shares_out), 4)
    except Exception:
        pass

    return {
        "timestamp": now_iso(),
        "symbol": symbol.upper(),
        "period": period,
        "snapshot": snap,
        "disclaimer": "yfinance 공개 재무데이터 기반 스냅샷(항목 부재 시 None). 매매 실행 없음.",
    }

# ---------------------------
# 3) Technicals (EMA/SMA/RSI/MACD/Bollinger/ATR, signals)
# ---------------------------

def compute_technicals(
    bars: List[Dict[str, Any]],
    indicators: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Compute common technical indicators and an aggregated aggressive signal.
    """
    indicators = indicators or [
        {"name": "EMA", "params": {"period": 20}},
        {"name": "SMA", "params": {"period": 50}},
        {"name": "RSI", "params": {"period": 14}},
        {"name": "MACD", "params": {"fast": 12, "slow": 26, "signal": 9}},
        {"name": "BOLL", "params": {"period": 20, "std": 2}},
        {"name": "ATR", "params": {"period": 14}},
    ]

    closes = [b["close"] for b in bars]
    out: Dict[str, Any] = {"indicators": {}, "signal": None, "confidence": 0.0, "rationale": []}

    # EMA/SMA
    for ind in indicators:
        name = ind["name"].upper()
        p = ind.get("params", {})
        if name == "EMA":
            period = int(p.get("period", 20))
            out["indicators"][f"EMA_{period}"] = ema(closes, period)
        elif name == "SMA":
            period = int(p.get("period", 50))
            out["indicators"][f"SMA_{period}"] = sma(closes, period)
        elif name == "RSI":
            period = int(p.get("period", 14))
            out["indicators"][f"RSI_{period}"] = rsi(closes, period)
        elif name == "MACD":
            fast = int(p.get("fast", 12))
            slow = int(p.get("slow", 26))
            sigp = int(p.get("signal", 9))
            ema_fast = ema(closes, fast)
            ema_slow = ema(closes, slow)
            macd = [ (f - s) if (f is not None and s is not None) else None
                     for f, s in zip(ema_fast, ema_slow) ]
            macd_clean = [x for x in macd if x is not None]
            sig = ema(macd_clean, sigp) if macd_clean else []
            # align signal length
            sig_full = [None]*(len(macd)-len(sig)) + sig
            hist = [ (m - s) if (m is not None and s is not None) else None
                     for m, s in zip(macd, sig_full) ]
            out["indicators"]["MACD"] = macd
            out["indicators"]["MACD_SIGNAL"] = sig_full
            out["indicators"]["MACD_HIST"] = hist
        elif name == "BOLL":
            period = int(p.get("period", 20))
            stdn = float(p.get("std", 2))
            ma = sma(closes, period)
            u, l = [], []
            for i in range(len(closes)):
                if i+1 >= period:
                    window = closes[i+1-period:i+1]
                    sd = statistics.pstdev(window) if len(window)>1 else 0.0
                    u.append((ma[i] if ma[i] is not None else None) + stdn*sd if ma[i] is not None else None)
                    l.append((ma[i] if ma[i] is not None else None) - stdn*sd if ma[i] is not None else None)
                else:
                    u.append(None); l.append(None)
            out["indicators"][f"BOLL_MA_{period}"] = ma
            out["indicators"][f"BOLL_UP_{period}"] = u
            out["indicators"][f"BOLL_LOW_{period}"] = l
        elif name == "ATR":
            period = int(p.get("period", 14))
            out["indicators"][f"ATR_{period}"] = atr(bars, period)

    # Aggressive signal heuristic
    last_close = closes[-1]
    ema20 = out["indicators"].get("EMA_20", [None])[-1]
    rsi14 = out["indicators"].get("RSI_14", [None])[-1]
    macd_hist = out["indicators"].get("MACD_HIST", [None])[-1]
    boll_up = out["indicators"].get("BOLL_UP_20", [None])[-1]
    boll_lo = out["indicators"].get("BOLL_LOW_20", [None])[-1]
    atr14 = out["indicators"].get("ATR_14", [None])[-1]

    score = 0.0
    if ema20 and last_close > ema20: score += 0.25
    if rsi14 and 45 <= rsi14 <= 65:  score += 0.2
    if macd_hist and macd_hist > 0:  score += 0.2
    if boll_up and last_close < boll_up: score += 0.15
    if atr14: score += 0.1  # volatility present suits aggressive entries
    score = clamp(score, 0, 1)
    action = "BUY" if score >= 0.6 else ("HOLD" if score >= 0.4 else "SHORT" if macd_hist and macd_hist < 0 and rsi14 and rsi14 > 60 else "HOLD")

    rationale = []
    if ema20 is not None: rationale.append(f"Price {'>' if last_close>ema20 else '<='} EMA20")
    if rsi14 is not None: rationale.append(f"RSI14={round(rsi14,1)} (45~65 선호)")
    if macd_hist is not None: rationale.append(f"MACD hist={'+' if macd_hist>0 else '-'}")
    if boll_up is not None and boll_lo is not None:
        rationale.append("Within Bollinger bands (mean-reversion risk 낮음)" if (boll_lo < last_close < boll_up) else "Band edge 접근")

    out["signal"] = action
    out["confidence"] = round(score, 2)
    out["rationale"] = rationale
    return out

# ---------------------------
# 4) News & Sentiment (mock; plug RSS/Twitter/DART)
# ---------------------------

def fetch_news_sentiment(
    query: str,
    since_minutes: int = 180,
    max_items: int = 10,
) -> Dict[str, Any]:
    """
    Fetch live news headlines from Google News RSS for a given query,
    and perform multilingual sentiment analysis using a Hugging Face model.
    """
    # 1️⃣ Fetch RSS
    rss_url = f"https://news.google.com/rss/search?q={query}+when:1d&hl=en-US&gl=US&ceid=US:en"
    rss_url = quote(rss_url, safe=':/?&=+')


    feed = feedparser.parse(rss_url)
    entries = feed.entries[:max_items]

    items: List[Dict[str, Any]] = []
    now = datetime.now(TZ_KST)
    horizon = now - timedelta(minutes=since_minutes)

    # 2️⃣ Iterate through recent items
    for entry in entries:
        try:
            ts = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc).astimezone(TZ_KST)
        except Exception:
            ts = now

        if ts < horizon:
            continue

        title = entry.title
        summary = getattr(entry, "summary", "")[:200]
        text_for_analysis = title + " " + summary

        # 3️⃣ Run sentiment model (1~5 stars → normalize 0~1)
        try:
            result = sentiment_model(text_for_analysis[:512])[0]
            label = result["label"]  # e.g., "4 stars"
            stars = int(label.split()[0])
            sentiment_score = round((stars - 1) / 4, 2)  # normalize 0~1
        except Exception as e:
            sentiment_score = 0.5

        items.append({
            "time": ts.isoformat(timespec="seconds"),
            "source": "Google News",
            "title": title,
            "summary": summary,
            "sentiment_score": sentiment_score,
            "link": entry.link,
            "key_sentences": [title],
        })

    # 4️⃣ Aggregate sentiment
    agg = statistics.mean([x["sentiment_score"] for x in items]) if items else 0.5

    return {
        "timestamp": now_iso(),
        "query": query,
        "since_minutes": since_minutes,
        "aggregate_sentiment": round(agg, 2),
        "items": items,
        "disclaimer": "뉴스 기반 감성 분석 결과입니다. 실제 매매 판단은 투자자 본인에게 있습니다.",
    }
    
# ---------------------------
# 5) Risk Analytics / Portfolio Simulator
# ---------------------------

def analyze_portfolio_risk(
    positions: List[Dict[str, Any]],
    history: Dict[str, List[float]] | None = None,
    horizon_days: int = 10,
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """
    단순 모수적 VaR & MaxDD 추정.
    history 미제공 시, 각 심볼의 1일봉 최근 252개를 yfinance로 수집해 일간 수익률 표준편차를 사용.
    """
    weights: List[float] = []
    vols: List[float] = []

    total_value = sum(float(p["qty"]) * float(p["price"]) for p in positions) or 1.0

    for p in positions:
        symbol = p["symbol"].upper()
        qty = float(p["qty"])
        px = float(p["price"])
        value = qty * px
        w = value / total_value
        weights.append(w)

        # 히스토리 확보
        hist_prices = (history or {}).get(symbol, [])
        if not hist_prices:
            # yfinance에서 252영업일(약 1년) 데이터 가져오기
            df = yf.download(symbol, period="1y", interval="1d", auto_adjust=False, threads=False, progress=False)
            if df is not None and not df.empty:
                closes = df["Close"].dropna().tolist()
                hist_prices = closes[-252:] if len(closes) > 252 else closes

        # 일간 변동성
        if hist_prices and len(hist_prices) > 5:
            rets = []
            for i in range(1, len(hist_prices)):
                prev = hist_prices[i-1]
                curr = hist_prices[i]
                if prev and not pd.isna(prev) and not pd.isna(curr) and prev != 0:
                    rets.append((curr - prev) / prev)
            vol = statistics.pstdev(rets) if len(rets) > 1 else 0.02
        else:
            vol = 0.03  # 데이터 부족 시 보수적 기본값

        vols.append(vol)

    # 상관 0 가정(간이)
    port_vol = math.sqrt(sum((w*v)**2 for w, v in zip(weights, vols)))
    z = 1.645 if abs(confidence - 0.95) < 1e-6 else 2.326 if abs(confidence - 0.99) < 1e-6 else 1.645
    daily_var = z * port_vol * total_value
    horizon_var = daily_var * math.sqrt(max(horizon_days, 1))
    max_dd_guess = 2.5 * port_vol * total_value

    per_trade_risk_pct = 0.02
    rec_position_size = per_trade_risk_pct * total_value

    return {
        "timestamp": now_iso(),
        "portfolio_value": round(total_value, 2),
        "daily_volatility_est": round(port_vol, 4),
        "var": {
            "confidence": confidence,
            "horizon_days": horizon_days,
            "daily_VaR_est": round(daily_var, 2),
            "horizon_VaR_est": round(horizon_var, 2),
        },
        "max_drawdown_est": round(max_dd_guess, 2),
        "position_size_hint": {
            "aggressive_risk_pct": per_trade_risk_pct,
            "recommended_value_per_trade": round(rec_position_size, 2),
        },
        "disclaimer": "yfinance 시세 기반 간이 리스크 추정(상관/테일리스크 미반영). 매매 실행 없음.",
    }
    
# ---------------------------
# 6) Event Calendar / Macro Data (mock; plug FOMC/CPI/KRX)
# ---------------------------

def fetch_event_calendar(
    date_from: str,
    date_to: str,
    symbols: Optional[List[str]] = None,
    region: Optional[str] = None,
) -> Dict[str, Any]:
    """
    심볼 기준 이벤트(어닝, 배당)를 yfinance로 조회.
    거시 이벤트(FOMC/CPI 등)는 별도 API 필요 → 여기서는 제공하지 않음.
    """
    start_dt = datetime.fromisoformat(date_from.replace("Z", "+00:00"))
    end_dt = datetime.fromisoformat(date_to.replace("Z", "+00:00"))
    symbols = ensure_list(symbols)

    events: List[Dict[str, Any]] = []

    # 심볼 지정 없으면 빈 결과(거시 API 미연동)
    if not symbols:
        return {
            "timestamp": now_iso(),
            "date_range": [date_from, date_to],
            "region": region or "global",
            "events": events,
            "disclaimer": "심볼 이벤트만 제공(어닝/배당). 거시 일정은 외부 캘린더 API 연동 필요. 매매 실행 없음.",
        }

    for sym in symbols:
        try:
            t = yf.Ticker(sym)
            cal = t.calendar  # DataFrame 형태일 수도, dict-like일 수도 있음
            if cal is not None and not (isinstance(cal, dict) and len(cal) == 0):
                # yfinance 구조가 자주 바뀌므로 dict & df 모두 처리
                def _extract(key_names):
                    # 다양한 키 후보에서 첫 매칭을 반환
                    for k in key_names:
                        if isinstance(cal, dict) and k in cal:
                            return cal[k]
                        if hasattr(cal, "loc") and k in cal.index:
                            return cal.loc[k].values[0] if hasattr(cal.loc[k], "values") else cal.loc[k]
                    return None

                earnings_dt = _extract(["Earnings Date", "EarningsDate"])
                ex_div_dt = _extract(["Ex-Dividend Date", "Ex-DividendDate", "Ex-Dividend"])

                def _to_kst_iso(x):
                    try:
                        if isinstance(x, (list, tuple)) and len(x) > 0:
                            x = x[0]
                        if isinstance(x, (pd.Timestamp, datetime)):
                            dt = x.to_pydatetime() if isinstance(x, pd.Timestamp) else x
                        else:
                            dt = pd.to_datetime(x, errors="coerce").to_pydatetime()
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        return dt.astimezone(TZ_KST).isoformat(timespec="seconds")
                    except Exception:
                        return None

                if earnings_dt:
                    iso = _to_kst_iso(earnings_dt)
                    if iso:
                        dt = datetime.fromisoformat(iso)
                        if start_dt <= dt <= end_dt:
                            events.append({
                                "time": iso,
                                "event": f"Earnings ({sym.upper()})",
                                "expected_impact": "Medium",
                                "notes": "Earnings announcement window",
                            })
                if ex_div_dt:
                    iso = _to_kst_iso(ex_div_dt)
                    if iso:
                        dt = datetime.fromisoformat(iso)
                        if start_dt <= dt <= end_dt:
                            events.append({
                                "time": iso,
                                "event": f"Ex-Dividend ({sym.upper()})",
                                "expected_impact": "Low",
                                "notes": "Dividend-related price adjustment possible",
                            })
        except Exception:
            continue

    events.sort(key=lambda x: x["time"])

    return {
        "timestamp": now_iso(),
        "date_range": [date_from, date_to],
        "region": region or "by-symbol",
        "events": events,
        "disclaimer": "yfinance 종목 이벤트 기반(거시 일정 미포함). 매매 실행 없음.",
    }
    
# ---------------------------
# Aggressive Recommendation Wrapper
# ---------------------------

def aggressive_recommendation(
    market: str,
    symbol: str,
    bars: List[Dict[str, Any]],
    news_sentiment: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Fuse technicals + (optional) sentiment into a single aggressive recommendation.
    """
    tech = compute_technicals(bars)
    conf = tech["confidence"]
    action = tech["signal"]

    # Sentiment tilt
    if news_sentiment:
        agg_sent = news_sentiment.get("aggregate_sentiment", 0.5)
        if agg_sent >= 0.6:
            conf = clamp(conf + 0.05, 0, 1)
        elif agg_sent <= 0.4:
            # Slightly decrease confidence for BUY; increase for SHORT
            conf = clamp(conf - 0.05 if action == "BUY" else conf + 0.05, 0, 1)

    last_price = bars[-1]["close"]
    atr14 = tech["indicators"].get("ATR_14", [None])[-1]
    # ATR-based stop/targets (aggressive)
    stop = round(last_price - 1.5 * (atr14 or last_price * 0.02), 3) if action == "BUY" else \
           round(last_price + 1.2 * (atr14 or last_price * 0.02), 3)
    targets = [round(last_price + k * (atr14 or last_price * 0.02), 3) for k in (1.5, 3.0)] if action == "BUY" else \
              [round(last_price - k * (atr14 or last_price * 0.02), 3) for k in (1.2, 2.5)]

    return {
        "timestamp": now_iso(),
        "market": market.upper(),
        "symbol": symbol.upper(),
        "recommendation": action,
        "confidence": round(conf, 2),
        "entry": last_price,
        "stop_loss": stop,
        "targets": targets,
        "rationale": tech["rationale"] + (["Sentiment tilt applied"] if news_sentiment else []),
        "risks": ["Event risk in calendar window", "High short-term volatility"],
        "disclaimer": "이 정보는 매매 권유가 아닙니다. 매매 실행 없음. 최종 판단은 사용자 책임입니다.",
    }


TOOLS = [
    FunctionTool(func=get_market_data),
    FunctionTool(func=get_fundamentals),
    FunctionTool(func=compute_technicals),
    FunctionTool(func=fetch_news_sentiment),
    FunctionTool(func=analyze_portfolio_risk),
    FunctionTool(func=fetch_event_calendar),
    FunctionTool(func=aggressive_recommendation),
]
