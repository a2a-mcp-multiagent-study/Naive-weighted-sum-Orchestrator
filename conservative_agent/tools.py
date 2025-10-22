from __future__ import annotations
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import yfinance as yf

from google.adk.tools import FunctionTool


# ---------------------------------------------------------
# 1) Function Tools (보수적 투자 점검용)
# ---------------------------------------------------------

def fetch_price(ticker: str, period: Optional[str] = None, interval: Optional[str] = None) -> Dict[str, Any]:
    """지정 종목의 가격 이력/현재가를 조회합니다.
    Args:
        ticker: 티커 
            ex) AAPL, SPY
        period: 가격 조회 기간
            default: '3y'
            ex)'1y','3y','5y','max'
        interval: 가격 조회 간격
            default: '1d'
            ex)'1d','1wk','1mo'
    Returns:
        {status, ticker, last_price, currency}
    """
    
    period = "3y" if period is None else period
    interval = "1d" if interval is None else interval
    
    t = yf.Ticker(ticker)
    hist = t.history(period=period, interval=interval, auto_adjust=True)
    if hist.empty:
        return {"status": "error", "message": f"No price data for {ticker}"}
    last = float(hist["Close"].iloc[-1])
    info = t.get_info() or {}

    return {
        "status": "success",
        "ticker": ticker,
        "last_price": last,
        "currency": info.get("currency"),
    }


def fetch_fundamentals(ticker: str) -> Dict[str, Any]:
    """기본 재무/밸류에이션 지표를 조회합니다.
    Args:
        ticker: 티커 
            ex) AAPL, SPY
    Returns:
        {status, market_cap, pe, pb, dividend_yield, sector}
    """
    t = yf.Ticker(ticker)
    info = t.get_info() or {}
    return {
        "status": "success",
        "ticker": ticker,
        "market_cap": info.get("marketCap"),
        "pe": info.get("trailingPE") or info.get("forwardPE"),
        "pb": info.get("priceToBook"),
        "dividend_yield": info.get("dividendYield"),
        "sector": info.get("sector"),
    }


def check_event_risk(ticker: str, lookahead_days: Optional[int] = None) -> Dict[str, Any]:
    """향후 실적/배당(Ex-date) 이벤트 리스크를 점검합니다.
    Args:
        ticker: 티커 
            ex) AAPL, SPY
        lookahead_days: 이벤트 리스크 점검 기간
            default: 60
            ex) 60, 30, 15
    Returns:
        {status, earnings_date, ex_div_date, within_lookahead}
    """
    lookahead_days = 60 if lookahead_days is None else lookahead_days
    
    t = yf.Ticker(ticker)
    cal = t.calendar or {}
    def normalize_date(x):
        if isinstance(x, (list, tuple)) and x:
            x = x[0]
        if isinstance(x, pd.Timestamp):
            return x.to_pydatetime()
        try:
            return pd.to_datetime(x).to_pydatetime()
        except Exception:
            return None

    earnings = normalize_date(cal.get("Earnings Date"))
    exdiv = normalize_date(cal.get("Ex-Dividend Date"))
    
    horizon = datetime.utcnow() + timedelta(days=lookahead_days)
    def in_window(dt): return bool(dt and dt <= horizon)
    
    return {
        "status": "success",
        "ticker": ticker,
        "earnings_date": earnings.isoformat() if earnings else None,
        "ex_div_date": exdiv.isoformat() if exdiv else None,
        "within_lookahead": in_window(earnings) or in_window(exdiv),
    }

def calc_risk(
    tickers: List[str],
    benchmark: Optional[str] = None,
    period: Optional[str] = None,
    interval: Optional[str] = None,
) -> Dict[str, Any]:
    """변동성/최대낙폭/베타/상관관계를 계산합니다.
    Args:
        tickers: 티커 목록
            ex) [AAPL, SPY]
        benchmark: 벤치마크
            default: SPY
            ex) SPY, ^KS11
        period: 가격 조회 기간
            default: '3y'
            ex)'1y','3y','5y','max'
        interval: 가격 조회 간격
            default: '1d'
            ex)'1d','1wk','1mo'
    Returns:
        {status, as_of, benchmark, assets: [{ticker, ann_vol, mdd, beta}], corr}
    """
    period = "3y" if period is None else period
    interval = "1d" if interval is None else interval
    benchmark = "SPY" if benchmark is None else benchmark
    
    uniq = list(dict.fromkeys(tickers + [benchmark]))
    data = yf.download(uniq, period=period, interval=interval, auto_adjust=True, progress=False)
    if "Close" not in data:
        return {"status": "error", "message": "Failed to download prices"}
    px = data["Close"].dropna(how="any")
    rets = px.pct_change().dropna()
    ann = 252 if interval == "1d" else 52 if interval == "1wk" else 12

    if benchmark not in rets.columns:
        return {"status": "error", "message": f"Benchmark {benchmark} missing"}
    b = rets[benchmark]

    assets = []
    for k in tickers:
        if k not in rets.columns: 
            continue
        r = rets[k]
        ann_vol = float(r.std() * np.sqrt(ann))
        cum = (1 + r).cumprod()
        peak = cum.cummax()
        mdd = float((cum/peak - 1).min())
        cov = np.cov(r, b)[0, 1]
        varb = np.var(b)
        beta = float(cov/varb) if varb > 0 else None
        assets.append({"ticker": k, "ann_vol": ann_vol, "max_drawdown": mdd, "beta": beta})

    corr = rets[[c for c in rets.columns if c in tickers]].corr()
    
    
    return {
        "status": "success",
        "as_of": datetime.now(timezone.utc).isoformat(),
        "benchmark": benchmark,
        "assets": assets,
        "corr": corr.to_dict(),
    }

def fetch_fx(
    symbol: Optional[str] = None, 
    period: Optional[str] = None,
) -> Dict[str, Any]:
    """환율 종가 조회(기본 USD/KRW)."""

    symbol = "USDKRW=X" if symbol is None else symbol
    period = "1y" if period is None else period
    
    t = yf.Ticker(symbol)
    hist = t.history(period=period, interval="1d")
    if hist.empty:
        return {"status": "error", "message": f"No FX data for {symbol}"}
    return {"status": "success", "symbol": symbol, "last": float(hist["Close"].iloc[-1])}


# ---------------------------------------------------------
# 🇰🇷 Korea Market Tools (국장용)
# ---------------------------------------------------------

def fetch_korea_index(
    period: Optional[str] = None,
    interval: Optional[str] = None,
    indices: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """코스피/코스닥 등 대표 지수의 가격 이력과 최근 종가를 조회합니다.
    Args:
        period: '1y'|'3y'|'5y'|'max'
        interval: '1d'|'1wk'|'1mo'
        indices: yfinance 심볼 목록 (기본: KOSPI/^KS11, KOSDAQ/^KQ11, KODEX200/069500.KS)
    Returns:
        {status, last: {symbol: last_close}, csv_path}
    """
    period = "3y" if period is None else period
    interval = "1d" if interval is None else interval
    if indices is None:
        indices = ["^KS11", "^KQ11", "069500.KS"]  # KOSPI, KOSDAQ, KODEX200(현물 ETF)
    
    data = yf.download(indices, period=period, interval=interval, auto_adjust=True, progress=False)
    if "Close" not in data:
        return {"status": "error", "message": "Failed to download index prices"}
    px = data["Close"].dropna(how="any")
    last = {}
    for s in indices:
        try:
            last[s] = float(px[s].iloc[-1])
        except Exception:
            # 단일 컬럼일 경우 등 edge 대응
            if s == indices[0] and px.shape[1] == 1:
                last[s] = float(px.iloc[-1, 0])
    path = "/tmp/korea_indices.csv"
    px.to_csv(path)
    return {"status": "success", "last": last, "csv_path": path, "period": period, "interval": interval}

def korea_market_health(
    period: Optional[str] = None,
    interval: Optional[str] = None,
    base_symbol: Optional[str] = None,           # 기본: KOSPI
    alt_symbol: Optional[str] = None,  # 보조: KOSDAQ
    regime_ma_days: Optional[int] = None,
) -> Dict[str, Any]:
    """국장(코스피 중심)의 '보수적 건강도'를 산출합니다.
    - 추세: 종가 vs 120D SMA (regime_ma_days)
    - 변동성: 14D ATR / 종가 (정규화)
    - 하락 위험: 52주 고점 대비 괴리율(drawdown_from_52w_high)
    - 추세크로스: 50D vs 200D (golden/death)
    - 간단 레짐 분류: 'risk_off' | 'neutral' | 'cautious_on'
    Returns:
        {status, as_of, base_symbol, metrics, regime}
    """
    period = "2y" if period is None else period
    interval = "1d" if interval is None else interval
    base_symbol = "^KS11" if base_symbol is None else base_symbol
    alt_symbol = "^KQ11" if alt_symbol is None else alt_symbol
    regime_ma_days = 120 if regime_ma_days is None else regime_ma_days
    
    syms = [s for s in [base_symbol, alt_symbol] if s]
    data = yf.download(syms, period=period, interval=interval, auto_adjust=True, progress=False)
    if "Close" not in data:
        return {"status": "error", "message": "Failed to download market prices"}
    close = data["Close"].dropna(how="any")
    high = data.get("High", close)
    low  = data.get("Low",  close)

    # 단일/다중 심볼 안전 처리
    def col(df, s):
        return df[s] if s in df.columns else df.iloc[:, 0]

    c = col(close, base_symbol)
    h = col(high,  base_symbol)
    l = col(low,   base_symbol)

    # 이동평균 & 기울기(미분 근사)
    sma20  = c.rolling(20).mean()
    sma50  = c.rolling(50).mean()
    sma120 = c.rolling(regime_ma_days).mean()
    sma200 = c.rolling(200).mean()

    slope20  = (sma20 - sma20.shift(5)) / 5
    slope60  = (c.rolling(60).mean() - c.rolling(60).mean().shift(5)) / 5
    slope120 = (sma120 - sma120.shift(5)) / 5

    # ATR(14)
    tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    vol_norm = (atr14 / c).iloc[-1] if len(atr14) else None

    # 52주 고점 대비 하락폭
    c_52w_high = c.rolling(252).max()
    dd_52w = float(c.iloc[-1] / c_52w_high.iloc[-1] - 1) if c_52w_high.notna().iloc[-1] else None

    # 골든/데스 크로스
    golden = bool(sma50.iloc[-1] > sma200.iloc[-1] and sma50.iloc[-2] <= sma200.iloc[-2]) if len(sma200)>1 else False
    death  = bool(sma50.iloc[-1] < sma200.iloc[-1] and sma50.iloc[-2] >= sma200.iloc[-2]) if len(sma200)>1 else False

    # 레짐 (보수적 기준)
    # - 기본 양호: 종가 > SMA120, 정규화 변동성 <= 2.0%, 52w 고점 괴리 >= -15%
    # - 경계: 조건 중 1~2개 위반
    # - 위험회피: 대부분 위반 or Death cross
    latest_price = float(c.iloc[-1])
    above_ma = bool(latest_price > float(sma120.iloc[-1])) if sma120.notna().iloc[-1] else False
    vol_ok   = (vol_norm is not None) and (vol_norm <= 0.02)   # ~2%/day 정도를 보수적 상한으로 가정
    dd_ok    = (dd_52w is not None) and (dd_52w >= -0.15)

    violations = sum([
        0 if above_ma else 1,
        0 if vol_ok else 1,
        0 if dd_ok else 1
    ])
    if death:
        regime = "risk_off"
    elif violations == 0:
        regime = "cautious_on"
    elif violations == 1 or violations == 2:
        regime = "neutral"
    else:
        regime = "risk_off"

    metrics = {
        "price": latest_price,
        "sma120": float(sma120.iloc[-1]) if sma120.notna().iloc[-1] else None,
        "above_sma120": above_ma,
        "atr14_over_price": float(vol_norm) if vol_norm is not None else None,
        "drawdown_from_52w_high": float(dd_52w) if dd_52w is not None else None,
        "sma20_slope_last": float(slope20.iloc[-1]) if slope20.notna().iloc[-1] else None,
        "sma60_slope_last": float(slope60.iloc[-1]) if slope60.notna().iloc[-1] else None,
        "sma120_slope_last": float(slope120.iloc[-1]) if slope120.notna().iloc[-1] else None,
        "golden_cross": golden,
        "death_cross": death,
    }

    return {
        "status": "success",
        "as_of": datetime.utcnow().isoformat(),
        "base_symbol": base_symbol,
        "metrics": metrics,
        "regime": regime,  # 'risk_off' | 'neutral' | 'cautious_on'
        "period": period,
        "interval": interval,
    }
# ---------------------------------------------------------
# 2) ADK Tool 등록
# ---------------------------------------------------------
TOOLS = [
    FunctionTool(func=fetch_price),
    FunctionTool(func=fetch_fundamentals),
    FunctionTool(func=check_event_risk),
    FunctionTool(func=calc_risk),
    FunctionTool(func=fetch_fx),
    FunctionTool(func=fetch_korea_index),
    FunctionTool(func=korea_market_health),
]
