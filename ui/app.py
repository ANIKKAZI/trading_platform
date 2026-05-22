"""
Unified Stock Intelligence Platform — single Streamlit entry point.

Two tabs:
  📊 Daily Scanner   — run the quant pipeline, view Top-N leaderboard + candlestick
  🔀 Stock Comparison — side-by-side probabilistic 5-month forecast for two stocks

Launch
------
    streamlit run ui/app.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from core.data_engine import fetch_symbol
from interfaces.compare_interface import CompareInterface
from orchestrator import run as run_pipeline

# ---------------------------------------------------------------------------
# Page config  (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Stock Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
    div[data-testid="metric-container"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 10px 14px;
    }
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.80em;
        font-weight: 600;
        margin: 3px 2px;
        line-height: 1.6;
    }
    .badge-green  { background:#1b4332; color:#52b788; border:1px solid #52b788; }
    .badge-red    { background:#3b1011; color:#ef9a9a; border:1px solid #ef5350; }
    .badge-yellow { background:#2d2400; color:#ffd54f; border:1px solid #ffa726; }
    .badge-blue   { background:#0d2137; color:#64b5f6; border:1px solid #42a5f5; }
    .badge-grey   { background:#21262d; color:#8b949e; border:1px solid #30363d; }
    .winner-banner {
        background: linear-gradient(135deg, #0d2137 0%, #1b4332 100%);
        border: 1px solid #52b788;
        border-radius: 10px;
        padding: 18px 28px;
        text-align: center;
    }
    .section-title {
        font-size: 1.05em;
        font-weight: 700;
        color: #58a6ff;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .sidebar-section-header {
        font-size: 0.82em;
        font-weight: 700;
        color: #8b949e;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin: 4px 0 8px 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Constants (comparison dashboard)
# ---------------------------------------------------------------------------

_COLORS: dict[str, str] = {
    "a_primary": "#4c78d4",
    "a_light": "#93b4f5",
    "b_primary": "#f97316",
    "b_light": "#fbbf7a",
    "bull": "#26a69a",
    "bear": "#ef5350",
    "base": "#90a4ae",
    "grid": "#21262d",
}

_POPULAR_TICKERS: list[str] = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META",
    "TSLA", "JPM", "V", "AMD", "NFLX", "ORCL",
    "UNH", "GS", "XOM", "CRM", "INTC", "SPY",
]

_HORIZON_MAP: dict[str, int] = {
    "1 Month": 1,
    "3 Months": 3,
    "5 Months": 5,
    "12 Months": 12,
}

_SENSITIVITY_MULT: dict[str, dict[str, float]] = {
    "Conservative": {"bull": 0.65, "base": 0.85, "bear": 1.35},
    "Balanced":     {"bull": 1.00, "base": 1.00, "bear": 1.00},
    "Aggressive":   {"bull": 1.45, "base": 1.15, "bear": 0.65},
}

_PLOTLY_BASE: dict = {
    "template": "plotly_dark",
    "paper_bgcolor": "#0d1117",
    "plot_bgcolor": "#0d1117",
    "font": {"color": "#c9d1d9", "size": 12},
    "margin": {"l": 48, "r": 20, "t": 44, "b": 44},
}

# ---------------------------------------------------------------------------
# Cached pipeline calls
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False, ttl=3_600)
def _run_comparison(symbol_a: str, symbol_b: str, horizon: int, force_refresh: bool) -> dict:
    iface = CompareInterface(force_refresh=force_refresh)
    return iface.run(
        symbol_a=symbol_a,
        symbol_b=symbol_b,
        horizon_months=horizon,
        print_output=False,
        save_output=True,
    )


@st.cache_data(show_spinner=False, ttl=3_600)
def _fetch_history(symbol: str) -> pd.DataFrame:
    try:
        return fetch_symbol(symbol, force_refresh=False)
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Scenario sensitivity post-processing
# ---------------------------------------------------------------------------


def _adjust_scenarios(scenarios: dict, sensitivity: str) -> dict:
    m = _SENSITIVITY_MULT[sensitivity]
    result: dict = {}
    for case_key, mult_key in [
        ("bull_case", "bull"),
        ("base_case", "base"),
        ("bear_case", "bear"),
    ]:
        sc = dict(scenarios[case_key])
        orig = sc["expected_move_percent"]
        new_pct = round(orig * m[mult_key], 1)
        sc["expected_move_percent"] = new_pct
        sc["expected_move"] = f"{new_pct / 100:+.0%}"
        pr = sc.get("price_range", [])
        if len(pr) == 2:
            mid_orig = sum(pr) / 2
            cur = (mid_orig / (1 + orig / 100)) if abs(orig) > 0.01 else mid_orig
            new_mid = cur * (1 + new_pct / 100)
            half = (pr[1] - pr[0]) / 2
            sc["price_range"] = [round(max(0.01, new_mid - half), 2), round(new_mid + half, 2)]
        result[case_key] = sc
    return result


# ---------------------------------------------------------------------------
# Forecast path helpers
# ---------------------------------------------------------------------------


def _monthly_path(
    current_price: float,
    total_move_pct: float,
    horizon: int,
    ann_vol: float,
) -> tuple[list[float], list[float], list[float]]:
    if current_price <= 0:
        zeros = [0.0] * horizon
        return zeros, zeros, zeros
    prices, upper, lower = [], [], []
    for m in range(1, horizon + 1):
        fraction = m / max(horizon, 1)
        price = current_price * (1.0 + total_move_pct / 100.0 * fraction)
        sigma = (ann_vol / np.sqrt(12)) * np.sqrt(m) * current_price
        prices.append(round(price, 2))
        upper.append(round(price + sigma, 2))
        lower.append(round(max(0.01, price - sigma), 2))
    return prices, upper, lower


def _return_distribution(
    base_pct: float, ann_vol: float, horizon: int, n_points: int = 300
) -> tuple[np.ndarray, np.ndarray]:
    horizon_std = max(0.01, ann_vol * np.sqrt(max(1, horizon) / 12)) * 100
    x = np.linspace(base_pct - 4 * horizon_std, base_pct + 4 * horizon_std, n_points)
    y = (1 / (horizon_std * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((x - base_pct) / horizon_std) ** 2
    )
    return x, y


def _hex_to_rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b}"


# ---------------------------------------------------------------------------
# Comparison charts
# ---------------------------------------------------------------------------


def _chart_projection(
    sym_a: str,
    sym_b: str,
    hist_a: pd.DataFrame,
    hist_b: pd.DataFrame,
    fc_a: dict,
    fc_b: dict,
    horizon: int,
    sensitivity: str,
) -> go.Figure:
    sc_a = _adjust_scenarios(fc_a["scenarios"], sensitivity)
    sc_b = _adjust_scenarios(fc_b["scenarios"], sensitivity)
    cur_a = fc_a["current_price"]
    cur_b = fc_b["current_price"]

    def _norm_history(df: pd.DataFrame) -> tuple[list, list]:
        if df.empty:
            return [], []
        tail = df.tail(252)
        base = float(tail["Close"].iloc[0])
        if base == 0:
            return [], []
        return tail.index.tolist(), ((tail["Close"] / base - 1) * 100).tolist()

    idx_a, pct_a = _norm_history(hist_a)
    idx_b, pct_b = _norm_history(hist_b)

    last_date = idx_a[-1] if idx_a else pd.Timestamp.today()
    hist_base_a = float(hist_a.tail(252)["Close"].iloc[0]) if not hist_a.empty else cur_a
    hist_base_b = float(hist_b.tail(252)["Close"].iloc[0]) if not hist_b.empty else cur_b

    future_dates = [last_date + pd.DateOffset(months=m) for m in range(0, horizon + 1)]

    def _forecast_pcts(sc: dict, cur_price: float, hist_base: float) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for case_key in ("bull_case", "base_case", "bear_case"):
            total_pct = sc[case_key]["expected_move_percent"]
            pts = [((cur_price / hist_base) - 1) * 100]
            for m in range(1, horizon + 1):
                interim = total_pct * (m / horizon)
                pts.append(round(((cur_price * (1 + interim / 100)) / hist_base - 1) * 100, 2))
            out[case_key] = pts
        return out

    fp_a = _forecast_pcts(sc_a, cur_a, hist_base_a)
    fp_b = _forecast_pcts(sc_b, cur_b, hist_base_b)

    fig = go.Figure()

    if idx_a:
        fig.add_trace(go.Scatter(
            x=idx_a, y=pct_a, name=f"{sym_a} (historical)",
            line=dict(color=_COLORS["a_primary"], width=2), mode="lines",
        ))
    if idx_b:
        fig.add_trace(go.Scatter(
            x=idx_b, y=pct_b, name=f"{sym_b} (historical)",
            line=dict(color=_COLORS["b_primary"], width=2), mode="lines",
        ))

    for sym, fp, color, light in [
        (sym_a, fp_a, _COLORS["a_primary"], _COLORS["a_light"]),
        (sym_b, fp_b, _COLORS["b_primary"], _COLORS["b_light"]),
    ]:
        fig.add_trace(go.Scatter(
            x=future_dates + future_dates[::-1],
            y=fp["bull_case"] + fp["bear_case"][::-1],
            fill="toself",
            fillcolor=f"rgba({_hex_to_rgb(color)},0.13)",
            line=dict(width=0), name=f"{sym} confidence band",
            showlegend=False, hoverinfo="skip",
        ))
        for case_key, dash, show_leg in [
            ("bull_case", "dot", False),
            ("base_case", "dash", True),
            ("bear_case", "dot", False),
        ]:
            label = case_key.replace("_case", "").capitalize()
            fig.add_trace(go.Scatter(
                x=future_dates, y=fp[case_key],
                name=f"{sym} {label}",
                line=dict(
                    color=light if case_key != "base_case" else color,
                    width=1.5 if case_key != "base_case" else 2.2,
                    dash=dash,
                ),
                mode="lines", showlegend=show_leg,
                hovertemplate=f"{sym} {label}: %{{y:.1f}}%<extra></extra>",
            ))

    # Use add_shape + add_annotation to avoid Plotly shapeannotation._mean() bug
    _today_x = last_date.isoformat()
    fig.add_shape(
        type="line", x0=_today_x, x1=_today_x, y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color="#555", dash="dot", width=1.5),
    )
    fig.add_annotation(
        x=_today_x, y=1, xref="x", yref="paper",
        text="Today", showarrow=False, xanchor="left",
        font=dict(color="#888", size=11),
    )
    fig.update_layout(
        **_PLOTLY_BASE,
        title=f"Cumulative Return: Historical (252d) + {horizon}-Month Projection",
        xaxis_title="Date", yaxis_title="Cumulative Return (%)",
        hovermode="x unified", height=520,
        legend=dict(orientation="h", y=-0.18, font=dict(size=11)),
    )
    return fig


def _chart_risk_reward(
    sym_a: str, sym_b: str, fc_a: dict, fc_b: dict, sc_a: dict, sc_b: dict,
) -> go.Figure:
    def _risk_score(fc: dict) -> float:
        vol = fc["market_context"]["volatility"]
        beta = abs(fc["market_context"].get("beta", 1.0))
        n_risks = len([r for r in fc.get("risk_factors", []) if "No significant" not in r])
        return round(vol * 40 + min(beta, 3) * 5 + n_risks * 3, 1)

    fig = go.Figure()
    for sym, fc, sc, color in [
        (sym_a, fc_a, sc_a, _COLORS["a_primary"]),
        (sym_b, fc_b, sc_b, _COLORS["b_primary"]),
    ]:
        base_ret = sc["base_case"]["expected_move_percent"]
        bull_ret = sc["bull_case"]["expected_move_percent"]
        bear_ret = sc["bear_case"]["expected_move_percent"]
        risk = _risk_score(fc)
        vol_pct = fc["market_context"]["volatility"] * 100
        fig.add_trace(go.Scatter(
            x=[risk], y=[base_ret], mode="markers+text", name=sym,
            text=[sym], textposition="top center",
            textfont=dict(size=13, color=color),
            marker=dict(size=22, color=color, line=dict(color="white", width=2)),
            customdata=[[vol_pct, bull_ret, bear_ret]],
            hovertemplate=(
                f"<b>{sym}</b><br>Risk Score: %{{x:.1f}}<br>"
                "Base Return: %{y:+.1f}%<br>Volatility: %{customdata[0]:.1f}%<br>"
                "Bull: %{customdata[1]:+.1f}%<br>Bear: %{customdata[2]:+.1f}%<extra></extra>"
            ),
        ))
    fig.add_hline(y=0, line=dict(color="#444", dash="dot", width=1))
    fig.update_layout(
        **_PLOTLY_BASE,
        title="Risk vs Expected Return",
        xaxis_title="Risk Score  (higher = riskier)",
        yaxis_title="Base Expected Return (%)",
        height=440,
    )
    return fig


def _chart_radar(sym_a: str, sym_b: str, result: dict) -> go.Figure:
    strat_a = result.get("strategy_signals", {}).get(sym_a, {})
    strat_b = result.get("strategy_signals", {}).get(sym_b, {})
    ml_a = float(result.get("ml_outputs", {}).get(sym_a, 0.0))
    ml_b = float(result.get("ml_outputs", {}).get(sym_b, 0.0))

    strategies = ["MomentumStrategy", "MeanReversionStrategy", "BreakoutStrategy", "VolatilityStrategy"]
    labels = ["Momentum", "Mean Reversion", "Breakout", "Volatility", "ML Confidence"]

    def _score(raw: dict, confs: dict, key: str) -> float:
        return round(0.5 + raw.get(key, 0) * confs.get(key, 0.5) * 0.5, 3)

    fig = go.Figure()
    for sym, strat, ml, color in [
        (sym_a, strat_a, ml_a, _COLORS["a_primary"]),
        (sym_b, strat_b, ml_b, _COLORS["b_primary"]),
    ]:
        raw = strat.get("raw_signals", {})
        confs = strat.get("confidences", {})
        vals = [_score(raw, confs, s) for s in strategies] + [round((ml + 1) / 2, 3)]
        closed_vals = vals + [vals[0]]
        closed_labels = labels + [labels[0]]
        fig.add_trace(go.Scatterpolar(
            r=closed_vals, theta=closed_labels, fill="toself", name=sym,
            line=dict(color=color, width=2.2),
            fillcolor=f"rgba({_hex_to_rgb(color)},0.22)",
        ))
    fig.update_layout(
        **_PLOTLY_BASE,
        title="Strategy Signal Radar  (0 = bearish · 0.5 = neutral · 1 = bullish)",
        polar=dict(
            bgcolor="#0d1117",
            radialaxis=dict(
                visible=True, range=[0, 1],
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                ticktext=["0", "0.25", "0.5", "0.75", "1.0"],
                tickfont=dict(size=9), gridcolor=_COLORS["grid"],
            ),
            angularaxis=dict(gridcolor=_COLORS["grid"]),
        ),
        height=460,
    )
    return fig


def _chart_sr_single(sym: str, fc: dict, sr: dict) -> go.Figure:
    cur = fc["current_price"]
    support_zones = sr.get("support", [])
    resistance_zones = sr.get("resistance", [])

    all_levels = [cur]
    for z in support_zones:
        all_levels.extend(z)
    for z in resistance_zones:
        all_levels.extend(z)

    y_min = min(all_levels) * 0.965
    y_max = max(all_levels) * 1.035

    fig = go.Figure()
    for i, zone in enumerate(support_zones[:4]):
        lo, hi = float(zone[0]), float(zone[1])
        fig.add_hrect(
            y0=lo, y1=hi, fillcolor="rgba(38,166,154,0.28)",
            line=dict(color="#26a69a", width=1),
            annotation_text=f"  S{i+1}: {lo:.0f}–{hi:.0f}",
            annotation_position="right",
            annotation_font=dict(color="#26a69a", size=10),
        )
    for i, zone in enumerate(resistance_zones[:4]):
        lo, hi = float(zone[0]), float(zone[1])
        fig.add_hrect(
            y0=lo, y1=hi, fillcolor="rgba(239,83,80,0.28)",
            line=dict(color="#ef5350", width=1),
            annotation_text=f"  R{i+1}: {lo:.0f}–{hi:.0f}",
            annotation_position="right",
            annotation_font=dict(color="#ef5350", size=10),
        )
    fig.add_hline(
        y=cur, line=dict(color="white", width=2.5),
        annotation_text=f"  ${cur:.2f}",
        annotation_position="right",
        annotation_font=dict(color="white", size=12, family="monospace"),
    )
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[y_min, y_max], mode="markers",
        marker=dict(opacity=0, size=1), showlegend=False, hoverinfo="skip",
    ))
    fig.update_layout(
        **_PLOTLY_BASE,
        title=f"{sym} — Support & Resistance Zones",
        yaxis=dict(range=[y_min, y_max], gridcolor=_COLORS["grid"], title="Price ($)"),
        xaxis=dict(visible=False, range=[0, 1]),
        height=420, showlegend=False,
    )
    return fig


def _chart_monthly(
    sym_a: str, sym_b: str, fc_a: dict, fc_b: dict, horizon: int, sensitivity: str,
) -> go.Figure:
    sc_a = _adjust_scenarios(fc_a["scenarios"], sensitivity)
    sc_b = _adjust_scenarios(fc_b["scenarios"], sensitivity)
    month_labels = ["Now"] + [f"M+{m}" for m in range(1, horizon + 1)]
    fig = go.Figure()
    for sym, fc, sc, color, light in [
        (sym_a, fc_a, sc_a, _COLORS["a_primary"], _COLORS["a_light"]),
        (sym_b, fc_b, sc_b, _COLORS["b_primary"], _COLORS["b_light"]),
    ]:
        cur = fc["current_price"]
        vol = fc["market_context"]["volatility"]
        base_pts, upper, lower = _monthly_path(cur, sc["base_case"]["expected_move_percent"], horizon, vol)
        bull_pts, _, _ = _monthly_path(cur, sc["bull_case"]["expected_move_percent"], horizon, vol)
        bear_pts, _, _ = _monthly_path(cur, sc["bear_case"]["expected_move_percent"], horizon, vol)
        base_all = [cur] + base_pts
        upper_all = [cur] + upper
        lower_all = [cur] + lower
        fig.add_trace(go.Scatter(
            x=month_labels + month_labels[::-1],
            y=upper_all + lower_all[::-1],
            fill="toself", fillcolor=f"rgba({_hex_to_rgb(color)},0.14)",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        for pts, sfx in [([cur] + bull_pts, "Bull"), ([cur] + bear_pts, "Bear")]:
            fig.add_trace(go.Scatter(
                x=month_labels, y=pts, name=f"{sym} {sfx}",
                line=dict(color=light, width=1.2, dash="dot"),
                mode="lines", showlegend=False,
                hovertemplate=f"{sym} {sfx}: $%{{y:.2f}}<extra></extra>",
            ))
        fig.add_trace(go.Scatter(
            x=month_labels, y=base_all, name=sym,
            line=dict(color=color, width=2.5),
            mode="lines+markers", marker=dict(size=7),
            hovertemplate=f"<b>{sym}</b>  Month: %{{x}}<br>Price: $%{{y:.2f}}<extra></extra>",
        ))
    fig.update_layout(
        **_PLOTLY_BASE,
        title="Month-by-Month Price Projection  (Base ± 1σ band, Bull/Bear bounds)",
        xaxis_title="Month", yaxis_title="Projected Price ($)",
        hovermode="x unified", height=460,
        legend=dict(orientation="h", y=-0.18),
    )
    return fig


def _chart_distribution(
    sym_a: str, sym_b: str, fc_a: dict, fc_b: dict, horizon: int, sensitivity: str,
) -> go.Figure:
    sc_a = _adjust_scenarios(fc_a["scenarios"], sensitivity)
    sc_b = _adjust_scenarios(fc_b["scenarios"], sensitivity)
    fig = go.Figure()
    for sym, fc, sc, color in [
        (sym_a, fc_a, sc_a, _COLORS["a_primary"]),
        (sym_b, fc_b, sc_b, _COLORS["b_primary"]),
    ]:
        base_pct = sc["base_case"]["expected_move_percent"]
        vol = fc["market_context"]["volatility"]
        x, y = _return_distribution(base_pct, vol, horizon)
        mask_pos = x >= 0
        if mask_pos.any():
            fig.add_trace(go.Scatter(
                x=x[mask_pos], y=y[mask_pos], fill="tozeroy",
                fillcolor=f"rgba({_hex_to_rgb(color)},0.10)",
                line=dict(width=0), showlegend=False, hoverinfo="skip",
            ))
        fig.add_trace(go.Scatter(
            x=x, y=y, name=sym, line=dict(color=color, width=2.5), mode="lines",
            hovertemplate=f"<b>{sym}</b><br>Return: %{{x:.1f}}%<br>Density: %{{y:.4f}}<extra></extra>",
        ))
        idx_peak = int(np.argmax(y))
        fig.add_annotation(
            x=float(x[idx_peak]), y=float(y[idx_peak]),
            text=f"{sym}: {base_pct:+.1f}%", showarrow=True,
            arrowhead=2, arrowcolor=color,
            font=dict(color=color, size=11), ay=-32,
        )
    fig.add_vline(x=0, line=dict(color="#555", dash="dot", width=1.5))
    fig.add_annotation(
        x=0, y=0, xref="x", yref="paper", text="0%",
        showarrow=False, font=dict(color="#888", size=10), yanchor="bottom",
    )
    fig.update_layout(
        **_PLOTLY_BASE,
        title=f"Return Probability Distribution  ({horizon}-Month Horizon)",
        xaxis_title="Expected Return (%)", yaxis_title="Probability Density",
        hovermode="x unified", height=420,
    )
    return fig


# ---------------------------------------------------------------------------
# Comparison UI helpers
# ---------------------------------------------------------------------------


def _metric_cards(sym: str, fc: dict, sc: dict, strat: dict) -> None:
    cur = fc["current_price"]
    base_ret = sc["base_case"]["expected_move_percent"]
    bull_ret = sc["bull_case"]["expected_move_percent"]
    bear_ret = sc["bear_case"]["expected_move_percent"]
    vol_pct = fc["market_context"]["volatility"] * 100
    signal = strat.get("signal_label", "NEUTRAL")
    score = strat.get("final_score", 0.0)
    confidence = round((score + 100) / 2)

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Price", f"${cur:.2f}")
    c2.metric("Base Return", f"{base_ret:+.1f}%",
              delta=f"{'▲' if base_ret >= 0 else '▼'} Expected",
              delta_color="normal" if base_ret >= 0 else "inverse")
    c3.metric("Bull Case", f"{bull_ret:+.1f}%", delta="Upside", delta_color="normal")
    c4.metric("Bear Case", f"{bear_ret:+.1f}%", delta="Downside", delta_color="inverse")
    c5.metric("Volatility", f"{vol_pct:.1f}%",
              delta="Low" if vol_pct < 25 else ("Mod" if vol_pct < 40 else "High"),
              delta_color="normal" if vol_pct < 25 else ("off" if vol_pct < 40 else "inverse"))
    c6.metric("Signal", signal)
    c7.metric("Confidence", f"{confidence}%")


def _insights_badges(sym: str, fc: dict, strat: dict) -> None:
    raw_sigs = strat.get("raw_signals", {})
    confs = strat.get("confidences", {})
    risks = fc.get("risk_factors", [])
    td = fc.get("market_context", {}).get("trend_direction", "mixed")

    html_parts: list[str] = []
    for name, sig in raw_sigs.items():
        conf = confs.get(name, 0.5)
        short = name.replace("Strategy", "")
        if sig == 1:
            cls, icon = "badge-green", "✓"
        elif sig == -1:
            cls, icon = "badge-red", "✗"
        else:
            cls, icon = "badge-grey", "~"
        html_parts.append(f'<span class="badge {cls}">{icon} {short} ({conf:.0%})</span>')
    td_label = td.replace("_", " ").title()
    td_cls = "badge-green" if "up" in td else ("badge-red" if "down" in td else "badge-blue")
    html_parts.append(f'<span class="badge {td_cls}">📈 {td_label}</span>')
    for r in risks:
        if "No significant" in r:
            continue
        snippet = r[:55] + "…" if len(r) > 55 else r
        html_parts.append(f'<span class="badge badge-yellow">⚠ {snippet}</span>')

    st.markdown(
        f'<div class="section-title">{sym}</div>' + " ".join(html_parts),
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------


def _render_scanner_tab(universe_type: str, top_n: int, force_refresh_scan: bool, run_scan: bool) -> None:
    """Renders the Daily Scanner tab content."""
    st.subheader("📊 Quantitative Stock Intelligence — Daily Scanner")

    if run_scan:
        with st.spinner("Running engine pipelines (fetching data, calculating features, processing strategies)…"):
            try:
                output = run_pipeline(
                    universe=universe_type,
                    force_refresh=force_refresh_scan,
                    top_n=top_n,
                    save_output=True,
                    verbose=False,
                )
                st.session_state["scan_output"] = output
                df = pd.DataFrame(output["top_10_stocks"])
                df.insert(0, "rank", range(1, len(df) + 1))
                st.session_state["scan_df"] = df
            except Exception as e:
                st.error(f"Engine Exception: {e}")
                st.exception(e)

    if "scan_df" in st.session_state and not st.session_state["scan_df"].empty:
        df_rank = st.session_state["scan_df"]
        st.success("Pipeline executed successfully!")

        col_left, col_right = st.columns([1, 2])

        with col_left:
            st.subheader("🏆 Strategy Leaderboard")
            st.dataframe(
                df_rank[["rank", "symbol", "score", "signal"]],
                use_container_width=True,
                hide_index=True,
            )

        with col_right:
            st.subheader("📈 Technical Chart Profile & S/R Zones")
            selected_symbol = st.selectbox(
                "Select Ticker to Chart",
                df_rank["symbol"].tolist(),
                key="scan_selectbox",
            )

            ticker_df = yf.download(
                selected_symbol, period="3mo", interval="1d",
                auto_adjust=True, progress=False,
            )
            ticker_df.columns = [
                c[0] if isinstance(c, tuple) else c for c in ticker_df.columns
            ]

            row = df_rank[df_rank["symbol"] == selected_symbol].iloc[0]
            support = row.get("support", [])
            resistance = row.get("resistance", [])

            scan_fig = go.Figure()
            if not ticker_df.empty:
                scan_fig.add_trace(go.Candlestick(
                    x=ticker_df.index,
                    open=ticker_df["Open"],
                    high=ticker_df["High"],
                    low=ticker_df["Low"],
                    close=ticker_df["Close"],
                    name=selected_symbol,
                ))

            if isinstance(resistance, list) and len(resistance) >= 2:
                scan_fig.add_hrect(
                    y0=resistance[0], y1=resistance[1],
                    fillcolor="crimson", opacity=0.15,
                    annotation_text="Resistance", annotation_position="top left",
                )
            elif isinstance(resistance, list) and len(resistance) == 1:
                scan_fig.add_hline(
                    y=resistance[0], line=dict(color="crimson", width=2, dash="dash"),
                    annotation_text="Resistance",
                )

            if isinstance(support, list) and len(support) >= 2:
                scan_fig.add_hrect(
                    y0=support[0], y1=support[1],
                    fillcolor="royalblue", opacity=0.15,
                    annotation_text="Support", annotation_position="bottom left",
                )
            elif isinstance(support, list) and len(support) == 1:
                scan_fig.add_hline(
                    y=support[0], line=dict(color="royalblue", width=2, dash="dash"),
                    annotation_text="Support",
                )

            scan_fig.update_layout(
                title=f"{selected_symbol} — Live Price Chart with S/R Zones",
                yaxis_title="Price ($)",
                xaxis_title="Date",
                xaxis_rangeslider_visible=False,
                template="plotly_dark",
                paper_bgcolor="#0d1117",
                height=450,
            )
            st.plotly_chart(scan_fig, use_container_width=True)

            st.markdown("**Signal Details**")
            detail_cols = ["symbol", "score", "signal", "strategy_score", "momentum_score"]
            available = [c for c in detail_cols if c in df_rank.columns]
            st.dataframe(
                df_rank[df_rank["symbol"] == selected_symbol][available],
                hide_index=True,
            )
    else:
        st.info("👈  Configure Scanner Settings in the sidebar and click **⚡ Execute Quant Pipeline** to begin.")


def _render_compare_tab(
    sym_a: str,
    sym_b: str,
    horizon_label: str,
    horizon: int,
    sensitivity: str,
    force_refresh_cmp: bool,
    run_compare: bool,
) -> None:
    """Renders the Stock Comparison tab content."""
    st.subheader("🔀 Stock Comparison Forecast Dashboard")
    st.caption(
        "Probabilistic multi-scenario projections powered by the "
        "Quantitative Stock Intelligence Platform"
    )

    if run_compare:
        if not sym_a or not sym_b:
            st.error("Please enter both ticker symbols.")
            return
        if sym_a == sym_b:
            st.error("Stock A and Stock B must be different.")
            return
        with st.spinner(f"Running pipeline: {sym_a} vs {sym_b}…"):
            try:
                result = _run_comparison(sym_a, sym_b, horizon, force_refresh_cmp)
                st.session_state.update(
                    cmp_result=result,
                    cmp_result_sym_a=sym_a,
                    cmp_result_sym_b=sym_b,
                    cmp_horizon=horizon,
                )
            except Exception as exc:
                st.error(f"Pipeline error: {exc}")
                st.exception(exc)
                return

    if "cmp_result" not in st.session_state:
        st.info("👈  Configure Comparison Setup in the sidebar and click **⚡ Run Comparison** to begin.")
        return

    result: dict = st.session_state["cmp_result"]
    sym_a = st.session_state.get("cmp_result_sym_a", sym_a)
    sym_b = st.session_state.get("cmp_result_sym_b", sym_b)
    horizon = st.session_state.get("cmp_horizon", horizon)

    fc_a = result["detailed_forecast"][sym_a]
    fc_b = result["detailed_forecast"][sym_b]
    strat_a = result.get("strategy_signals", {}).get(sym_a, {})
    strat_b = result.get("strategy_signals", {}).get(sym_b, {})
    sc_a = _adjust_scenarios(fc_a["scenarios"], sensitivity)
    sc_b = _adjust_scenarios(fc_b["scenarios"], sensitivity)

    with st.spinner("Loading price history…"):
        hist_a = _fetch_history(sym_a)
        hist_b = _fetch_history(sym_b)

    # Winner banner
    winner = result["winner"]
    confidence = result["confidence"]
    w_color = _COLORS["a_primary"] if winner == sym_a else _COLORS["b_primary"]
    st.markdown(
        f'<div class="winner-banner">'
        f'<h3 style="color:{w_color}; margin:0 0 6px 0;">🏆 {winner} leads</h3>'
        f'<span style="color:#c9d1d9;">Confidence: <b>{confidence}</b>'
        f'&ensp;|&ensp;Horizon: <b>{horizon_label}</b>'
        f'&ensp;|&ensp;Sensitivity: <b>{sensitivity}</b></span>'
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(f'<div class="section-title">📋 {sym_a}</div>', unsafe_allow_html=True)
    _metric_cards(sym_a, fc_a, sc_a, strat_a)

    st.markdown(f'<div class="section-title">📋 {sym_b}</div>', unsafe_allow_html=True)
    _metric_cards(sym_b, fc_b, sc_b, strat_b)

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Projection", "🎯 Risk / Reward", "🕸️ Strategy Radar",
        "🟩 S/R Zones", "📅 Monthly Trend", "📊 Distribution",
    ])

    with tab1:
        st.plotly_chart(
            _chart_projection(sym_a, sym_b, hist_a, hist_b, fc_a, fc_b, horizon, sensitivity),
            use_container_width=True,
        )
    with tab2:
        st.plotly_chart(
            _chart_risk_reward(sym_a, sym_b, fc_a, fc_b, sc_a, sc_b),
            use_container_width=True,
        )
    with tab3:
        st.plotly_chart(_chart_radar(sym_a, sym_b, result), use_container_width=True)
    with tab4:
        sr_data = result.get("sr_zone_data", {})
        col_sr1, col_sr2 = st.columns(2)
        with col_sr1:
            st.plotly_chart(
                _chart_sr_single(sym_a, fc_a, sr_data.get(sym_a, {})),
                use_container_width=True,
            )
        with col_sr2:
            st.plotly_chart(
                _chart_sr_single(sym_b, fc_b, sr_data.get(sym_b, {})),
                use_container_width=True,
            )
    with tab5:
        st.plotly_chart(
            _chart_monthly(sym_a, sym_b, fc_a, fc_b, horizon, sensitivity),
            use_container_width=True,
        )
    with tab6:
        st.plotly_chart(
            _chart_distribution(sym_a, sym_b, fc_a, fc_b, horizon, sensitivity),
            use_container_width=True,
        )

    st.markdown("---")
    st.subheader("🔑 Key Factors")
    key_factors = result.get("key_factors", [])
    if key_factors:
        cols_kf = st.columns(min(len(key_factors), 3))
        for i, factor in enumerate(key_factors):
            cols_kf[i % len(cols_kf)].success(f"✓  {factor}")

    st.markdown("---")
    st.subheader("💡 Signal Intelligence")
    ins_a, ins_b = st.columns(2)
    with ins_a:
        _insights_badges(sym_a, fc_a, strat_a)
    with ins_b:
        _insights_badges(sym_b, fc_b, strat_b)

    with st.expander("📊 Full Comparison Scorecard"):
        for item in result.get("comparison_summary", []):
            st.markdown(f"- {item}")


# ---------------------------------------------------------------------------
# Main — unified entry point
# ---------------------------------------------------------------------------


def main() -> None:
    # Top-level page tabs
    tab_scan, tab_compare = st.tabs(["📊 Daily Scanner", "🔀 Stock Comparison"])

    # ── Shared sidebar ─────────────────────────────────────────────────────────
    with st.sidebar:
        # ── Scanner section ───────────────────────────────────────────────────
        st.markdown('<div class="sidebar-section-header">📊 Scanner Settings</div>', unsafe_allow_html=True)

        universe_type = st.selectbox(
            "Universe Selection",
            ["sp500", "custom", "AAPL,MSFT,NVDA,TSLA"],
            key="scan_universe",
        )
        top_n = st.slider("Top Ranked Stocks", 3, 20, 10, key="scan_top_n")
        force_refresh_scan = st.checkbox("Force Data Refresh (Scanner)", value=False, key="scan_refresh")
        st.caption("Connected to Quantitative Engine v1.0")
        run_scan_btn = st.button(
            "⚡ Execute Quant Pipeline", use_container_width=True, key="run_scan"
        )

        st.markdown("---")

        # ── Comparison section ────────────────────────────────────────────────
        st.markdown('<div class="sidebar-section-header">🔀 Comparison Setup</div>', unsafe_allow_html=True)

        sym_a = st.text_input("Stock A", value="NVDA", max_chars=10, key="cmp_sym_a").upper().strip()
        sym_b = st.text_input("Stock B", value="AAPL", max_chars=10, key="cmp_sym_b").upper().strip()

        with st.expander("🔍 Quick-pick tickers", expanded=False):
            qp_cols = st.columns(3)
            for i, t in enumerate(_POPULAR_TICKERS):
                if qp_cols[i % 3].button(t, key=f"qt_{t}", use_container_width=True):
                    st.session_state["_qpick"] = t
            if "_qpick" in st.session_state:
                picked = st.session_state.pop("_qpick")
                st.info(f"Copy **{picked}** into Stock A or B above.")

        horizon_label = st.radio(
            "Forecast Horizon",
            list(_HORIZON_MAP.keys()),
            index=2,
            key="cmp_horizon_label",
        )
        horizon = _HORIZON_MAP[horizon_label]

        sensitivity = st.radio(
            "Scenario Sensitivity",
            ["Conservative", "Balanced", "Aggressive"],
            index=1,
            key="cmp_sensitivity",
            help=(
                "**Conservative**: wider bear / tighter bull.\n\n"
                "**Aggressive**: wider bull / tighter bear."
            ),
        )

        force_refresh_cmp = st.checkbox("Force Data Refresh (Compare)", value=False, key="cmp_refresh")
        run_compare_btn = st.button(
            "⚡ Run Comparison", use_container_width=True, key="run_compare", type="primary"
        )

    # ── Tab content ────────────────────────────────────────────────────────────
    with tab_scan:
        _render_scanner_tab(universe_type, top_n, force_refresh_scan, run_scan_btn)

    with tab_compare:
        _render_compare_tab(
            sym_a, sym_b, horizon_label, horizon, sensitivity,
            force_refresh_cmp, run_compare_btn,
        )


if __name__ == "__main__":
    main()
