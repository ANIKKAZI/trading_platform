import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os

# Ensure the project root directory is in the system path so Python can find orchestrator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator import run

st.set_page_config(layout="wide", page_title="Stock Intelligence Platform")

st.title("📊 Quantitative Stock Intelligence Control Panel")
st.markdown("---")

# --- Sidebar Configuration ---
st.sidebar.header("Pipeline Settings")
universe_type = st.sidebar.selectbox(
    "Universe Selection",
    ["sp500", "custom", "AAPL,MSFT,NVDA,TSLA"]
)
top_n = st.sidebar.slider("Top Ranked Stocks to Return", 3, 20, 10)
force_refresh = st.sidebar.checkbox("Force Market Data Refresh", value=False)

st.sidebar.markdown("---")
st.sidebar.caption("Connected to Quantitative Engine v1.0")

# --- Run pipeline and store result in session state ---
if st.sidebar.button("⚡ Execute Quant Pipeline", use_container_width=True):
    with st.spinner("Running engine pipelines (Fetching data, calculating features, processing strategies)..."):
        try:
            output = run(
                universe=universe_type,
                force_refresh=force_refresh,
                top_n=top_n,
                save_output=True,
                verbose=False
            )
            st.session_state["output"] = output
            st.session_state["df_rank"] = pd.DataFrame(output["top_10_stocks"])
            st.session_state["df_rank"].insert(0, "rank", range(1, len(st.session_state["df_rank"]) + 1))
        except Exception as e:
            st.error(f"Engine Exception Encountered: {e}")
            st.exception(e)

# --- Render dashboard from session state (persists across dropdown changes) ---
if "df_rank" in st.session_state and not st.session_state["df_rank"].empty:
    df_rank = st.session_state["df_rank"]
    st.success("Pipeline executed successfully!")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("🏆 Strategy Leaderboard")
        st.dataframe(
            df_rank[["rank", "symbol", "score", "signal"]],
            use_container_width=True,
            hide_index=True
        )

    with col_right:
        st.subheader("📈 Technical Chart Profile & S/R Zones")
        selected_symbol = st.selectbox("Select Ticker to Chart", df_rank["symbol"].tolist())

        # Get real price data from yfinance for the selected symbol
        import yfinance as yf
        ticker_df = yf.download(selected_symbol, period="3mo", interval="1d", auto_adjust=True, progress=False)
        ticker_df.columns = [c[0] if isinstance(c, tuple) else c for c in ticker_df.columns]

        # Get actual S/R levels for selected symbol
        row = df_rank[df_rank["symbol"] == selected_symbol].iloc[0]
        support = row.get("support", [])
        resistance = row.get("resistance", [])

        fig = go.Figure()
        if not ticker_df.empty:
            fig.add_trace(go.Candlestick(
                x=ticker_df.index,
                open=ticker_df["Open"],
                high=ticker_df["High"],
                low=ticker_df["Low"],
                close=ticker_df["Close"],
                name=selected_symbol
            ))

        # Plot actual S/R zones from pipeline
        if isinstance(resistance, list) and len(resistance) >= 2:
            fig.add_hrect(y0=resistance[0], y1=resistance[1],
                          fillcolor="crimson", opacity=0.15,
                          annotation_text="Resistance", annotation_position="top left")
        elif isinstance(resistance, list) and len(resistance) == 1:
            fig.add_hline(y=resistance[0], line=dict(color="crimson", width=2, dash="dash"),
                          annotation_text="Resistance")

        if isinstance(support, list) and len(support) >= 2:
            fig.add_hrect(y0=support[0], y1=support[1],
                          fillcolor="royalblue", opacity=0.15,
                          annotation_text="Support", annotation_position="bottom left")
        elif isinstance(support, list) and len(support) == 1:
            fig.add_hline(y=support[0], line=dict(color="royalblue", width=2, dash="dash"),
                          annotation_text="Support")

        fig.update_layout(
            title=f"{selected_symbol} — Live Price Chart with S/R Zones",
            yaxis_title="Price ($)",
            xaxis_title="Date",
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)

        # Show signal details for selected stock
        st.markdown("**Signal Details**")
        detail_cols = ["symbol", "score", "signal", "strategy_score", "momentum_score"]
        available = [c for c in detail_cols if c in df_rank.columns]
        st.dataframe(df_rank[df_rank["symbol"] == selected_symbol][available], hide_index=True)

else:
    st.info("💡 Adjust setup parameters in the sidebar panel and click 'Execute Quant Pipeline' to visualize live strategy matrix data.")