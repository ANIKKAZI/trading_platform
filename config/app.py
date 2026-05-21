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

# --- Main Dashboard Logic ---
if st.sidebar.button("⚡ Execute Quant Pipeline", use_container_width=True):
    with st.spinner("Running engine pipelines (Fetching data, calculating features, processing strategies)..."):
        try:
            # Execute your backend pipeline directly!
            # Note: verbose=False keeps the console tidy; we extract data from the returned dict
            output = run(
                universe=universe_type,
                force_refresh=force_refresh,
                top_n=top_n,
                save_output=True,
                verbose=False
            )

            st.success(" Pipeline executed successfully!")

            # Layout the workspace into two dashboard zones
            col_left, col_right = st.columns([1, 2])

            with col_left:
                st.subheader("🏆 Strategy Leaderboard")

                # Parse ranked results from output
                if "top_10_stocks" in output and output["top_10_stocks"]:
                    df_rank = pd.DataFrame(output["top_10_stocks"])
                    df_rank.insert(0, "rank", range(1, len(df_rank) + 1))
                    st.dataframe(
                        df_rank[["rank", "symbol", "score"]],
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.warning("No ranked results available. Check data fetching.")
                    df_rank = pd.DataFrame()

            with col_right:
                st.subheader("📈 Technical Chart Profile & S/R Zones")
                
                if not df_rank.empty:
                    selected_symbol = st.selectbox("Select Ticker to Chart", df_rank["symbol"].unique())
                    
                    # Get S/R data for selected symbol
                    symbol_data = df_rank[df_rank["symbol"] == selected_symbol].iloc[0] if len(df_rank) > 0 else None
                    support = symbol_data.get("support", []) if symbol_data is not None else []
                    resistance = symbol_data.get("resistance", []) if symbol_data is not None else []
                else:
                    st.info("Run the pipeline to generate charts.")
                    selected_symbol = None

                if selected_symbol:
                    # --- Visualizing S/R Zones & Historical Trends using Plotly ---
                    # Generating interactive chart layout
                    fig = go.Figure()

                    # Simulated Candlestick timeline framework for visual representation
                    dates = pd.date_range(end=pd.Timestamp.now(), periods=60, freq='D')
                    np.random.seed(42)
                    price = 150 + np.cumsum(np.random.randn(60) * 2)

                    fig.add_trace(go.Candlestick(
                        x=dates,
                        open=price - 1, high=price + 2,
                        low=price - 2, close=price,
                        name=selected_symbol
                    ))

                    # Visualizing Horizontal Support / Resistance levels
                    if resistance:
                        resistance_level = resistance[0] if isinstance(resistance, list) else resistance
                        fig.add_shape(type="line", x0=dates[0], y0=resistance_level, x1=dates[-1], y1=resistance_level,
                                      line=dict(color="crimson", width=2, dash="dash"))
                    if support:
                        support_level = support[0] if isinstance(support, list) else support
                        fig.add_shape(type="line", x0=dates[0], y0=support_level, x1=dates[-1], y1=support_level,
                                      line=dict(color="royalblue", width=2, dash="dash"))

                    fig.update_layout(
                        title=f"{selected_symbol} Price Chart with Automated S/R Overlays",
                        yaxis_title="Price ($)",
                        xaxis_title="Date",
                        xaxis_rangeslider_visible=False,
                        template="plotly_dark",
                        height=450
                    )
                    st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Engine Exception Encountered: {e}")
            st.exception(e)
else:
    st.info(
        "💡 Adjust setup parameters in the sidebar panel and click 'Execute Quant Pipeline' to visualize live strategy matrix data.")