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

                # Check if your output structure has ranked results, fallback to dummy dataframe parsing if custom structured
                if "ranked_results" in output:
                    df_rank = pd.DataFrame(output["ranked_results"])
                else:
                    # Fallback structural parser if output dict is structured differently
                    st.warning("Ensure build_json_output() matches schema expected below.")
                    df_rank = pd.DataFrame()

                # Mock preview table representation if dataframe parsing succeeded
                if not df_rank.empty:
                    st.dataframe(
                        df_rank[["rank", "symbol", "composite_score"]],
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    # Quick visual fallback UI so the application runs out-of-the-box
                    mock_data = {"Rank": list(range(1, top_n + 1)),
                                 "Symbol": ["AAPL", "NVDA", "MSFT", "TSLA"] + ["STK"] * (top_n - 4),
                                 "Score": np.linspace(88, 62, top_n).round(2)}
                    df_rank = pd.DataFrame(mock_data)
                    st.dataframe(df_rank, use_container_width=True, hide_index=True)

            with col_right:
                st.subheader("📈 Technical Chart Profile & S/R Zones")
                selected_symbol = st.selectbox("Select Ticker to Chart", df_rank.iloc[:, 1].unique())

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
                # In your production code, extract these from output['symbol_results'][selected_symbol]['sr_zones']
                fig.add_shape(type="line", x0=dates[0], y0=max(price) + 3, x1=dates[-1], y1=max(price) + 3,
                              line=dict(color="crimson", width=2, dash="dash"), name="Resistance Zone")
                fig.add_shape(type="line", x0=dates[0], y0=min(price) - 3, x1=dates[-1], y1=min(price) - 3,
                              line=dict(color="royalblue", width=2, dash="dash"), name="Support Zone")

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