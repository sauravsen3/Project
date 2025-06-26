#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 20:40:56 2025

@author: sauravsen
"""

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf # Import yfinance

# --- Data Loading from Yahoo Finance ---
@st.cache_data # Cache data loading for performance in Streamlit
def load_yahoo_data(ticker_symbol, period='1y', interval='1d'):
    """
    Fetches historical OHLCV data from Yahoo Finance for the given ticker.
    Adjust period and interval as needed.
    """
    try:
        # yfinance often uses a '.FX' suffix for currency pairs on some platforms,
        # but for direct FX data (like EURUSD=X), it's typically direct.
        # However, for plotting purposes, we'll simplify to common stock/index symbols
        # or common FX symbols if they work directly.
        # For true FX, you might need 'EURUSD=X', 'GBPUSD=X', 'JPY=X', etc.
        # For simplicity and broader data availability on Yahoo, I'll use index proxies like SPY, QQQ.
        # You can replace with 'EURUSD=X' if you verify it works consistently for your needs.
        
        # Mapping for display vs actual Yahoo Finance symbol
        yf_ticker_map = {
            'EUR/USD': 'EURUSD=X', # Common FX format
            'GBP/USD': 'GBPUSD=X',
            'SPY (S&P 500 ETF)': 'SPY', # Example stock/ETF
            'QQQ (Nasdaq 100 ETF)': 'QQQ' # Example stock/ETF
        }
        
        actual_ticker = yf_ticker_map.get(ticker_symbol, ticker_symbol)
        
        data = yf.download(actual_ticker, period=period, interval=interval)
        if data.empty:
            st.error(f"No data found for {actual_ticker}. Please check the ticker symbol or try a different period/interval.")
            return None
        
        # Calculate some basic daily features
        data['Daily_Return'] = data['Adj Close'].pct_change()
        data['Daily_Volatility'] = data['Daily_Return'].rolling(window=5).std() # 5-day rolling volatility
        data['Volume_Change'] = data['Volume'].pct_change()

        # For "microstructure-like" features from OHLCV, we simulate simple directional pressure.
        # These are *not* true OBI/OrderFlow but simple proxies.
        data['Simulated_Directional_Pressure'] = (data['Close'] - data['Open']) / data['Open'] # Close-to-Open return as pressure proxy
        
        # Drop NaN values created by pct_change and rolling windows
        data = data.dropna()
        
        # Prepare correlation matrix for selected features
        features_for_corr = data[['Daily_Return', 'Daily_Volatility', 'Volume_Change', 'Simulated_Directional_Pressure']]
        corr_matrix = features_for_corr.corr()
        
        # Simulate "binned impact" for a generic feature vs. return
        # In a real scenario, this would be from a more complex analysis
        # Here, we'll just bin 'Simulated_Directional_Pressure' and see average returns
        bins = pd.cut(data['Simulated_Directional_Pressure'], bins=5, labels=False)
        binned_impact = data.groupby(bins)['Daily_Return'].mean().reset_index()
        binned_impact.columns = ['Pressure_Bin', 'Avg_Future_Impact']
        
        return {
            'corr_matrix': corr_matrix,
            'raw_data_sample': data.tail(100).copy(), # Take a sample for plotting
            'binned_feature_impact': binned_impact
        }
    except Exception as e:
        st.error(f"Error fetching data from Yahoo Finance for {ticker_symbol}: {e}")
        return None

# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Market Data Insights Dashboard")

st.title("Market Data Insights Dashboard (Powered by Yahoo Finance)")
st.markdown(
    """
    This interactive dashboard explores relationships between daily market data features and returns.
    **Note:** Data is sourced from Yahoo Finance (OHLCV), thus real-time high-frequency order book
    features like Order Book Imbalance (OBI) are not available here.
    """
)

# --- Sidebar for Inputs ---
st.sidebar.header("Configuration")
selected_ticker = st.sidebar.selectbox(
    "Select Asset:",
    ('EUR/USD', 'GBP/USD', 'SPY (S&P 500 ETF)', 'QQQ (Nasdaq 100 ETF)') # Added common ETFs as alternatives
)

data_period = st.sidebar.selectbox(
    "Data Period:",
    ('1y', '6mo', '3mo', '1mo'), # Common periods for daily data
    index=0
)

st.sidebar.markdown("---") # Separator
risk_appetite = st.sidebar.select_slider(
    "Select Strategy Risk Profile:",
    options=['Conservative', 'Moderate', 'Aggressive'],
    value='Moderate'
)
st.sidebar.info(
    "Risk Profile adjusts how signals are interpreted based on feature correlations and historical patterns."
)

# --- Main Content Area ---
st.header(f"Analysis for **{selected_ticker}**")
st.markdown("---")

data_store = load_yahoo_data(selected_ticker, period=data_period)

if data_store:
    corr_matrix = data_store['corr_matrix']
    raw_data_sample = data_store['raw_data_sample']
    binned_feature_impact = data_store['binned_feature_impact']

    # --- Tabbed Interface for Different Views ---
    tab1, tab2, tab3 = st.tabs(["Feature Correlations", "Feature Impact", "Risk-Adjusted Insights"])

    with tab1:
        st.subheader("Correlation Heatmap of Market Features")
        st.write(
            "This heatmap visualizes Pearson correlation coefficients between "
            "daily returns, volatility, volume change, and a simulated directional pressure metric. "
            "It helps identify linear relationships within the data."
        )
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax, vmin=-1, vmax=1)
        ax.set_title(f"Feature Correlation Matrix for {selected_ticker}", fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        st.pyplot(fig)
        plt.close(fig) # Close plot to free memory

        st.markdown("---")
        st.subheader("Interpreting the Heatmap:")
        st.write(
            "- **Positive Correlation (Red):** Features tend to increase/decrease together.\n"
            "- **Negative Correlation (Blue):** Features tend to move in opposite directions.\n"
            "- **Magnitude:** Values closer to +1 or -1 indicate a stronger linear relationship."
        )

    with tab2:
        st.subheader("Simulated Directional Pressure and Average Daily Return Impact")
        st.write(
            "This bar chart shows the average Daily Return associated with different "
            "bins of the 'Simulated Directional Pressure' feature (based on Close-Open return). "
            "It gives a simplified view of how intraday movement might relate to overall daily return."
        )
        fig_impact, ax_impact = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Pressure_Bin', y='Avg_Future_Impact', data=binned_feature_impact, palette='viridis', ax=ax_impact)
        ax_impact.set_title(f"Average Daily Return by Pressure Bin for {selected_ticker}", fontsize=16)
        ax_impact.set_xlabel("Pressure Bin (Categorized from Negative to Positive Close-Open Return)")
        ax_impact.set_ylabel("Average Daily Return")
        st.pyplot(fig_impact)
        plt.close(fig_impact)

        st.markdown("---")
        st.subheader("Sample of Daily Price and Volume Dynamics")
        st.write(
            "Observe the recent trends in Adjusted Close Price and Trading Volume for the selected asset."
        )
        fig_raw_dynamics, ax_raw_dynamics = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        ax_raw_dynamics[0].plot(raw_data_sample.index, raw_data_sample['Adj Close'], label='Adjusted Close Price', color='blue')
        ax_raw_dynamics[0].set_ylabel("Price", color='blue')
        ax_raw_dynamics[0].tick_params(axis='y', labelcolor='blue')
        ax_raw_dynamics[0].set_title(f"Price and Volume Dynamics for {selected_ticker} (Recent Data)", fontsize=16)
        ax_raw_dynamics[0].legend(loc="upper left")
        ax_raw_dynamics[0].grid(True, linestyle='--', alpha=0.7)

        ax_raw_dynamics[1].plot(raw_data_sample.index, raw_data_sample['Volume'], label='Volume', color='green', alpha=0.7)
        ax_raw_dynamics[1].set_xlabel("Date")
        ax_raw_dynamics[1].set_ylabel("Volume", color='green')
        ax_raw_dynamics[1].tick_params(axis='y', labelcolor='green')
        ax_raw_dynamics[1].legend(loc="upper left")
        ax_raw_dynamics[1].grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        st.pyplot(fig_raw_dynamics)
        plt.close(fig_raw_dynamics)

    with tab3:
        st.subheader(f"Strategy Insights for {risk_appetite} Profile")
        if risk_appetite == 'Conservative':
            st.info(
                "A **Conservative** strategy prioritizes stability and lower risk. "
                "Focus on robust, long-term trends and signals with very high statistical confidence. "
                "Avoid highly volatile periods or reliance on short-term predictive features."
            )
            st.write("**Considerations:**\n"
                     "- Utilize features with strong, consistent correlations over longer periods.\n"
                     "- Filter out periods of high daily volatility.\n"
                     "- Emphasize diversified positions rather than concentrated bets.")
        elif risk_appetite == 'Moderate':
            st.warning(
                "A **Moderate** strategy balances potential returns with controlled risk. "
                "This involves considering a wider range of signals and adapting to evolving market conditions, "
                "without taking excessive exposure to highly speculative movements."
            )
            st.write("**Considerations:**\n"
                     "- Combine multiple features, weighting them by their observed predictive power.\n"
                     "- Adjust position sizing based on a dynamic risk assessment.\n"
                     "- Monitor for significant shifts in correlations or feature impact.")
        else: # Aggressive
            st.error(
                "An **Aggressive** strategy seeks to capitalize on shorter-term, potentially higher-volatility opportunities. "
                "This approach might involve leveraging features with transient but strong predictive power, "
                "and being comfortable with higher potential drawdowns for larger gains."
            )
            st.write("**Considerations:**\n"
                     "- Focus on features showing high absolute correlation, even if inconsistent.\n"
                     "- Utilize strategies that can react quickly to sudden price movements or volume spikes.\n"
                     "- Implement strict stop-loss and take-profit orders due to increased risk.")

        st.markdown("---")
        st.info(
            "**Note:** In a real quantitative finance setting, these risk profiles would map to "
            "specific model parameters, filtering rules, and position sizing algorithms, "
            "dynamically adjusting the trading strategy based on the defined risk appetite."
        )

else:
    st.error(f"Could not load data for {selected_ticker}. Please ensure the ticker is valid for Yahoo Finance and check your internet connection.")

st.markdown("---")
st.markdown("Developed by Saurav Sen as part of a Quantitative Finance Project.")