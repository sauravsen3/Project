import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# --- Data Loading from Yahoo Finance ---
@st.cache_data # Cache data loading for performance in Streamlit
def load_yahoo_data_for_portfolio(ticker_symbols_list, custom_ticker_symbol, start_date, end_date, interval='1d'):
    """
    Fetches historical OHLCV data for a list of ticker symbols (from multiselect)
    and a single custom ticker symbol, calculates daily returns for each,
    and prepares data for correlation.
    """
    all_tickers_to_process = list(ticker_symbols_list) # Start with multiselect tickers

    # Add custom ticker if provided and not already in the list
    if custom_ticker_symbol and custom_ticker_symbol.strip() and custom_ticker_symbol.strip().upper() not in [t.split('(')[-1].replace(')','').strip() for t in all_tickers_to_process]:
        all_tickers_to_process.append(custom_ticker_symbol.strip().upper())
    
    if not all_tickers_to_process:
        return None, None # Return empty if no tickers selected or entered

    all_returns = pd.DataFrame()
    raw_data_for_first_ticker = None # To keep individual feature analysis for the first selected ticker

    # Define a more comprehensive mapping, including more diversified assets
    yf_ticker_map = {
        'EUR/USD': 'EURUSD=X',
        'GBP/USD': 'GBPUSD=X',
        'SPY (S&P 500 ETF)': 'SPY',
        'QQQ (Nasdaq 100 ETF)': 'QQQ',
        'Apple (AAPL)': 'AAPL',
        'Microsoft (MSFT)': 'MSFT',
        'Amazon (AMZN)': 'AMZN',
        'Google (GOOGL)': 'GOOGL',
        'Tesla (TSLA)': 'TSLA',
        'Gold ETF (GLD)': 'GLD', # Commodity ETF
        'Silver ETF (SLV)': 'SLV', # Commodity ETF
        'US Aggregate Bond ETF (AGG)': 'AGG', # Broad Bond ETF
        '20+ Year Treasury Bond ETF (TLT)': 'TLT', # Long-term Bond ETF
        'Crude Oil ETF (USO)': 'USO', # Oil Commodity ETF
        'Emerging Markets ETF (EEM)': 'EEM', # Emerging Markets Equity
        'Real Estate ETF (XLRE)': 'XLRE', # Real Estate Sector
        'Healthcare ETF (XLV)': 'XLV' # Healthcare Sector
    }
    
    for i, ticker_display_name in enumerate(all_tickers_to_process):
        # Handle both predefined display names and custom input
        actual_ticker = yf_ticker_map.get(ticker_display_name, ticker_display_name)
        
        try:
            data = yf.download(actual_ticker, start=start_date, end=end_date, interval=interval)
            
            if data.empty:
                st.warning(f"No data found for {actual_ticker} for the selected period. Skipping.")
                continue # Skip to next ticker if no data
            
            # Ensure 'Adj Close' column exists
            if 'Adj Close' not in data.columns:
                if 'Close' in data.columns:
                    # st.warning(f"'Adj Close' not found for {actual_ticker}. Using 'Close' price for calculations.")
                    data['Adj Close'] = data['Close']
                else:
                    st.warning(f"Neither 'Adj Close' nor 'Close' found for {actual_ticker}. Skipping.")
                    continue

            # Calculate Daily Returns for portfolio correlation
            daily_returns = data['Adj Close'].pct_change().dropna()
            all_returns[ticker_display_name] = daily_returns
            
            # Store raw data and features for the first ticker for individual analysis tabs
            # This applies to the first ticker in the combined list (selected or custom)
            if i == 0:
                raw_data_for_first_ticker = data.copy()
                raw_data_for_first_ticker['Daily_Return'] = raw_data_for_first_ticker['Adj Close'].pct_change()
                raw_data_for_first_ticker['Daily_Volatility'] = raw_data_for_first_ticker['Daily_Return'].rolling(window=5).std()
                raw_data_for_first_ticker['Volume_Change'] = raw_data_for_first_ticker['Volume'].pct_change()
                raw_data_for_first_ticker['Simulated_Directional_Pressure'] = (raw_data_for_first_ticker['Close'] - raw_data_for_first_ticker['Open']) / raw_data_for_first_ticker['Open']
                raw_data_for_first_ticker = raw_data_for_first_ticker.dropna()

        except Exception as e:
            st.warning(f"Error fetching data for {actual_ticker}: {e}. Skipping.")
            continue

    if all_returns.empty:
        st.error("No valid data could be loaded for any of the selected or entered assets. Please try different selections.")
        return None, None

    # Calculate portfolio correlation matrix
    portfolio_corr_matrix = all_returns.corr()

    # Prepare individual asset features for the first ticker (if available)
    individual_asset_features_data = None
    if raw_data_for_first_ticker is not None and not raw_data_for_first_ticker.empty:
        features_for_corr = raw_data_for_first_ticker[['Daily_Return', 'Daily_Volatility', 'Volume_Change', 'Simulated_Directional_Pressure']]
        individual_asset_features_corr = features_for_corr.corr()
        
        bins = pd.cut(raw_data_for_first_ticker['Simulated_Directional_Pressure'], bins=5, labels=False, duplicates='drop')
        # Ensure bins is not empty before grouping
        if not bins.empty and len(bins.unique()) > 1:
            binned_impact = raw_data_for_first_ticker.groupby(bins)['Daily_Return'].mean().reset_index()
            binned_impact.columns = ['Pressure_Bin', 'Avg_Future_Impact']
        else:
            binned_impact = pd.DataFrame(columns=['Pressure_Bin', 'Avg_Future_Impact'])
            st.warning(f"Not enough data to bin 'Simulated_Directional_Pressure' for {all_tickers_to_process[0]}.")

        individual_asset_features_data = {
            'corr_matrix': individual_asset_features_corr,
            'raw_data_sample': raw_data_for_first_ticker.tail(100).copy(),
            'binned_feature_impact': binned_impact
        }

    return portfolio_corr_matrix, individual_asset_features_data


# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Portfolio & Market Data Insights")

st.title("Portfolio & Market Data Insights Dashboard")
st.markdown(
    """
    Explore inter-asset correlations for portfolio diversification, and detailed feature analysis for individual assets.
    **Note:** Data is sourced from Yahoo Finance (OHLCV). Real-time high-frequency order book data is not available.
    """
)

# --- Sidebar for Inputs ---
st.sidebar.header("Configuration")

all_available_tickers = [
    'SPY (S&P 500 ETF)', 'QQQ (Nasdaq 100 ETF)', 'Apple (AAPL)', 'Microsoft (MSFT)',
    'Amazon (AMZN)', 'Google (GOOGL)', 'Tesla (TSLA)',
    'Gold ETF (GLD)', 'Silver ETF (SLV)', 'US Aggregate Bond ETF (AGG)',
    '20+ Year Treasury Bond ETF (TLT)', 'Crude Oil ETF (USO)',
    'Emerging Markets ETF (EEM)', 'Real Estate ETF (XLRE)', 'Healthcare ETF (XLV)',
    'EUR/USD', 'GBP/USD' # FX at the end, acknowledging their potential unreliability
]

selected_tickers = st.sidebar.multiselect(
    "Select Assets for Portfolio Analysis (up to 5):",
    options=all_available_tickers,
    default=['SPY (S&P 500 ETF)', 'QQQ (Nasdaq 100 ETF)', 'US Aggregate Bond ETF (AGG)'],
    max_selections=5
)

# New: Text input for custom ticker
custom_ticker = st.sidebar.text_input(
    "Or enter a custom Yahoo Finance Ticker (e.g., TSLA, BTC-USD):",
    value=""
)
if custom_ticker and custom_ticker.strip().upper() in [t.split('(')[-1].replace(')','').strip() for t in selected_tickers]:
    st.sidebar.warning("Custom ticker already selected in the list above.")


# Date inputs
st.sidebar.markdown("### Date Range")
default_end_date = datetime.now()
default_start_date = default_end_date - timedelta(days=365 * 2) # Default to 2 years for more data

start_date_input = st.sidebar.date_input("Start Date", default_start_date)
end_date_input = st.sidebar.date_input("End Date", default_end_date)

# Convert date objects to strings for the function
start_date_str = start_date_input.strftime('%Y-%m-%d')
end_date_str = end_date_input.strftime('%Y-%m-%d')


st.sidebar.markdown("---")
risk_appetite = st.sidebar.select_slider(
    "Select Strategy Risk Profile:",
    options=['Conservative', 'Moderate', 'Aggressive'],
    value='Moderate'
)
st.sidebar.info(
    "Risk Profile provides general guidance for strategy interpretation."
)

# --- Main Content Area ---
st.header("Portfolio Diversification & Individual Asset Insights")
st.markdown("---")

# Pass both selected and custom tickers to the data loading function
portfolio_corr_matrix, individual_asset_features_data = load_yahoo_data_for_portfolio(
    selected_tickers, custom_ticker, start_date_str, end_date_str
)

if portfolio_corr_matrix is not None:
    # --- Tabbed Interface for Different Views ---
    tab1, tab2, tab3, tab4 = st.tabs(["Portfolio Correlations", "Individual Asset Features", "Feature Impact", "Risk-Adjusted Insights"])

    with tab1:
        st.subheader("Inter-Asset Correlation Heatmap (Daily Returns)")
        st.write(
            "This heatmap visualizes the Pearson correlation coefficients of daily returns "
            "between the selected assets. Lower positive or negative correlations indicate "
            "better diversification benefits."
        )
        fig_portfolio_corr, ax_portfolio_corr = plt.subplots(figsize=(10, 8))
        sns.heatmap(portfolio_corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax_portfolio_corr, vmin=-1, vmax=1)
        ax_portfolio_corr.set_title("Portfolio Asset Correlation Matrix", fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        st.pyplot(fig_portfolio_corr)
        plt.close(fig_portfolio_corr)

        st.markdown("---")
        st.subheader("Interpreting Portfolio Correlations:")
        st.write(
            "- **Positive Correlation (Red):** Assets tend to move in the same direction.\n"
            "- **Negative Correlation (Blue):** Assets tend to move in opposite directions (good for diversification).\n"
            "- **Zero Correlation:** Assets move independently (also good for diversification)."
        )
        st.info("For diversification, you generally seek assets with low positive or negative correlations.")

    # Individual Asset Analysis tabs (now conditional on having data for at least one asset)
    if individual_asset_features_data and selected_tickers: # Ensure a selected ticker for individual analysis
        # Get the display name of the first selected ticker for these tabs
        first_selected_ticker_display = selected_tickers[0]
        
        with tab2:
            st.subheader(f"Feature Correlation Heatmap for {first_selected_ticker_display}")
            st.write(
                "This heatmap visualizes Pearson correlation coefficients between "
                "daily returns, volatility, volume change, and a simulated directional pressure metric "
                f"for **{first_selected_ticker_display}**. It helps identify linear relationships within the data."
            )
            fig_features, ax_features = plt.subplots(figsize=(10, 8))
            sns.heatmap(individual_asset_features_data['corr_matrix'], annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax_features, vmin=-1, vmax=1)
            ax_features.set_title(f"Feature Correlation Matrix for {first_selected_ticker_display}", fontsize=16)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            st.pyplot(fig_features)
            plt.close(fig_features)

            st.markdown("---")
            st.subheader("Sample of Daily Price and Volume Dynamics")
            st.write(
                f"Observe the recent trends in Adjusted Close Price and Trading Volume for **{first_selected_ticker_display}**."
            )
            fig_raw_dynamics, ax_raw_dynamics = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

            ax_raw_dynamics[0].plot(individual_asset_features_data['raw_data_sample'].index, individual_asset_features_data['raw_data_sample']['Adj Close'], label='Adjusted Close Price', color='blue')
            ax_raw_dynamics[0].set_ylabel("Price", color='blue')
            ax_raw_dynamics[0].tick_params(axis='y', labelcolor='blue')
            ax_raw_dynamics[0].set_title(f"Price and Volume Dynamics for {first_selected_ticker_display} (Recent Data)", fontsize=16)
            ax_raw_dynamics[0].legend(loc="upper left")
            ax_raw_dynamics[0].grid(True, linestyle='--', alpha=0.7)

            ax_raw_dynamics[1].plot(individual_asset_features_data['raw_data_sample'].index, individual_asset_features_data['raw_data_sample']['Volume'], label='Volume', color='green', alpha=0.7)
            ax_raw_dynamics[1].set_xlabel("Date")
            ax_raw_dynamics[1].set_ylabel("Volume", color='green')
            ax_raw_dynamics[1].tick_params(axis='y', labelcolor='green')
            ax_raw_dynamics[1].legend(loc="upper left")
            ax_raw_dynamics[1].grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            st.pyplot(fig_raw_dynamics)
            plt.close(fig_raw_dynamics)

        with tab3:
            st.subheader(f"Simulated Directional Pressure and Average Daily Return Impact for {first_selected_ticker_display}")
            if not individual_asset_features_data['binned_feature_impact'].empty:
                fig_impact, ax_impact = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Pressure_Bin', y='Avg_Future_Impact', data=individual_asset_features_data['binned_feature_impact'], palette='viridis', ax=ax_impact)
                ax_impact.set_title(f"Average Daily Return by Pressure Bin for {first_selected_ticker_display}", fontsize=16)
                ax_impact.set_xlabel("Pressure Bin (Categorized from Negative to Positive Close-Open Return)")
                ax_impact.set_ylabel("Average Daily Return")
                st.pyplot(fig_impact)
                plt.close(fig_impact)
            else:
                st.warning(f"Insufficient data to show 'Simulated Directional Pressure' impact for {first_selected_ticker_display} for the selected date range.")

        with tab4:
            st.subheader(f"Strategy Insights for {risk_appetite} Profile (Focus on {first_selected_ticker_display})")
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
        st.info("Select at least one asset to see individual asset features and insights. If you entered a custom ticker, it will be the primary asset if no multi-selected tickers are chosen.")


else:
    st.error("Please select or enter at least one asset to begin analysis.")

st.markdown("---")
st.markdown("Developed by Saurav Sen as part of a Quantitative Finance Project.")
