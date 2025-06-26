import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import warnings

# Suppress specific sklearn warnings to keep the output clean
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Data Loading from Yahoo Finance ---
@st.cache_data # Cache data loading for performance in Streamlit
def load_yahoo_data_for_portfolio(ticker_symbols_list, custom_ticker_symbol, start_date, end_date, interval='1d'):
    """
    Fetches historical OHLCV data for a list of ticker symbols (from multiselect)
    and a single custom ticker symbol, calculates daily returns for each,
    and prepares data for correlation and ML.
    """
    all_tickers_to_process = list(ticker_symbols_list) # Start with multiselect tickers

    # Add custom ticker if provided and not already in the list
    if custom_ticker_symbol and custom_ticker_symbol.strip() and custom_ticker_symbol.strip().upper() not in [t.split('(')[-1].replace(')','').strip() for t in all_tickers_to_process]:
        all_tickers_to_process.append(custom_ticker_symbol.strip().upper())
    
    # If after adding custom ticker, we have more than 5, truncate for processing
    if len(all_tickers_to_process) > 5:
        st.warning("Limiting analysis to the first 5 selected/entered assets for consistency.")
        all_tickers_to_process = all_tickers_to_process[:5]

    if not all_tickers_to_process:
        return None, None, None # Return empty if no tickers selected or entered

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
                # Ensure these columns exist before operations
                if 'Adj Close' in raw_data_for_first_ticker.columns:
                    raw_data_for_first_ticker['Daily_Return'] = raw_data_for_first_ticker['Adj Close'].pct_change()
                else:
                    raw_data_for_first_ticker['Daily_Return'] = np.nan # Placeholder

                if 'Daily_Return' in raw_data_for_first_ticker.columns:
                    raw_data_for_first_ticker['Daily_Volatility'] = raw_data_for_first_ticker['Daily_Return'].rolling(window=5).std()
                else:
                    raw_data_for_first_ticker['Daily_Volatility'] = np.nan

                if 'Volume' in raw_data_for_first_ticker.columns:
                    raw_data_for_first_ticker['Volume_Change'] = raw_data_for_first_ticker['Volume'].pct_change()
                else:
                    raw_data_for_first_ticker['Volume_Change'] = np.nan

                if 'Close' in raw_data_for_first_ticker.columns and 'Open' in raw_data_for_first_ticker.columns:
                    raw_data_for_first_ticker['Simulated_Directional_Pressure'] = (raw_data_for_first_ticker['Close'] - raw_data_for_first_ticker['Open']) / raw_data_for_first_ticker['Open']
                else:
                    raw_data_for_first_ticker['Simulated_Directional_Pressure'] = np.nan

                raw_data_for_first_ticker = raw_data_for_first_ticker.dropna()

        except Exception as e:
            st.warning(f"Error fetching data for {actual_ticker}: {e}. Skipping.")
            continue

    if all_returns.empty:
        st.error("No valid data could be loaded for any of the selected or entered assets. Please try different selections.")
        return None, None, None # Return None for all outputs

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

    return portfolio_corr_matrix, individual_asset_features_data, all_returns # Return all_returns for ML


# --- Machine Learning Model ---
@st.cache_data
def predict_portfolio_outcome(all_returns_df):
    """
    Trains a simple Logistic Regression model to predict portfolio daily direction.
    Features: Lagged portfolio daily return, lagged portfolio volatility.
    Target: Binary (1 if next day's portfolio return is positive, 0 otherwise).
    Returns: accuracy, prediction_text, confidence_text, model, X_test, y_test, portfolio_returns_series
    """
    if all_returns_df is None or all_returns_df.empty:
        return None, "Not enough data for prediction.", None, None, None, None, None

    # Calculate simple average portfolio daily return (assuming equal weights for simplicity)
    portfolio_returns = all_returns_df.mean(axis=1).dropna()

    if len(portfolio_returns) < 100: # Need sufficient data points for meaningful training
        return None, "Not enough historical data for robust prediction (need at least 100 data points).", None, None, None, None, None

    # Create target variable: 1 if next day's return is positive, 0 otherwise
    # Shift returns by -1 to get the 'next day's return' as the target
    portfolio_returns_future = portfolio_returns.shift(-1)
    
    # Create features: Lagged portfolio return and lagged portfolio volatility
    features_df = pd.DataFrame({
        'Lag_Return_1d': portfolio_returns.shift(1),
        'Lag_Return_5d_MA': portfolio_returns.rolling(window=5).mean().shift(1),
        'Lag_Volatility_5d': portfolio_returns.rolling(window=5).std().shift(1)
    })

    # Combine features and target, then drop NaNs created by shifting/rolling
    ml_data = pd.DataFrame({'Target': (portfolio_returns_future > 0).astype(int)})
    ml_data = pd.concat([ml_data, features_df], axis=1).dropna()

    if ml_data.empty:
        return None, "Not enough data after feature engineering and NaN removal for prediction.", None, None, None, None, None

    X = ml_data[['Lag_Return_1d', 'Lag_Return_5d_MA', 'Lag_Volatility_5d']]
    y = ml_data['Target']

    if len(X) < 50: # Minimum data points for train/test split
         return None, f"Insufficient data ({len(X)} points) for robust ML model training. Need at least 50.", None, None, None, None, None

    # Check for target class distribution
    if y.nunique() < 2:
        return None, "Target variable has only one unique class (e.g., all up days or all down days). Cannot train a classifier.", None, None, None, None, None
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Check for class imbalance in training data to avoid errors if a class is missing in the split
    if y_train.nunique() < 2:
        return None, "Training data contains only one class for the target after splitting. Cannot train a classifier.", None, None, None, None, None
    
    model = LogisticRegression(random_state=42, solver='liblinear') # 'liblinear' is good for small datasets
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Predict the most likely direction for the "next" available day
    # Use the last available data point's features to predict the immediate future
    if not X.empty:
        last_features = X.iloc[[-1]] # Get the last row of features
        future_prediction_proba = model.predict_proba(last_features)[0]
        future_prediction_class = model.predict(last_features)[0]
        
        outcome_text = f"**Predicted Portfolio Direction for Next Day:** {'UP (Positive Return)' if future_prediction_class == 1 else 'DOWN (Negative/Zero Return)'}"
        confidence_text = f"**Confidence (UP vs. DOWN):** {future_prediction_proba[1]:.2f} (UP) vs {future_prediction_proba[0]:.2f} (DOWN)"
        
        return accuracy, outcome_text, confidence_text, model, X_test, y_test, portfolio_returns
    
    return accuracy, "Could not make a future prediction due to data issues.", None, None, None, None, None


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
# Ensure custom ticker doesn't exceed total 5 selected
if custom_ticker and custom_ticker.strip().upper() in [t.split('(')[-1].replace(')','').strip() for t in selected_tickers]:
    st.sidebar.warning("Custom ticker already selected in the list above.")
elif custom_ticker and (len(selected_tickers) >= 5): # Changed from > to >= to enforce strict 5 limit with custom
     st.sidebar.warning(f"Adding '{custom_ticker.upper()}' would exceed the 5-asset limit. Please deselect an asset if you wish to add this one.")


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
portfolio_corr_matrix, individual_asset_features_data, all_asset_daily_returns = load_yahoo_data_for_portfolio(
    selected_tickers, custom_ticker, start_date_str, end_date_str
)

if portfolio_corr_matrix is not None:
    # --- Tabbed Interface for Different Views ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Portfolio Correlations", "Individual Asset Features", "Feature Impact", "Risk-Adjusted Insights", "ML Portfolio Outcome"])

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
    if individual_asset_features_data and (selected_tickers or (custom_ticker and custom_ticker.strip())): # Ensure a selected/entered ticker for individual analysis
        # Get the display name of the first selected/entered ticker for these tabs
        # This logic needs to be careful: if custom is first processed and selected_tickers is empty, it should be custom
        # If selected_tickers has values, it should be the first one from there.
        active_tickers_list = list(selected_tickers)
        if custom_ticker and custom_ticker.strip():
            # Add custom_ticker to a temporary list to determine the 'first' ticker, avoiding duplicates
            if custom_ticker.strip().upper() not in [t.split('(')[-1].replace(')','').strip() for t in selected_tickers]:
                 active_tickers_list.append(custom_ticker.strip().upper())
        
        first_selected_ticker_display = active_tickers_list[0] if active_tickers_list else "Selected Asset"

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
    
    with tab5:
        st.subheader("Machine Learning Prediction: Portfolio Daily Direction")
        st.write(
            "This section attempts to predict the daily direction (up or down) of the "
            "portfolio's average return using a simple machine learning model (Logistic Regression). "
            "**Note: This is for demonstration purposes only and should NOT be used for actual investment decisions.** "
            "Financial market prediction is highly complex, and this simplified model is not robust enough for real-world use."
        )

        accuracy, outcome_text, confidence_text, model, X_test, y_test, portfolio_returns_series = predict_portfolio_outcome(all_asset_daily_returns)
        
        if accuracy is not None:
            st.write(f"**Model Accuracy (on test data):** {accuracy:.2%}")
            st.write(outcome_text)
            st.write(confidence_text)
            st.markdown("---")
            
            # Plotting model insights
            st.subheader("Model Insights & Visualization")
            
            # Plot 1: Feature Coefficients
            if model is not None and hasattr(model, 'coef_') and len(model.coef_[0]) == len(X_test.columns):
                coefficients = pd.DataFrame({
                    'Feature': X_test.columns,
                    'Coefficient': model.coef_[0]
                }).sort_values(by='Coefficient', ascending=False)

                fig_coef, ax_coef = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Coefficient', y='Feature', data=coefficients, palette='viridis', ax=ax_coef)
                ax_coef.set_title("Logistic Regression Feature Coefficients", fontsize=16)
                ax_coef.set_xlabel("Coefficient Value")
                ax_coef.set_ylabel("Feature")
                st.pyplot(fig_coef)
                plt.close(fig_coef)
                st.markdown(
                    "**Interpreting Coefficients:** Positive coefficients suggest that an increase in the feature value "
                    "increases the likelihood of an 'UP' day, and vice-versa for negative coefficients. "
                    "The magnitude indicates the strength of this relationship."
                )
                st.markdown("---")

            # Plot 2: Predicted Probabilities of UP Day vs. Actual Portfolio Returns
            if portfolio_returns_series is not None and not portfolio_returns_series.empty and model is not None and X_test is not None:
                # Use the full data for plotting probabilities, not just test set
                # Re-calculate features for the full portfolio_returns_series
                full_features_df = pd.DataFrame({
                    'Lag_Return_1d': portfolio_returns_series.shift(1),
                    'Lag_Return_5d_MA': portfolio_returns_series.rolling(window=5).mean().shift(1),
                    'Lag_Volatility_5d': portfolio_returns_series.rolling(window=5).std().shift(1)
                }).dropna()

                if not full_features_df.empty:
                    # Predict probabilities for the entire data used for features
                    predicted_probabilities = model.predict_proba(full_features_df)[:, 1] # Probability of class 1 (UP)
                    
                    plot_data = pd.DataFrame({
                        'Date': full_features_df.index,
                        'Predicted_Prob_UP': predicted_probabilities
                    })
                    # Merge with actual portfolio returns on Date
                    plot_data = plot_data.set_index('Date').join(portfolio_returns_series.rename('Actual_Portfolio_Return'))
                    plot_data = plot_data.dropna()

                    if not plot_data.empty:
                        fig_prob, ax_prob1 = plt.subplots(figsize=(12, 7))

                        # Plot Predicted Probability of UP
                        ax_prob1.plot(plot_data.index, plot_data['Predicted_Prob_UP'], color='purple', label='Predicted Probability of UP Day', alpha=0.7)
                        ax_prob1.set_xlabel("Date")
                        ax_prob1.set_ylabel("Predicted Probability (UP)", color='purple')
                        ax_prob1.tick_params(axis='y', labelcolor='purple')
                        ax_prob1.set_title("Predicted Probability of UP Day vs. Actual Portfolio Returns", fontsize=16)
                        ax_prob1.grid(True, linestyle='--', alpha=0.6)
                        ax_prob1.legend(loc="upper left")

                        # Create a second y-axis for Actual Portfolio Returns
                        ax_prob2 = ax_prob1.twinx()
                        ax_prob2.plot(plot_data.index, plot_data['Actual_Portfolio_Return'], color='green', label='Actual Daily Portfolio Return', alpha=0.5, linestyle='--')
                        ax_prob2.set_ylabel("Actual Daily Portfolio Return", color='green')
                        ax_prob2.tick_params(axis='y', labelcolor='green')
                        ax_prob2.legend(loc="upper right")

                        plt.tight_layout()
                        st.pyplot(fig_prob)
                        plt.close(fig_prob)

                        st.markdown(
                            "**Interpreting Probability Plot:** This graph shows how the model's predicted probability of "
                            "an 'UP' day (purple line) fluctuates over time, overlaid with the actual daily portfolio returns (green dashed line). "
                            "When the purple line is high, the model predicts a higher chance of a positive return."
                        )
                    else:
                        st.warning("Not enough data to plot predicted probabilities vs. actual returns after alignment.")
                else:
                    st.warning("Could not generate features for plotting probabilities. Check date range or data availability.")
            else:
                st.warning("Could not generate probability plot. Ensure sufficient data and a trained model are available.")

            st.markdown("---")
            st.info(
                "**How it works:** The model uses historical lagged daily returns and volatility of the portfolio "
                "to predict if the *next* day's average portfolio return will be positive (UP) or negative/zero (DOWN)."
            )
            st.warning(
                "**Disclaimer:** This is a basic example. Real-world predictive models require vast amounts of data, "
                "sophisticated feature engineering, advanced algorithms, and rigorous backtesting to be potentially effective. "
                "Past performance is not indicative of future results."
            )
        else:
            st.warning(outcome_text) # Display the reason for not being able to predict
else:
    st.error("Please select or enter at least one asset to begin analysis.")

st.markdown("---")
st.markdown("Developed by Saurav Sen as part of a Quantitative Finance Project.")
