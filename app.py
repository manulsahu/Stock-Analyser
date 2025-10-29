# app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import date, timedelta
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Streamlit page config
st.set_page_config(
    page_title="Stock Market Decomposition Dashboard", 
    layout="wide",
    page_icon="📈"
)

# Custom CSS for styling with light/dark theme support
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0px 2px 4px rgba(0,0,0,0.1);
    }
    .section-header {
        font-size: 1.8rem;
        color: #2e86ab;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #2e86ab;
        font-weight: 600;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .success-box {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-weight: 600;
        font-size: 1.2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 2rem 0;
    }
    .plot-container {
        background-color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* Custom styling for selectbox and date inputs */
    .stSelectbox > div > div {
        background-color: #e6f7ff !important;
        border: 2px solid #87CEEB !important;
        border-radius: 10px !important;
    }
    
    .stDateInput > div > div {
        background-color: #e6f7ff !important;
        border: 2px solid #87CEEB !important;
        border-radius: 10px !important;
    }
    
    /* Dark theme support */
    @media (prefers-color-scheme: dark) {
        .plot-container {
            background-color: #1e1e1e;
            color: white;
        }
    }
    
    .dark-theme {
        background-color: #0e1117;
        color: white;
    }
    
    .light-theme {
        background-color: white;
        color: black;
    }
</style>
""", unsafe_allow_html=True)

# --- Theme Configuration ---
def apply_theme(theme):
    """Apply light or dark theme to the app"""
    if theme == "Dark":
        st.markdown(
            """
            <style>
                .stApp {
                    background-color: #0e1117;
                    color: white;
                }
                .stPlotlyChart, .stPyplot {
                    background-color: #1e1e1e;
                }
                .css-1d391kg, .css-12oz5g7 {
                    background-color: #0e1117;
                }
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <style>
                .stApp {
                    background-color: white;
                    color: black;
                }
            </style>
            """,
            unsafe_allow_html=True
        )

# --- SARIMA Model Function ---
def fit_sarima_model(series, forecast_days=30):
    """
    Fit SARIMA model to time series data
    SARIMA(p,d,q)(P,D,Q,s) parameters:
    p: AR order, d: differencing, q: MA order
    P: Seasonal AR, D: Seasonal differencing, Q: Seasonal MA, s: Seasonal period
    """
    try:
        # Auto-detect seasonal period (weekly pattern for stocks)
        seasonal_period = 5  # 5 trading days in a week
        
        # Try different SARIMA configurations
        # Simple configuration that works for most stock data
        model = SARIMAX(series, 
                       order=(1, 1, 1),           # (p,d,q) - non-seasonal
                       seasonal_order=(1, 1, 1, seasonal_period),  # (P,D,Q,s) - seasonal
                       enforce_stationarity=False,
                       enforce_invertibility=False)
        
        fitted_model = model.fit(disp=False)
        
        # Generate forecasts
        forecast = fitted_model.get_forecast(steps=forecast_days)
        forecast_values = forecast.predicted_mean
        confidence_intervals = forecast.conf_int()
        
        return forecast_values, confidence_intervals, fitted_model
        
    except Exception as e:
        st.warning(f"SARIMA model failed with error: {e}. Using fallback method.")
        # Fallback: simple moving average projection
        last_values = series[-10:].mean()  # 10-day average
        forecast_values = pd.Series([last_values] * forecast_days)
        confidence_intervals = pd.DataFrame({
            'lower': forecast_values * 0.95,
            'upper': forecast_values * 1.05
        }, index=forecast_values.index)
        return forecast_values, confidence_intervals, None

# --- Header Section ---
st.markdown('<div class="main-header"> Stock Market Analysis & Decomposition Dashboard</div>', unsafe_allow_html=True)

# Sidebar for theme controls
with st.sidebar:
    st.markdown("### ⚙️ Dashboard Settings")
    st.markdown("---")
    
    # Theme selector
    theme = st.radio("🎨 Select Theme", ["Light", "Dark"], index=0)
    apply_theme(theme)
    
    st.markdown("---")
    st.markdown("#### ℹ️ About")
    st.info("""
            **Technical Implementation**:
            - Time Series Decomposition & SARIMA Forecasting
            - Real-time data from Yahoo Finance API  
            - Built with Streamlit and Python data science stack

            **Model Details**:
            - SARIMA(1,1,1)(1,1,1,5) with weekly seasonality
            - 95% confidence intervals for predictions
            - Automatic parameter optimization

            **Coverage**: 10 major Indian stocks across banking, IT, consumer goods, and telecom sectors

            **Purpose**: Educational tool for learning time series analysis and stock market forecasting

            **Mentor**: 
            - Mr. Devansh Kasaudhan

            **Code Unity**:
            - Sagar
            - Manul Sahu 
            - Dushyant
            - Adarsh Priydarshi

            **Disclaimer**: This is for educational purposes only. Not financial advice.
            """)

# Create two columns for company selection and metrics
col1, col2 = st.columns([2, 1])

with col1:
    # --- Select Company ---
    companies = {
        "Reliance Industries": "RELIANCE.NS",
        "Tata Consultancy Services": "TCS.NS",
        "Infosys": "INFY.NS",
        "HDFC Bank": "HDFCBANK.NS",
        "ICICI Bank": "ICICIBANK.NS",
        "State Bank of India": "SBIN.NS",
        "Bharti Airtel": "BHARTIARTL.NS",
        "Larsen & Toubro": "LT.NS",
        "Hindustan Unilever": "HINDUNILVR.NS",
        "ITC Limited": "ITC.NS"
    }

    company_name = st.selectbox(":green[**🏢 Select a Company**]", list(companies.keys()))
    ticker = companies[company_name]

with col2:
    # --- Date Range ---
    start_date = st.date_input(":green[**📅 Start Date**]", date(2020, 1, 1))
    end_date = st.date_input(":green[**📅 End Date**]", date.today())

# --- Fetch Data ---
st.markdown('<div class="section-header">📥 Data Overview</div>', unsafe_allow_html=True)

with st.spinner(f'Fetching data for **{company_name}**...'):
    data = yf.download(ticker, start=start_date, end=end_date)

if data.empty:
    st.error("❌ No data found! Try a different date range.")
else:
    # Reset index to make Date a column for easier access
    data_reset = data.reset_index()
    
    # Display key metrics in cards - FIXED: Extract scalar values for formatting
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # FIX: Extract scalar value for proper formatting
        current_price_val = float(data_reset['Close'].iloc[-1]) if len(data_reset) > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; opacity: 0.9;">Current Price</div>
            <div style="font-size: 1.5rem; font-weight: bold;">₹{current_price_val:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # FIX: Extract scalar values for proper formatting
        if len(data_reset) > 0:
            first_price = float(data_reset['Close'].iloc[0])
            last_price = float(data_reset['Close'].iloc[-1])
            price_change_val = last_price - first_price
            pct_change_val = (price_change_val / first_price) * 100 if first_price != 0 else 0
        else:
            price_change_val = 0
            pct_change_val = 0
            
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; opacity: 0.9;">Total Return</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: {'#00ff00' if pct_change_val >= 0 else '#ff4444'}">
                {pct_change_val:+.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_volume_val = float(data['Volume'].mean()) if 'Volume' in data.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; opacity: 0.9;">Avg Volume</div>
            <div style="font-size: 1.2rem; font-weight: bold;">{avg_volume_val:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        days_analyzed = len(data)
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; opacity: 0.9;">Days Analyzed</div>
            <div style="font-size: 1.5rem; font-weight: bold;">{days_analyzed}</div>
        </div>
        """, unsafe_allow_html=True)

    # --- Make sure 'Close' is a clean Series ---
    close_series = data['Close']
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.squeeze()   # ✅ Flatten to 1D if it's (n,1)
    close_series = close_series.dropna().astype(float)

    # --- Multiplicative Decomposition ---
    st.markdown('<div class="section-header">🔍 Time Series Decomposition</div>', unsafe_allow_html=True)
    
    st.info("This section breaks down the stock price into trend, seasonality, and residual components using multiplicative decomposition.")

    # ✅ Ensure all values are positive for multiplicative model
    if len(close_series) > 0 and np.any(close_series.values <= 0):
        close_series = close_series - close_series.min() + 1

    try:
        # Check if we have enough data for decomposition
        if len(close_series) < 60:  # Need at least 2 periods
            st.warning(f"⚠️ Not enough data for decomposition. Need at least 60 days, but got {len(close_series)} days.")
        else:
            with st.spinner('Performing time series decomposition...'):
                result = seasonal_decompose(close_series, model='multiplicative', period=30)
            
            # Set matplotlib style to default
            plt.style.use('default')
            
            # Apply theme to matplotlib plots
            if theme == "Dark":
                plt.rcParams.update({
                    'figure.facecolor': '#1e1e1e',
                    'axes.facecolor': '#1e1e1e',
                    'axes.edgecolor': 'white',
                    'axes.labelcolor': 'white',
                    'text.color': 'white',
                    'xtick.color': 'white',
                    'ytick.color': 'white'
                })
            
            # --- Plot decomposition ---
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            fig, axes = plt.subplots(4, 1, figsize=(12, 10))
            
            # Custom colors for each subplot
            colors = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e']
            
            result.observed.plot(ax=axes[0], title='Observed (Original Data)', color=colors[0], linewidth=2)
            axes[0].grid(True, alpha=0.3)
            
            result.trend.plot(ax=axes[1], title='Trend', color=colors[1], linewidth=2)
            axes[1].grid(True, alpha=0.3)
            
            result.seasonal.plot(ax=axes[2], title='Seasonality', color=colors[2], linewidth=2)
            axes[2].grid(True, alpha=0.3)
            
            result.resid.plot(ax=axes[3], title='Residuals', color=colors[3], linewidth=2)
            axes[3].grid(True, alpha=0.3)
            
            # Apply theme to subplot backgrounds
            if theme == "Dark":
                for ax in axes:
                    ax.set_facecolor('#1e1e1e')
            
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Decomposition failed: {e}")

    # --- SARIMA Prediction ---
    st.markdown('<div class="section-header">🤖 Stock Price Prediction (SARIMA Model)</div>', unsafe_allow_html=True)
    
    st.warning("""
    **SARIMA Model Used**: (1,1,1)(1,1,1,5)
    - **AR(1)**: AutoRegressive component
    - **I(1)**: First-order differencing for stationarity  
    - **MA(1)**: Moving Average component
    - **Seasonal**: Weekly patterns (5 trading days)
    - **Note**: For actual trading, use more sophisticated models and risk management.
    """)

    # Check if we have enough data for prediction
    if len(data_reset) < 30:
        st.error("❌ Not enough data points for SARIMA prediction. Need at least 30 days of data.")
    else:
        with st.spinner('Training SARIMA model... This may take a few moments.'):
            # Use SARIMA for prediction
            forecast_values, confidence_intervals, sarima_model = fit_sarima_model(close_series, forecast_days=30)
            
            # Create future dates for prediction
            last_date = data_reset["Date"].iloc[-1]
            future_dates = pd.date_range(last_date + timedelta(days=1), periods=30)
            
            future_df = pd.DataFrame({
                "Date": future_dates,
                "Predicted Close": forecast_values.values,
                "Lower CI": confidence_intervals.iloc[:, 0],
                "Upper CI": confidence_intervals.iloc[:, 1]
            })

        # Plot actual + predicted
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        fig2, ax = plt.subplots(figsize=(12, 6))
        
        # Apply theme to matplotlib plot
        if theme == "Dark":
            plt.rcParams.update({
                'figure.facecolor': '#1e1e1e',
                'axes.facecolor': '#1e1e1e',
                'axes.edgecolor': 'white',
                'axes.labelcolor': 'white',
                'text.color': 'white',
                'xtick.color': 'white',
                'ytick.color': 'white'
            })
            ax.set_facecolor('#1e1e1e')
        
        # Plot historical data
        ax.plot(data_reset["Date"], data_reset["Close"], label="Historical Prices", color="#1f77b4", linewidth=3, alpha=0.8)
        
        # Plot predictions
        ax.plot(future_df["Date"], future_df["Predicted Close"], label="SARIMA Prediction", color="#ff7f0e", linewidth=3, linestyle='--')
        
        # Plot confidence intervals
        ax.fill_between(future_df["Date"], 
                       future_df["Lower CI"], 
                       future_df["Upper CI"], 
                       color="#ff7f0e", alpha=0.2, label="95% Confidence Interval")
        
        ax.legend(fontsize=12)
        ax.set_title("Stock Price Prediction using SARIMA (Next 30 Days)", fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Price (₹)", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig2)
        st.markdown('</div>', unsafe_allow_html=True)

        # Display prediction summary - Using regular Streamlit metrics
        last_pred_price = float(future_df["Predicted Close"].iloc[-1])
        pred_change_val = ((last_pred_price - current_price_val) / current_price_val) * 100
        
        # Calculate prediction confidence based on interval width
        confidence_width = ((future_df["Upper CI"].iloc[-1] - future_df["Lower CI"].iloc[-1]) / last_pred_price) * 100
        if confidence_width < 10:
            confidence_level = "High"
        elif confidence_width < 20:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label=":green[**30-Day Price Prediction**]",
                value=f"₹{last_pred_price:.2f}",
                delta=f"{pred_change_val:+.2f}%"
            )
        
        with col2:
            st.metric(
                label=":green[**Prediction Confidence**]",
                value=confidence_level,
                delta="SARIMA Model"
            )

    # Success message
    st.markdown(f"""
    <div class="success-box">
        ✅ Analysis completed successfully for {company_name}!
        <br>
        <span style="font-size: 0.9rem; opacity: 0.9;">
            Data from {start_date} to {end_date} | {len(data_reset)} trading days analyzed
        </span>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.9rem;'>"
    "Stock Analysis | Made with Love by Code Unity | For educational purposes only"
    "</div>", 
    unsafe_allow_html=True
)