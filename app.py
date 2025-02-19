import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

model = load_model("stock_prediction_model.h5", custom_objects={'mse': MeanSquaredError()})


# --- Helper Functions ---
@st.cache_data
def fetch_stock_data(ticker, period="5y"):
    data = yf.download(ticker, period=period)
    return data

def prepare_scaler_and_data(df, window=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Close']])
    return scaled_data, scaler

def forecast_prices(scaled_data, model, scaler, forecast_days=30, window=60):
    last_window = scaled_data[-window:]
    forecast_input = last_window.reshape((1, window, 1))
    
    forecast = []
    for _ in range(forecast_days):
        pred = model.predict(forecast_input)[0, 0]
        forecast.append(pred)
        forecast_input = np.roll(forecast_input, -1, axis=1)
        forecast_input[0, -1, 0] = pred
        
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
    return forecast

# --- Streamlit App Layout ---
st.set_page_config(page_title="Stock Forecast Dashboard", layout="wide")
st.title("📈 Stock Data & Forecast Dashboard")

# Sidebar
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL")
period = st.sidebar.selectbox("Select Data Period", options=["1y", "3y", "5y", "10y"], index=2)
forecast_trigger = st.sidebar.button("Generate 30-Day Forecast")

# Main Panel
if ticker:
    # Fetch Data
    data = fetch_stock_data(ticker, period)
    
    # Layout: Split main area into two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"{ticker} Historical Data")
        st.dataframe(data.tail())
        
        # Display Candlestick Chart
        st.subheader("Candlestick Chart")
        fig = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            increasing_line_color='green',
            decreasing_line_color='red'
        )])
        fig.update_layout(title=f"{ticker} Candlestick Chart", xaxis_title="Date", yaxis_title="Price (USD)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Stock Overview")
        # Fetch additional stock info
        ticker_info = yf.Ticker(ticker).info
        info_df = pd.DataFrame({
            "Metric": ["Previous Close", "Open", "Day High", "Day Low", "Volume", "Market Cap"],
            "Value": [ticker_info.get("previousClose", "N/A"), 
                      ticker_info.get("open", "N/A"), 
                      ticker_info.get("dayHigh", "N/A"), 
                      ticker_info.get("dayLow", "N/A"), 
                      ticker_info.get("volume", "N/A"), 
                      ticker_info.get("marketCap", "N/A")]
        })
        st.table(info_df)
    
    # Forecast Section
    st.markdown("---")
    st.subheader("30-Day Forecast")
    
    scaled_data, scaler = prepare_scaler_and_data(data)
    if forecast_trigger:
        forecast = forecast_prices(scaled_data, model, scaler)
        forecast_dates = pd.date_range(start=data.index[-1], periods=31, freq='B')[1:]
        forecast_df = pd.DataFrame({
            "Date": forecast_dates,
            "Predicted Close": forecast.flatten()
        })
        st.dataframe(forecast_df)
        
        # Plot Historical & Forecasted Prices
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Historical Prices'))
        fig2.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Predicted Close"], mode='lines', name='Forecasted Prices'))
        fig2.update_layout(title=f"{ticker} Historical & Forecasted Prices", xaxis_title="Date", yaxis_title="Price (USD)")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Click the button in the sidebar to generate the 30-day forecast.")
