import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Load your pre-trained LSTM model
model = load_model("your_model.h5")  # Replace with your model file name

# Define a function to fetch historical stock data from Yahoo Finance
@st.cache_data
def get_stock_data(ticker, period="5y"):
    data = yf.download(ticker, period=period)
    return data

# Define a function to prepare data using MinMaxScaler
def prepare_data(df, sequence_length=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Close']])
    return scaled_data, scaler

# Define a function to forecast future stock prices using the pre-trained model
def forecast_stock(scaled_data, model, scaler, forecast_days=30, sequence_length=60):
    # Get the last sequence_length days from the scaled data
    last_sequence = scaled_data[-sequence_length:]
    forecast_input = last_sequence.reshape((1, sequence_length, 1))
    
    forecast = []
    for _ in range(forecast_days):
        # Predict the next day price
        next_price = model.predict(forecast_input)[0, 0]
        forecast.append(next_price)
        # Update the input sequence by appending the predicted value and dropping the first value
        forecast_input = np.roll(forecast_input, -1, axis=1)
        forecast_input[0, -1, 0] = next_price
        
    # Inverse transform the forecast back to the original scale
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
    return forecast

# Begin Streamlit dashboard layout
st.title("Stock Data and Forecast Dashboard")
st.markdown("This dashboard displays historical stock data along with a 30‑day forecast using a pre-trained LSTM model.")

# Sidebar for user inputs
st.sidebar.header("User Input Options")
ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL")
period = st.sidebar.selectbox("Select Data Period", options=["1y", "3y", "5y", "10y"], index=2)

if ticker:
    # Fetch the stock data
    data = get_stock_data(ticker, period)
    
    st.subheader(f"{ticker} Historical Data")
    st.dataframe(data.tail())
    
    # Display a candlestick chart of the historical data
    st.subheader("Candlestick Chart")
    fig = go.Figure(data=[
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Candlestick'
        )
    ])
    fig.update_layout(title=f"{ticker} Candlestick Chart")
    st.plotly_chart(fig, use_container_width=True)
    
    # Display some basic stock information using yfinance info
    st.subheader("Stock Information")
    ticker_info = yf.Ticker(ticker).info
    info_data = {
        "Previous Close": ticker_info.get("previousClose", "N/A"),
        "Open": ticker_info.get("open", "N/A"),
        "Day High": ticker_info.get("dayHigh", "N/A"),
        "Day Low": ticker_info.get("dayLow", "N/A"),
        "Volume": ticker_info.get("volume", "N/A"),
        "Market Cap": ticker_info.get("marketCap", "N/A")
    }
    st.table(pd.DataFrame(list(info_data.items()), columns=["Metric", "Value"]))
    
    # Prepare data for forecasting
    scaled_data, scaler = prepare_data(data)
    
    # Button to trigger forecast generation
    if st.button("Generate 30-Day Forecast"):
        forecast_prices = forecast_stock(scaled_data, model, scaler, forecast_days=30)
        # Create a date range for forecast dates (business days)
        last_date = data.index[-1]
        forecast_dates = pd.date_range(start=last_date, periods=31, freq='B')[1:]
        forecast_df = pd.DataFrame({
            "Date": forecast_dates,
            "Predicted Close": forecast_prices.flatten()
        })
        st.subheader("30-Day Forecast")
        st.dataframe(forecast_df)
        
        # Plot historical prices and forecasted prices together
        st.subheader("Historical vs Forecasted Prices")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Historical Prices'))
        fig2.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Predicted Close"], mode='lines', name='Forecasted Prices'))
        fig2.update_layout(title=f"{ticker} Historical and Forecasted Prices",
                           xaxis_title="Date", yaxis_title="Price (USD)")
        st.plotly_chart(fig2, use_container_width=True)
