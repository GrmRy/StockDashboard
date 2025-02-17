import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import yfinance as yf

# We'll import your model-related functions
from model import prepare_data, train_lstm_model, forecast_stock

# 1. Page Title
st.title("Stock Data and Forecast App")

# 2. Sidebar Ticker Selection
st.sidebar.header("Select a stock ticker")
ticker = st.sidebar.text_input("Enter ticker (e.g., AAPL, TSLA)", value="AAPL")

# 3. Fetch Data
@st.cache_data
def get_stock_data(ticker, period='5y'):
    """Fetch historical data from Yahoo Finance."""
    data = yf.download(ticker, period=period)
    return data

if ticker:
    data = get_stock_data(ticker)
    
    # 4. Display Data and Candlestick Chart
    st.subheader(f"{ticker} Historical Data (last 5 years)")
    st.dataframe(data.tail())  # Show last few rows
    
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Candlesticks'
            )
        ]
    )
    fig.update_layout(title=f"{ticker} Candlestick Chart")
    st.plotly_chart(fig, use_container_width=True)
    
    # 5. Basic Stock Stats
    # Using yfinance get_info() to pull basic stats
    stock_info = yf.Ticker(ticker).info
    st.subheader("Basic Stock Information")
    # Show a few key stats
    basic_stats = {
        'Previous Close': stock_info.get('previousClose'),
        'Open': stock_info.get('open'),
        'Day High': stock_info.get('dayHigh'),
        'Day Low': stock_info.get('dayLow'),
        'Market Cap': stock_info.get('marketCap'),
        'Volume': stock_info.get('volume')
    }
    st.table(pd.DataFrame(list(basic_stats.items()), columns=["Metric", "Value"]))
    
    # 6. LSTM Forecast Section
    st.subheader("Price Forecast")
    
    # Button to trigger forecast
    if st.button("Generate 30-Day Forecast"):
        # Prepare Data
        X, y, scaler = prepare_data(data)
        # Train Model (or you could load a pre-trained model if you have one saved)
        model = train_lstm_model(X, y)
        # Forecast
        forecast_df = forecast_stock(data, model, scaler)
        
        st.write("#### Forecasted Prices (Next 30 Days)")
        st.dataframe(forecast_df)
        
        # Plot Forecast vs Historical
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Historical Prices'))
        fig2.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Predicted Close'],
                                  name='Predicted Prices'))
        fig2.update_layout(title=f"{ticker} - Historical & Forecasted Prices")
        st.plotly_chart(fig2, use_container_width=True)
