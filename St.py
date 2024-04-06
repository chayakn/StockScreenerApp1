import streamlit as st
st. set_page_config(layout="wide")
col1, col2 = st.columns([1, 3], gap = 'medium')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
import math
from datetime import datetime, timedelta
import plotly.offline as po
import st2 as page2

def bidirectional_slider(label, min_value, max_value, default_value):
    left_value, right_value = st.sidebar.slider(label, min_value=min_value, max_value=max_value, value=(min_value, max_value))
    return left_value, right_value
    
@st.experimental_memo
def calculate_metrics_for_all_stocks(data):
    
    summary_data = pd.DataFrame(columns=['Stock', 'Quarterly Sales Variance', 'P/E', 'Dividend Yield %', 'Buy/Hold/Sell'])

    # Calculate metrics for each stock
    for stock in data['Stock'].unique():
        # Mock calculations for demonstration
        quarterly_sales_variance = np.random.uniform(0, 10)
        pe_ratio = np.random.uniform(5, 20)
        dividend_yield = np.random.uniform(0, 5)
        buy_hold_sell = 'Buy' if np.random.rand() < 0.5 else 'Sell'  # Random buy/sell recommendation
        
        summary_data = pd.concat([summary_data, pd.DataFrame({
            'Stock': [stock],
            'Quarterly Sales Variance': [quarterly_sales_variance],
            'P/E': [pe_ratio],
            'Dividend Yield %': [dividend_yield],
            'Buy/Hold/Sell': [buy_hold_sell]
        })], ignore_index=True)

    return summary_data
    
def decompose_time_series(data):
    result = seasonal_decompose(data, model='additive', period=1)
    trend = result.trend
    seasonal = result.seasonal
    residual = result.resid
    return trend, seasonal, residual

# Function to plot time series data
def plot_time_series(data, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Price')
    return fig

# Function to display basic statistics
def display_basic_statistics(data):
    st.subheader("Basic Statistics")
    st.write(data.describe())

# Function to display first few rows of the dataset
def display_first_few_rows(data):
    st.subheader("First Few Rows of Data")
    st.write(data.head())



# Function to plot decomposed components
def plot_decomposed_components(trend, seasonal, residual):
    st.subheader("Seasonal Decomposition")
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(trend, label='Trend')
    axes[0].legend()
    axes[1].plot(seasonal, label='Seasonal')
    axes[1].legend()
    axes[2].plot(residual, label='Residual')
    axes[2].legend()
    st.pyplot(fig)

# Function to perform Prophet forecast
def prophet_forecast(data):
    df = data.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
    model = Prophet(daily_seasonality=False)
    model.fit(df)
    
    # Calculate the last date in the dataset
    last_date = data.index[-1]
    
    # Make future dataframe starting from the day after the last date in the dataset
    future = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=200, freq='D')
    future = pd.DataFrame({'ds': future})
    
    forecast = model.predict(future)
    return forecast

# Function to plot Prophet forecast
def plot_prophet_forecast(data, forecast):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast', line=dict(color='orange')))
    fig.update_layout(title='Prophet Forecast', xaxis_title='Date', yaxis_title='Price')
    return fig

# Main function
def main():
    image_url = "https://raw.githubusercontent.com/swathi0710/StockScreenerApp1/main/stock-market-6368031_640.jpg"
    st.image(image_url,width=150)
    st.markdown("<h2><span style='color: blue;'>Stock Screener</span></h2>", unsafe_allow_html=True)

    # Load data
    url = 'https://drive.google.com/file/d/1riceAkRePuCgkG9QzhJ56-2EyQHUK3Qv/view?usp=sharing'
    url = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
    data = pd.read_csv(url)
    
    
    # Convert 'Date' column to datetime format and set it as index
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)


    #rsi
    delta = data['Close'].diff()

    # Separate gains and losses
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

    # Calculate RS (Relative Strength)
    rs = gain / loss
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))

    # Assign RSI values to the DataFrame
    data['RSI'] = rsi

    #MACD
    short_ema = data['Close'].ewm(span=12, adjust=False).mean()
    long_ema = data['Close'].ewm(span=26, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=12, adjust=False).mean()
    macd_histogram = macd_line - signal_line


    data['MACD'] = macd_line
    data['MACD_Signal'] = signal_line
    data['MACD_Histogram'] = macd_histogram

    # Sidebar - Select stock and forecast date range
    st.sidebar.title("Select Stock data to be filtered")
    cur_A = st.sidebar.selectbox('Choose Stock', sorted(data['Stock'].unique()))
    
    # Button to display all stock data
    if st.sidebar.button("ALL STOCK DATA"):
        # Calculate metrics for all stocks
        page2()
        # st.dataframe(data)
        # summary_data = calculate_metrics_for_all_stocks(data)
        # st.subheader("Summary Statistics of All Stocks")
        # st.dataframe(summary_data)

    # Filter data for selected stock
    selected_data = data[data['Stock'] == cur_A]
    # Filter data for selected stock
    selected_data = data[data['Stock'] == cur_A]
    columns_with_bidirectional_slider=['Open','Close','Volume','Change Pct','RSI','MACD']
    # Filter Rows by close column
    for col in columns_with_bidirectional_slider:
        try:
            low,up=bidirectional_slider(col, min_value=selected_data[col].min(), max_value=selected_data[col].max(), default_value=(0.0,30.0))
            selected_data =selected_data[(selected_data[col]>=low) & (selected_data[col]<=up)]
        except:
            pass
    # Filter Rows by close column
    # close_slider = st.sidebar.slider('Close Price', min_value=data['Close'].min(), max_value=data['Close'].max())
    

    # Display basic statistics and first few rows
    st.subheader(f"Stock Price Analysis for {cur_A}")
    #st.write(selected_data.describe())

    page_number = st.number_input('Page Number', min_value=1, max_value=len(selected_data) // 10 + 1, value=1)
    start_idx = (page_number - 1) * 10
    end_idx = min(start_idx + 10, len(selected_data))
    paginated_df = selected_data.iloc[start_idx:end_idx]
    st.write(paginated_df)
    # st.dataframe(selected_data)
    # Plot time series data
    
    st.subheader(f"Stock Price Analysis for {cur_A}")

        # Create traces
    trace_macd = go.Scatter(x=selected_data.index, y=selected_data['MACD'], mode='lines', name='MACD')
    trace_signal = go.Scatter(x=selected_data.index, y=selected_data['MACD_Signal'], mode='lines', name='MACD Signal')
    trace_histogram = go.Bar(x=selected_data.index, y=selected_data['MACD_Histogram'], name='MACD Histogram')

# Create figure
    fig = go.Figure()

# Add traces to the figure
    fig.add_trace(trace_macd)
    fig.add_trace(trace_signal)
    fig.add_trace(trace_histogram)

# Update layout
    fig.update_layout(title='MACD Analysis', xaxis_title='Date', yaxis_title='Value')

# Show plot
    fig.show()
    st.plotly_chart(fig)
    fig = plot_time_series(selected_data, f"Stock Price Analysis for {cur_A}")
    st.plotly_chart(fig)

    if len(selected_data)>0:
        # Sidebar control for moving average window size
        # window_size = st.sidebar.slider('Moving Average Window Size', min_value=1, max_value=30, value=10)
        # # Calculate moving averages
        # for col in selected_data.columns[1:]:
        #     selected_data[f'{col} Moving Average'] = selected_data[col].rolling(window=window_size).mean()
        
        # # Create a Plotly figure
        # fig = go.Figure()
        

        
       

        # Decompose time series into trend, seasonal, and residual components
        try:
            trend, seasonal, residual = decompose_time_series(selected_data['Close'])
            plot_decomposed_components(trend, seasonal, residual)
        except:
            pass
    
        # Prophet Forecast
        try:
            st.subheader("Prophet Forecast")
            forecast = prophet_forecast(selected_data)
            fig_forecast = plot_prophet_forecast(selected_data, forecast)
            st.plotly_chart(fig_forecast)
        except:
            pass

if __name__ == "__main__":
    main()
