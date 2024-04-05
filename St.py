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
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
import math
from datetime import datetime, timedelta
import plotly.offline as po
import main_functions as mfn

@st.experimental_memo
# Function to decompose time series into trend, seasonal, and residual components
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

# Main function
def main():
    st.title("Stock Screener")

    # Load data
    url = 'https://drive.google.com/file/d/1riceAkRePuCgkG9QzhJ56-2EyQHUK3Qv/view?usp=sharing'
    url = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
    data = pd.read_csv(url)

    # Sidebar - Select stock
    st.sidebar.title("Select Stock")
    cur_A = st.sidebar.selectbox('Choose Stock', sorted(data['Stock'].unique()))

    # Filter data for selected stock
    selected_data = data[data['Stock'] == cur_A]

    # Display basic statistics and first few rows
    display_basic_statistics(selected_data)
    display_first_few_rows(selected_data)

    # Plot time series data
    st.subheader(f"Stock Price Analysis for {cur_A}")
    fig = plot_time_series(selected_data, f"Stock Price Analysis for {cur_A}")
    st.plotly_chart(fig)

    # Decompose time series into trend, seasonal, and residual components
    trend, seasonal, residual = decompose_time_series(selected_data['Close'])
    plot_decomposed_components(trend, seasonal, residual)

if __name__ == "__main__":
    main()
