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
    




# Main function
def main():
    image_url = "https://github.com/swathi0710/StockScreenerApp1/blob/main/stock-market-6368031_640.jpg"
    st.image(image_url, use_column_width=True)
    st.markdown("<h2><span style='color: blue;'>Stock Screener</span></h2>", unsafe_allow_html=True)

    # # Load data
    # url = 'https://drive.google.com/file/d/1riceAkRePuCgkG9QzhJ56-2EyQHUK3Qv/view?usp=sharing'
    # url = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
    # data = pd.read_csv(url)
    data=calculate_metrics_for_all_stocks()
    st.write(data)
    

 

if __name__ == "__main__":
    main()
