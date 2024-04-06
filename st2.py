import streamlit as st
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

def bidirectional_slider(label, min_value, max_value, default_value):
    left_value, right_value = st.sidebar.slider(label, min_value=min_value, max_value=max_value, value=(min_value, max_value))
    return left_value, right_value

@st.experimental_memo
def calculate_metrics_for_all_stocks(data):
    
    summary_data = pd.DataFrame(columns=['Stock', 'Quarterly Sales Variance', 'P/E', 'Dividend Yield %', 'Buy/Hold/Sell'])

    # Calculate metrics for each stock
    for stock in data['Stock'].unique():  # Mock calculations for demonstration
        quarterly_sales_variance = np.random.uniform(0, 10)
        pe_ratio = np.random.uniform(5, 20)
        dividend_yield = np.random.uniform(0, 5)
        buy_hold_sell = 'Buy' if np.random.rand() < 0.5 else 'Sell'  # Random buy/sell recommendation
        volume = np.random.uniform(1, 1000000)
        market_cap = np.random.uniform(1, 1000000)
        Industry = np.random.choice(['Healthcare', 'Basic Materials', 'Finacial', 'Consumer Defensive', 'Technology', 'Communication'])
        change = np.random.uniform(1.0, 100.0)
        summary_data = pd.concat([summary_data, pd.DataFrame({
            'Stock': [stock],
            'Industry': [Industry],
            'Volume': [volume],
            'Market Capping': [market_cap],
            'Change percentage': [change],
            'Quarterly Sales Variance': [quarterly_sales_variance],
            'P/E': [pe_ratio],
            'Dividend Yield %': [dividend_yield],
            'Buy/Hold/Sell': [buy_hold_sell],
        })], ignore_index=True)
        
        columns_with_bidirectional_slider = ['Volume', 'Market Capping', 'Change percentage', 'P/E', 'Dividend Yield %']
        # Filter Rows by close column
        selected_data = summary_data
        # for col in columns_with_bidirectional_slider:
        #     # try:
        #         # low, up = bidirectional_slider(col, min_value=selected_data[col].min(), max_value=selected_data[col].max(), default_value=(0.0, 30.0))
        #     selected_data = selected_data[(selected_data[col] >= low) & (selected_data[col] <= up)]
        #     # except:
        #         # pass
    return selected_data
 



# Main function
def main():
    
    url = 'https://drive.google.com/file/d/1riceAkRePuCgkG9QzhJ56-2EyQHUK3Qv/view?usp=sharing'
    url = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
    data = pd.read_csv(url)
  
    data1=calculate_metrics_for_all_stocks(data)
    st.dataframe(data1)
    

 

if __name__ == "__main__":
    main()
