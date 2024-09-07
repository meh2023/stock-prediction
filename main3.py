# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import streamlit as st

# Set the style for seaborn plots
sns.set_style('darkgrid')

def predict_stock_prices(stock_symbol, start_date, end_date, future_days):
    data = yf.download(stock_symbol, start=start_date, end=end_date)

    if data.empty:
        st.error(f"No data found for {stock_symbol}.")
        return

    data.fillna(method='ffill', inplace=True)
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data = data.dropna()

    X = data[['SMA_20', 'SMA_50']]
    y = data['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    
    st.write(f"Mean Squared Error (MSE) on the test set: {mse:.2f}")

    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=future_days, freq='D')
    future_data = pd.DataFrame(index=future_dates)
    future_data['SMA_20'] = data['SMA_20'].iloc[-1]
    future_data['SMA_50'] = data['SMA_50'].iloc[-1]
    future_predictions = model.predict(future_data[['SMA_20', 'SMA_50']])
    
    future_predictions_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Close': future_predictions
    })

    st.write(f"Future predictions:")
    st.write(future_predictions_df)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(data['Close'], label='Close Price')
    ax.plot(data['SMA_20'], label='20-Day SMA', linestyle='--')
    ax.plot(data['SMA_50'], label='50-Day SMA', linestyle='--')
    ax.set_title(f'{stock_symbol} Stock Price and Moving Averages')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y_test.index, y_test, label='Actual Prices')
    ax.plot(y_test.index, predictions, label='Predicted Prices', linestyle='--')
    ax.set_title(f'{stock_symbol} Actual vs Predicted Prices')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)

# Streamlit UI
st.title("Stock Price Prediction Tool")

stock_symbol = st.text_input("Enter stock symbol (e.g., AAPL):").upper()
start_date = st.date_input("Select start date", value=pd.to_datetime('2010-01-01'))
end_date = st.date_input("Select end date", value=pd.to_datetime('2023-12-31'))
future_days = st.slider("Select the number of future days to predict", min_value=1, max_value=365, value=30)

if st.button("Predict"):
    if stock_symbol and start_date and end_date:
        predict_stock_prices(stock_symbol, start_date, end_date, future_days)
    else:
        st.error("Please enter all fields.")
