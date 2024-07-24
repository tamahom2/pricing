import yfinance as yf
import numpy as np
from datetime import datetime

def historical_data(ticker):

    data = yf.download(ticker, period='max')
    data['Log Return'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
    # Drop the first row with NaN value due to the shift
    data = data.dropna()
    stock = yf.Ticker(ticker)
    historical_prices = stock.history(period='1d', interval='1m')
    return data,historical_prices['Close'].iloc[-1]
