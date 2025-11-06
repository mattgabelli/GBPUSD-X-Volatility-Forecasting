import yfinance as yf
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ticker_symbol = 'EURUSD=X'
start_date = '2015-01-01'
end_date = '2025-01-01'

def get_log_returns(ticker=ticker_symbol, start=start_date, end=end_date):
    # compute log returns for 10 years of GBP/USD data
    data = yf.download(ticker, start=start)
    data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
    data.dropna(inplace=True)
    return data

forex_data = get_log_returns()
print(forex_data.tail())