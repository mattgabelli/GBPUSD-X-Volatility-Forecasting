import yfinance as yf
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ticker_symbol = 'GBPUSD=X'
start_date = datetime.datetime(2015, 1, 1)
end_date = datetime.datetime(2025, 1, 1)

def get_log_returns(ticker=ticker_symbol, start=start_date, end=end_date):
    # compute log returns for 10 years of GBP/USD data
    data = yf.download(ticker, start=start, end=end)
    lr = data.Close.pct_change().dropna()
    return lr


