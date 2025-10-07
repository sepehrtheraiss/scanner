# fetch.py
# downloads stock dataframe
# saves and loads from file.csv
# all functions returns map[ticker] -> dataframe

import yfinance as yf
import pandas as pd


class Fetch:
    def __init__(self, period='3mo', interval='1d', start=None, end=None):
        self.period = period
        self.interval = interval
        self.start = start
        self.end = end
        
    def test_data(self): 
        data = {}
        data['AAPL'] = pd.read_csv('AAPL_2023-07-21_2024-05-17_1d',index_col=0, parse_dates=True)
        return data

    def mag_7(self):
        data = {}
        MAG_7 = ['AAPL', 'AMZN', 'GOOGL', 'NVDA', 'MSFT', 'META', 'TSLA']
        for ticker in MAG_7:
            data[ticker] = yf.download(ticker, period=self.period, start=self.start, end=self.end, interval=self.interval, multi_level_index=False)
        return data

    def sp500(self):
        data = {}
        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        tickers = sp500['Symbol'].tolist()
        for ticker in tickers:
            data[ticker] = yf.download(ticker, period=self.period, start=self.start, end=self.end, interval=self.interval, multi_level_index=False)
        return data

    def ticker(self, ticker):
        data = {}
        data[ticker] = yf.download(ticker, period=self.period, start=self.start, end=self.end, interval=self.interval, multi_level_index=False)
        return data

    def file(self, fname):
        data = {}
        with open(fname, 'r') as f:
            tickers = f.readlines()
            for ticker in tickers:
                ticker = ticker.strip()
                data[ticker] = yf.download(fname, period=self.period, start=self.start, end=self.end, interval=self.interval, multi_level_index=False)
        return data
