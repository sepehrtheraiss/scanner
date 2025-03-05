import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot(df, upper_trendline=None, lower_trendline=None):
    # Visualize the pattern
    plt.plot(df.index, df['Close'], color='blue')
    plt.plot(df.index, df['Open'], color='black')

    plt.plot(df.index, upper_trendline, color='green')

    plt.plot(df.index, lower_trendline, color='red')

    plt.show()

def identify_falling_wedge(df, window=3):

    # Calculate upper and lower trendlines

    upper_trendline = df['High'].rolling(window=window).max()

    lower_trendline = df['Low'].rolling(window=window).min()

    plot(df, upper_trendline, lower_trendline)
    return 0

    # Check for pattern formation

    is_falling_wedge = (upper_trendline < upper_trendline.shift(1)) & (lower_trendline < lower_trendline.shift(1)) 

    return is_falling_wedge


# Assuming 'df' is your price data
ticker = 'AAPL'
df = yf.download(ticker, period='12mo')
#print(df)
pattern_detected = identify_falling_wedge(df, window=10) 
#pattern_detected = detect_triangle_pattern(df) 
#print("pattern_detected: ", pattern_detected)

