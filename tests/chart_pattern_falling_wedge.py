import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import argparse

def is_falling_wedge(data, extrema_indices, tolerance=0.02):
    """
    Checks if the given price data and extrema indices form a falling wedge pattern.

    Args:
        data (pd.Series): Price data.
        extrema_indices (np.ndarray): Indices of local extrema (minima and maxima).
        tolerance (float): Tolerance for angle difference between trendlines.

    Returns:
        bool: True if it's a falling wedge, False otherwise.
    """
    if len(extrema_indices) < 4:
        return False

    # Separate highs and lows
    high_indices = extrema_indices[argrelextrema(data.values[extrema_indices], np.greater)[0]]
    low_indices = extrema_indices[argrelextrema(data.values[extrema_indices], np.less)[0]]

    if len(high_indices) < 2 or len(low_indices) < 2:
        return False

    # Calculate trendlines
    high_slope, high_intercept = np.polyfit(high_indices, data.iloc[high_indices], 1)
    low_slope, low_intercept = np.polyfit(low_indices, data.iloc[low_indices], 1)

    # Check if both trendlines are downward sloping
    if high_slope >= 0 or low_slope >= 0:
        return False

    # Check if trendlines are converging
    if high_slope >= low_slope:
        return False

    # Check if the angle difference between trendlines is within tolerance
    angle_diff = np.arctan(low_slope) - np.arctan(high_slope)
    if angle_diff > tolerance:
        return False

    return True

def detect_falling_wedges(data, window=10):
    """
    Detects falling wedge patterns in the given price data.

    Args:
        data (pd.Series): Price data.
        window (int): Window size for finding local extrema.

    Returns:
        list: List of tuples, where each tuple contains:
            - start_index (int): Start index of the falling wedge pattern.
            - end_index (int): End index of the falling wedge pattern.
            - high_indices (np.ndarray): Indices of local maxima within the pattern.
            - low_indices (np.ndarray): Indices of local minima within the pattern.
    """
    extrema_indices = argrelextrema(data.values, np.less_equal, order=window)[0]
    print("extrema: ", extrema_indices)
    wedge_patterns = []

    for i in range(len(extrema_indices) - 3):
        pattern_indices = extrema_indices[i:i+4]
        fall_wedge = is_falling_wedge(data, pattern_indices)
        print("falling wedge: ",fall_wedge) 
        if fall_wedge: 
            wedge_patterns.append((pattern_indices[0], pattern_indices[-1]))

    return wedge_patterns

def plot_falling_wedge(data, start_index, end_index):
    """
    Plots the falling wedge pattern on the given price data.

    Args:
        data (pd.Series): Price data.
        start_index (int): Start index of the falling wedge pattern.
        end_index (int): End index of the falling wedge pattern.
    """
    wedge_data = data.iloc[start_index:end_index+1]

    # Get high and low indices within the wedge
    extrema_indices = argrelextrema(wedge_data.values, np.less_equal, order=2)[0]
    high_indices = extrema_indices[argrelextrema(wedge_data.values[extrema_indices], np.greater)[0]]
    low_indices = extrema_indices[argrelextrema(wedge_data.values[extrema_indices], np.less)[0]]

    # Calculate trendlines
    high_slope, high_intercept = np.polyfit(high_indices, wedge_data.iloc[high_indices], 1)
    low_slope, low_intercept = np.polyfit(low_indices, wedge_data.iloc[low_indices], 1)

    high_trendline = high_slope * np.arange(len(wedge_data)) + high_intercept
    low_trendline = low_slope * np.arange(len(wedge_data)) + low_intercept

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(wedge_data.index, wedge_data.values, label='Price Data', color='blue')
    plt.plot(wedge_data.index[high_indices], wedge_data.iloc[high_indices], 'o', color='red', label='Highs')
    plt.plot(wedge_data.index[low_indices], wedge_data.iloc[low_indices], 'o', color='green', label='Lows')
    plt.plot(wedge_data.index, high_trendline, '--', color='red', label='High Trendline')
    plt.plot(wedge_data.index, low_trendline, '--', color='green', label='Low Trendline')
    plt.title('Falling Wedge Pattern')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


def main(period='6mo'):
    # Example usage
    #ticker = 'AAPL'
    ticker = 'CARR'
    print("downloading historical data for ", ticker)
    data = yf.download(ticker, period=period)
    wedge_patterns = detect_falling_wedges(data)

    if wedge_patterns:
        start_index, end_index = wedge_patterns[0]
        plot_falling_wedge(data, start_index, end_index)
    else:
        print("No falling wedge patterns found in the data.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    parser.add_argument('--p', help='period')
    args = parser.parse_args()
    main(args.p)
