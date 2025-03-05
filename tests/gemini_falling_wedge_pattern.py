import yfinance as yf
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt
from scipy.stats import linregress

def simple_draw_upper_trendline(data):
    """
    Draws an upper trendline on a plot of the given data.

    Args:
        data (list or numpy.ndarray): A list or array of numerical data points.
    """
    x = np.arange(len(data))

    df_high = data.copy()

    while len(df_high) > 2:
        slope, intercept, r_value, p_value, std_err = linregress(x, df_high)

        # Remove the lowest point
        df_high = df_high[df_high > (slope * x + intercept)]
        x = x[df_high > (slope * x + intercept)]

    plt.plot(x, slope * x + intercept, '--r', label='Upper Trendline')

def draw_upper_trendline(df, lookback=50, peak_window=3):
    """Draws an upper trendline connecting lower highs."""

    high = df['High'].values

    # 1. Find Peaks (Potential Lower Highs)
    peak_indices = argrelextrema(high[-lookback:], np.greater, order=peak_window)[0]  # Find peaks

    # Filter for "Lower Highs" - IMPROVED
    lower_highs_indices = []
    for i in range(1, len(peak_indices)):
      if high[-lookback:][peak_indices[i]] < high[-lookback:][peak_indices[i-1]]: #Current peak is lower than the previous
        lower_highs_indices.append(peak_indices[i])

    lower_highs_x = np.array(lower_highs_indices)
    lower_highs_y = high[-lookback:][lower_highs_indices]

    if len(lower_highs_x) < 2:  # Need at least 2 points for a line
        return None  # No suitable trendline

    try:
        # 2. RANSAC for Robust Line Fitting
        ransac = RANSACRegressor()
        ransac.fit(lower_highs_x.reshape(-1, 1), lower_highs_y)

        # 3. Extend the Trendline
        x_ext = np.array([lower_highs_x[0], lookback - 1])  # Extend to the right edge
        y_ext = ransac.predict(x_ext.reshape(-1, 1))

        #Convert to actual indices relative to the dataframe
        x_ext_final = len(high) - lookback + x_ext

        return (x_ext_final[0], y_ext[0], x_ext_final[1], y_ext[1]) #return tuple for plotting

    except ValueError:  # Handle cases where RANSAC fails
        return None

def detect_falling_wedge(df, lookback=50, peak_window=3, trough_window=3, convergence_factor=0.5):
    """
    Detects falling wedge patterns in price data.

    Args:
        df (pd.DataFrame): DataFrame with OHLCV data (Open, High, Low, Close, Volume).
        lookback (int): Lookback period for pattern identification.
        peak_window (int): Window for peak detection.
        trough_window (int): Window for trough detection.
        convergence_factor (float): Factor for convergence check (0-1).

    Returns:
        tuple: (True, upper_line, lower_line) if a pattern is detected, 
               (False, None, None) otherwise.  upper_line and lower_line are tuples
               of (x1, y1, x2, y2) representing the lines.
    """

    high = df['High'].values
    low = df['Low'].values
    volume = df['Volume'].values
    # 1. Peak and Trough Detection
    peak_indices = argrelextrema(high[-lookback:], np.greater, order=peak_window)[0]
    trough_indices = argrelextrema(low[-lookback:], np.less, order=trough_window)[0]
    breakpoint()

    peaks_x = peak_indices
    peaks_y = high[-lookback:][peak_indices]
    #print("peak_indices",peak_indices)
    #print(peaks_y)
    troughs_x = trough_indices
    troughs_y = low[-lookback:][trough_indices]


    if len(peaks_x) < 3 or len(troughs_x) < 3:  # Need at least 3 points for a line
      print('here')
      return False, None, None

    # 2. RANSAC Line Fitting (Upper Resistance)
    try:
        ransac_upper = RANSACRegressor()
        ransac_upper.fit(peaks_x.reshape(-1, 1), peaks_y)

        # Extend the line (crude extension - improve this)
        x_ext_upper = np.array([peaks_x[0], lookback -1]) #from the first peak to the present
        y_ext_upper = ransac_upper.predict(x_ext_upper.reshape(-1, 1))

        upper_line = (x_ext_upper[0] + len(high) - lookback, y_ext_upper[0], x_ext_upper[1] + len(high) - lookback, y_ext_upper[1])

    except ValueError: #handles the case where RANSAC fails to fit
        return False, None, None
        print('here1')


    # 3. RANSAC Line Fitting (Lower Support)
    try:
        ransac_lower = RANSACRegressor()
        ransac_lower.fit(troughs_x.reshape(-1, 1), troughs_y)

        x_ext_lower = np.array([troughs_x[0], lookback - 1])
        y_ext_lower = ransac_lower.predict(x_ext_lower.reshape(-1, 1))

        lower_line = (x_ext_lower[0] + len(high) - lookback, y_ext_lower[0], x_ext_lower[1] + len(high) - lookback, y_ext_lower[1])

    except ValueError: #handles the case where RANSAC fails to fit
        print('here2')
        return False, None, None

    # 4. Convergence Check (Simplified)
    # Check if the slopes have the correct signs and are converging.
    slope_upper = (upper_line[3] - upper_line[1]) / (upper_line[2] - upper_line[0]) if (upper_line[2] - upper_line[0]) != 0 else 0
    slope_lower = (lower_line[3] - lower_line[1]) / (lower_line[2] - lower_line[0]) if (lower_line[2] - lower_line[0]) != 0 else 0

    if slope_upper >= 0 or slope_lower >= 0 or slope_upper > slope_lower: #upper should be negative, lower should be negative and lower should be less steep
        print('here3')
        return True, upper_line, lower_line
        return False, None, None

    # 5. Lower Highs/Lows Check (Basic) - Could be more robust
    # Check if the highs and lows within the wedge are generally decreasing.
    # This is a basic check; you might want to add more sophisticated logic.

    # 6. Volume Check (Optional) - Add this if you have volume data
    # Check if volume is decreasing within the wedge.

    return True, upper_line, lower_line



ticker = 'AAPL'
df = yf.download(ticker, period='12mo')
simple_draw_upper_trendline(df)
exit()
#df.to_csv('AAPL.csv', index=False)
# Example Usage (replace with your data loading):
# Assuming your data is in a CSV file named 'data.csv'
#try:
    #df = pd.read_csv('AAPL.csv')
    # Make sure your CSV has columns 'Open', 'High', 'Low', 'Close', and 'Volume'
#    df['Date'] = pd.to_datetime(df['Date']) # If you have a date column, convert it to datetime
#    df = df.set_index('Date') # Set the date as index
#except FileNotFoundError:
#    print("Please provide a 'data.csv' file.")
#    exit()

#pattern_detected, upper_line, lower_line = detect_falling_wedge(df, lookback=250)

if pattern_detected:
    print("Falling Wedge Pattern Detected!")
    print("Upper Line:", upper_line)
    print("Lower Line:", lower_line)

    # You can now plot the lines on your chart using a charting library
    # like Matplotlib or Plotly.

    plt.plot(df.index, df['Close'])  # Plot the closing prices
    plt.plot([df.index[int(upper_line[0])], df.index[int(upper_line[2])]], [upper_line[1], upper_line[3]], color='blue', label='Upper Trendline')
    plt.plot([df.index[int(lower_line[0])], df.index[int(lower_line[2])]], [lower_line[1], lower_line[3]], color='red', label='Lower Trendline')

    plt.legend()
    plt.title('Falling Wedge Pattern')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()

else:
    print("No Falling Wedge Pattern detected.")
