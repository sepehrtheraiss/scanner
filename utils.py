import pandas as pd
import numpy as np
from scipy.signal import find_peaks

def _print(x):
    if x.iloc[-1]-x.iloc[0] > 0:
        return 1
    else: 
        if x[-1]-x[0] < 0:
            return -1
        return 0

def detect_wedge(df, window=3):
    # Define the rolling window
    roll_window = window
    # Create a rolling window for High and Low
    df['high_roll_max'] = df['High'].rolling(window=roll_window).max()
    df['low_roll_min'] = df['Low'].rolling(window=roll_window).min()
    df['trend_high'] = df['High'].rolling(window=roll_window).apply(lambda x: 1 if (x.iloc[-1]-x.iloc[0])>0 else -1 if (x.iloc[-1]-x.iloc[0])<0 else 0)
    df['trend_low'] = df['Low'].rolling(window=roll_window).apply(lambda x: 1 if (x.iloc[-1]-x.iloc[0])>0 else -1 if (x.iloc[-1]-x.iloc[0])<0 else 0)
    # Create a boolean mask for Wedge Up pattern
    mask_wedge_up = (df['high_roll_max'] >= df['High'].shift(1)) & (df['low_roll_min'] <= df['Low'].shift(1)) & (df['trend_high'] == 1) & (df['trend_low'] == 1)
    # Create a boolean mask for Wedge Down pattern
        # Create a boolean mask for Wedge Down pattern
    mask_wedge_down = (df['high_roll_max'] <= df['High'].shift(1)) & (df['low_roll_min'] >= df['Low'].shift(1)) & (df['trend_high'] == -1) & (df['trend_low'] == -1)
    # Create a new column for Wedge Up and Wedge Down pattern and populate it using the boolean masks
    df['wedge_pattern'] = np.nan
    df.loc[mask_wedge_up, 'wedge_pattern'] = 1#'Wedge Up'
    df.loc[mask_wedge_down, 'wedge_pattern'] = -1#'Wedge Down'
    return df

def utils_find_peaks(data, prominence=None, width=None, threshold=None, negative=False):
    """
    Finds peaks in a specified column of a Pandas DataFrame.

    Args:
        data: array 
        prominence (optional):  Minimum prominence of peaks.  See scipy.signal.find_peaks documentation.
        width (optional): Minimum width of peaks in samples. See scipy.signal.find_peaks documentation.
        threshold (optional): Minimum threshold for peaks. See scipy.signal.find_peaks documentation.

    Returns:
        A tuple containing:
            - A list of peak indices (integers).
            - A dictionary of properties for each peak (prominences, widths, etc.) as returned by scipy.signal.find_peaks.
            Returns None, None if the input is invalid or the column doesn't exist.
    """

    peaks, properties = find_peaks(data, prominence=prominence, width=width, threshold=threshold)

    if negative:
        # For negative peaks (troughs), invert the data
        inverted_data = -data
        peaks, properties = find_peaks(inverted_data, prominence=prominence, width=width, threshold=threshold)
        peak_indices = peaks.tolist()
        #invert the data to plot correctly
        peak_values = -inverted_data[peaks]
    else:
        peaks, properties = find_peaks(data, prominence=prominence, width=width, threshold=threshold)
        peak_indices = peaks.tolist()
        peak_values = data[peaks]
    return peak_indices, properties, peak_values
    #return peak_values

def multi_index_find_peaks_in_dataframe(df, ticker, column_name, prominence=None, width=None, threshold=None, negative=False):
    """
    Finds peaks in a specified column of a Pandas DataFrame.

    Args:
        df: The Pandas DataFrame.
        column_name: The name of the column to find peaks in.
        prominence (optional):  Minimum prominence of peaks.  See scipy.signal.find_peaks documentation.
        width (optional): Minimum width of peaks in samples. See scipy.signal.find_peaks documentation.
        threshold (optional): Minimum threshold for peaks. See scipy.signal.find_peaks documentation.

    Returns:
        A tuple containing:
            - A list of peak indices (integers).
            - A dictionary of properties for each peak (prominences, widths, etc.) as returned by scipy.signal.find_peaks.
            Returns None, None if the input is invalid or the column doesn't exist.
    """

    if not isinstance(df, pd.DataFrame) or df.empty or column_name not in df.columns:
        return None, None

    #data = df.Close.AAPL.values
    data = df[(column_name, ticker)].values

    peaks, properties = find_peaks(data, prominence=prominence, width=width, threshold=threshold)

    if negative:
        # For negative peaks (troughs), invert the data
        inverted_data = -data
        peaks, properties = find_peaks(inverted_data, prominence=prominence, width=width, threshold=threshold)
        peak_indices = peaks.tolist()
        #invert the data to plot correctly
        peak_values = -inverted_data[peaks]
    else:
        peaks, properties = find_peaks(data, prominence=prominence, width=width, threshold=threshold)
        peak_indices = peaks.tolist()
        peak_values = data[peaks]
    return peak_indices, properties, peak_values
    #return peaks.tolist(), properties

def slope():
    x1, x2 = df.index[close_peak_indices][-2:]
    y1, y2 = close_peak_values[-2:]
    x1_num = pd.Timestamp(x1).timestamp()
    x2_num = pd.Timestamp(x2).timestamp()
    m = (y2 - y1) / (x2_num - x1_num)
    # Calculate the y-intercept (b) of the line (y = mx + b)
    b = y1 - m * x1_num

    def abline(slope, intercept):
        """Plot a line from slope and intercept"""
        axes = plt.gca()
        x_vals = np.array(axes.get_xlim())
        y_vals = intercept + slope * x_vals
        plt.plot(x_vals, y_vals, '--')
