import yfinance as yf
import pandas as pd
from scipy.signal import find_peaks, find_peaks_cwt
import matplotlib.pyplot as plt

def find_peaks_in_column(df, column_name, prominence=None, width=None, threshold=None, negative=False):
    """
    Finds peaks (positive or negative) in a specified column of a Pandas DataFrame (handles MultiIndex).

    Args:
        df: The Pandas DataFrame.
        column_name: The name of the column to find peaks in.
        prominence (optional): Minimum prominence of peaks.
        width (optional): Minimum width of peaks in samples.
        threshold (optional): Minimum threshold for peaks.
        negative (bool): If True, finds negative peaks (troughs).  Defaults to False (positive peaks).

    Returns:
        A tuple containing:
            - A list of peak indices (integers).
            - A dictionary of properties for each peak (prominences, widths, etc.) as returned by scipy.signal.find_peaks.
            Returns None, None if the input is invalid or the column doesn't exist.
    """

    if not isinstance(df, pd.DataFrame) or df.empty or column_name not in df.columns:
        return None, None

    data = df[('Close', 'AAPL')].values if column_name == 'Close' else df[('High', 'AAPL')].values if column_name == 'High' else df[('Low', 'AAPL')].values if column_name == 'Low' else df[('Open', 'AAPL')].values if column_name == 'Open' else df[('Volume', 'AAPL')].values if column_name == 'Volume' else None
    if data is None:
        return None, None

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


ticker = 'AAPL'
df = yf.download(ticker, period='12mo')

# Example 1: Find positive peaks in the 'Close' column
close_peak_indices, close_peak_properties, close_peak_values = find_peaks_in_column(df, 'Close', prominence=5, width=5)

if close_peak_indices is not None:
    print("Positive Peaks in 'Close' column indices:", close_peak_indices)
    print("Positive Peaks in 'Close' column properties:", close_peak_properties)

    plt.figure(figsize=(12, 6))
    plt.plot(df[('Close', 'AAPL')], label='Close')
    plt.plot(df.index[close_peak_indices], close_peak_values, 'ro', label='Positive Peaks')
    plt.title(f"Positive Peaks in {ticker} 'Close' Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("Could not find positive peaks in 'Close' column.")

# Example 2: Find negative peaks (troughs) in the 'Close' column
close_trough_indices, close_trough_properties, close_trough_values = find_peaks_in_column(df, 'Close', prominence=5, width=5, negative=True)

if close_trough_indices is not None:
    print("\nNegative Peaks (Troughs) in 'Close' column indices:", close_trough_indices)
    print("Negative Peaks (Troughs) in 'Close' column properties:", close_trough_properties)

    plt.figure(figsize=(12, 6))
    plt.plot(df[('Close', 'AAPL')], label='Close')
    plt.plot(df.index[close_trough_indices], close_trough_values, 'go', label='Negative Peaks (Troughs)')  # Use a different color
    plt.title(f"Negative Peaks (Troughs) in {ticker} 'Close' Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("Could not find negative peaks (troughs) in 'Close' column.")



# Example 3: Find positive peaks in the 'Volume' column (adjust prominence as needed)
volume_peak_indices, volume_peak_properties, volume_peak_values = find_peaks_in_column(df, 'Volume', prominence=50000000)

if volume_peak_indices is not None:
    print("\nPositive Peaks in 'Volume' column indices:", volume_peak_indices)
    print("Positive Peaks in 'Volume' column properties:", volume_peak_properties)

    plt.figure(figsize=(12, 6))
    plt.plot(df[('Volume', 'AAPL')], label='Volume')  # Correctly access the 'Volume' column
    plt.plot(df.index[volume_peak_indices], volume_peak_values, 'ro', label='Positive Peaks')
    plt.title(f"Positive Peaks in {ticker} 'Volume'")
    plt.xlabel("Date")
    plt.ylabel("Volume")
    plt.legend()
    plt.grid(True)
    plt.show()

else:
    print("Could not find positive peaks in 'Volume' column.")

# Example 4: Find negative peaks in the 'Volume' column (adjust prominence as needed)
volume_trough_indices, volume_trough_properties, volume_trough_values = find_peaks_in_column(df, 'Volume', prominence=50000000, negative=True)

if volume_trough_indices is not None:
    print("\nNegative Peaks (Troughs) in 'Volume' column indices:", volume_trough_indices)
    print("Negative Peaks (Troughs) in 'Volume' column properties:", volume_trough_properties)

    plt.figure(figsize=(12, 6))
    plt.plot(df[('Volume', 'AAPL')], label='Volume')  # Correctly access the 'Volume' column
    plt.plot(df.index[volume_trough_indices], volume_trough_values, 'go', label='Negative Peaks (Troughs)')
    plt.title(f"Negative Peaks (Troughs) in {ticker} 'Volume'")
    plt.xlabel("Date")
    plt.ylabel("Volume")
    plt.legend()
    plt.grid(True)
    plt.show()

else:
    print("Could not find negative peaks (troughs) in 'Volume' column.")
