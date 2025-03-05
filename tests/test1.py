import yfinance as yf
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def find_peaks_in_column(df, column_name, prominence=None, width=None, threshold=None):
    """Finds peaks in a specified column of a Pandas DataFrame (handles MultiIndex)."""

    if not isinstance(df, pd.DataFrame) or df.empty or column_name not in df.columns:
        return None, None

    # Access the column correctly using tuple notation for MultiIndex
    data = df[('Close', 'AAPL')].values if column_name == 'Close' else df[('High', 'AAPL')].values if column_name == 'High' else df[('Low', 'AAPL')].values if column_name == 'Low' else df[('Open', 'AAPL')].values if column_name == 'Open' else df[('Volume', 'AAPL')].values if column_name == 'Volume' else None

    if data is None:
        return None, None

    peaks, properties = find_peaks(data, prominence=prominence, width=width, threshold=threshold)

    return peaks.tolist(), properties


ticker = 'AAPL'
df = yf.download(ticker, period='12mo')

# Example 1: Find peaks in the 'Close' column
close_peak_indices, close_peak_properties = find_peaks_in_column(df, 'Close', prominence=5, width=5)

if close_peak_indices is not None:
    print("Peaks in 'Close' column indices:", close_peak_indices)
    print("Peaks in 'Close' column properties:", close_peak_properties)

    plt.figure(figsize=(12, 6))
    plt.plot(df[('Close', 'AAPL')], label='Close') # Correctly access the 'Close' column
    plt.plot(df.index[close_peak_indices], df[('Close', 'AAPL')][close_peak_indices], 'ro', label='Peaks')
    plt.title(f"Peaks in {ticker} 'Close' Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("Could not find peaks in 'Close' column.")



# Example 2: Find peaks in the 'Volume' column (adjust prominence as needed)
volume_peak_indices, volume_peak_properties = find_peaks_in_column(df, 'Volume', prominence=50000000)

if volume_peak_indices is not None:
    print("\nPeaks in 'Volume' column indices:", volume_peak_indices)
    print("Peaks in 'Volume' column properties:", volume_peak_properties)

    plt.figure(figsize=(12, 6))
    plt.plot(df[('Volume', 'AAPL')], label='Volume')  # Correctly access the 'Volume' column
    plt.plot(df.index[volume_peak_indices], df[('Volume', 'AAPL')][volume_peak_indices], 'ro', label='Peaks')
    plt.title(f"Peaks in {ticker} 'Volume'")
    plt.xlabel("Date")
    plt.ylabel("Volume")
    plt.legend()
    plt.grid(True)
    plt.show()

else:
    print("Could not find peaks in 'Volume' column.")


# Example 3: Find peaks in the 'High' column
high_peak_indices, high_peak_properties = find_peaks_in_column(df, 'High', prominence=5, width=5)

if high_peak_indices is not None:
    print("\nPeaks in 'High' column indices:", high_peak_indices)
    print("Peaks in 'High' column properties:", high_peak_properties)

    plt.figure(figsize=(12, 6))
    plt.plot(df[('High', 'AAPL')], label='High') # Correctly access the 'High' column
    plt.plot(df.index[high_peak_indices], df[('High', 'AAPL')][high_peak_indices], 'ro', label='Peaks')
    plt.title(f"Peaks in {ticker} 'High' Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("Could not find peaks in 'High' column.")
