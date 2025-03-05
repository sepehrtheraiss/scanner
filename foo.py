import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import yfinance as yf

ticker = 'AAPL'
df = yf.download(ticker, period='3mo', interval='1h', multi_level_index=False)
prices = pd.DataFrame(df.Close.values)

# 1. Smoothing (Example: 3-day moving average)
prices_smooth = prices.rolling(window=3).mean()

# 2. Handle NaNs (Important!)
prices_smooth = prices_smooth.fillna(method='bfill') # Backfill for the beginning
prices_smooth = prices_smooth.dropna() # Drop any remaining NaNs

prices_smooth_np = prices_smooth.values.reshape(-1)  # Key change: reshape(-1)

# 3. Invert data (for finding valleys/lower highs)
inverted_prices = -prices_smooth_np

# 4. Find peaks (which are valleys in the inverted data)
peaks, properties = find_peaks(inverted_prices, distance=5, prominence=0.5)  # Adjust parameters

# 5. Post-processing: Identify Lower Highs
lower_highs_indices = []
lower_highs_values = []

if peaks.size > 0:  # Check if any peaks were found
    previous_peak_value = -inverted_prices[peaks[0]]  # Get the first peak's actual price
    lower_highs_indices.append(peaks[0])
    lower_highs_values.append(previous_peak_value)

    for i in range(1, len(peaks)):
        current_peak_value = -inverted_prices[peaks[i]]  # Get the current peak's actual price
        if current_peak_value < previous_peak_value:
            lower_highs_indices.append(peaks[i])
            lower_highs_values.append(current_peak_value)
            previous_peak_value = current_peak_value

lower_highs_indices = np.array(lower_highs_indices)
lower_highs_values = np.array(lower_highs_values)

# ... (Now you have lower_highs_indices and lower_highs_values) ...

# Plotting (Example - Corrected for DatetimeIndex)
import matplotlib.pyplot as plt

# Get the dates corresponding to the lower highs
lower_highs_dates = prices.index[lower_highs_indices]

plt.plot(prices.index, prices, label="Original Prices")
plt.plot(lower_highs_dates, lower_highs_values, "o", color="red", label="Lower Highs") # Use lower_highs_dates
plt.legend()
plt.title("Lower Highs Detection")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()

