import yfinance as yf
import mplfinance as mpf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils import *
from patternpy import tradingpatterns as tp

 
# tv 
df = yf.download('AAPL', start="2023-06-20", end="2024-07-30", interval='1d', multi_level_index=False)
#df = yf.download('CARR', start="2024-11-20", end="2025-02-10", interval='1d', multi_level_index=False)
# latest
#df = yf.download('AAPL', start="2024-12-01", end="2025-03-01", interval='1d', multi_level_index=False)
#mpf.plot(df,type='candle')

# schwab fallign wedge
#df = yf.download('AAPL', start="2024-9-20", end="2024-11-15", interval='1d', multi_level_index=False)
#uptrend
#df = yf.download('AAPL', start="2024-11-15", end="2024-12-20", interval='1d', multi_level_index=False)

w = tp.detect_wedge(pd.DataFrame(df))
sp = tp.calculate_support_resistance(pd.DataFrame(df))
trend = tp.detect_trendline(pd.DataFrame(df))
print(trend)
pivs = tp.find_pivots(pd.DataFrame(df))
HH = pivs[pivs['signal'] == 'HH']
HL = pivs[pivs['signal'] == 'HL']
LH = pivs[pivs['signal'] == 'LH']
LL = pivs[pivs['signal'] == 'LL']

fig, axs = plt.subplots(2)
fig.set_label('APPL pivots')

axs[0].set_title('PatternPy')
axs[0].plot(df.Close, label='AAPL Close')
axs[0].plot(HH.Close, 'go', label='HH')
axs[0].plot(HL.Close, 'ro', label='HL')
axs[0].plot(LH.Close, 'bo', label='LH')
axs[0].plot(LL.Close, 'yo', label='LL')
axs[0].legend()

close_peak_indices, close_peak_properties, close_peak_values = utils_find_peaks(df.Close.values, prominence=2, width=2)
axs[1].set_title('In House')
axs[1].plot(df.Close, label="AAPL Close")
axs[1].plot(df.index[close_peak_indices], close_peak_values, 'go', label='Positive Peaks')
close_peak_indices, close_peak_properties, close_peak_values = utils_find_peaks(df.Close.values, prominence=2, width=2, negative=True)
axs[1].plot(df.index[close_peak_indices], close_peak_values, 'ro', label='Negative Peaks')
axs[1].legend()


plt.show()



