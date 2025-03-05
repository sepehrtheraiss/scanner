# https://raposa.trade/blog/higher-highs-lower-lows-and-calculating-price-trends-in-python/

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from scipy.signal import argrelextrema
from matplotlib.lines import Line2D
from datetime import timedelta
from argparse import ArgumentParser

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

from collections import deque
def get_consecutive_peaks():
    # Get K consecutive higher peaks
    K = 2
    high_idx = argrelextrema(data['Close'].values, np.greater, order=5)[0]
    highs = data.iloc[high_idx]['Close']

    extrema = []
    ex_deque = deque(maxlen=K)
    for i, idx in enumerate(high_idx):
      if i == 0:
        ex_deque.append(idx)
        continue
      if highs[i] < highs[i-1]:
        ex_deque.clear()

      ex_deque.append(idx)
      if len(ex_deque) == K:
        # K-consecutive higher highs found
        extrema.append(ex_deque.copy())
    return extrema

def plot_cons_peaks(data):
    close = data['Close'].values
    dates = data.index
    extrema = get_consecutive_peaks()
    plt.figure(figsize=(15, 8))
    plt.plot(data['Close'])
    _ = [plt.plot(dates[i], close[i], c=colors[1]) for i in extrema]
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.title(f'Higher Highs for {ticker} Closing Price')
    plt.legend(['Close', 'Consecutive Highs'])
    plt.show()

def plot_max_min(order):
    '''
    data['local_max'] = data['Close'][
      (data['Close'].shift(1) < data['Close']) &
      (data['Close'].shift(-1) < data['Close'])]

    data['local_min'] = data['Close'][
      (data['Close'].shift(1) > data['Close']) &
      (data['Close'].shift(-1) > data['Close'])]
    '''
    max_idx = argrelextrema(data['Close'].values, np.greater, order=order)[0]
    min_idx = argrelextrema(data['Close'].values, np.less, order=order)[0]
    plt.figure(figsize=(15, 8))
    plt.plot(data['Close'], zorder=0)
    '''
    plt.scatter(data.index, data['local_max'], s=100,
      label='Maxima', marker='^', c=colors[1])
    plt.scatter(data.index, data['local_min'], s=100,
      label='Minima', marker='v', c=colors[2])
    '''
    plt.scatter(data.iloc[max_idx].index, data.iloc[max_idx]['Close'],
        label='Maxima', s=100, color=colors[1], marker='^')
    plt.scatter(data.iloc[min_idx].index, data.iloc[min_idx]['Close'],
        label='Minima', s=100, color=colors[2], marker='v')

    #plt.xlabel('Date')
    #plt.ylabel('Price ($)')
    #plt.title(f'Local Maxima and Minima for {ticker}')
    plt.legend()
    plt.show()

def getHigherLows(data: np.array, order=5, K=2):
    '''
    Finds consecutive higher lows in price pattern.
    Must not be exceeded within the number of periods indicated by the width 
    parameter for the value to be confirmed.
    K determines how many consecutive lows need to be higher.
    '''
    # Get lows
    low_idx = argrelextrema(data, np.less, order=order)[0]
    lows = data[low_idx]
    # Ensure consecutive lows are higher than previous lows
    extrema = []
    ex_deque = deque(maxlen=K)
    for i, idx in enumerate(low_idx):
        if i == 0:
          ex_deque.append(idx)
          continue
        if lows[i] < lows[i-1]:
          ex_deque.clear()

        ex_deque.append(idx)
        if len(ex_deque) == K:
          extrema.append(ex_deque.copy())

    return extrema

def getLowerHighs(data: np.array, order=5, K=2):
    '''
    Finds consecutive lower highs in price pattern.
    Must not be exceeded within the number of periods indicated by the width 
    parameter for the value to be confirmed.
    K determines how many consecutive highs need to be lower.
    '''
    # Get highs
    high_idx = argrelextrema(data, np.greater, order=order)[0]
    highs = data[high_idx]
    # Ensure consecutive highs are lower than previous highs
    extrema = []
    ex_deque = deque(maxlen=K)
    for i, idx in enumerate(high_idx):
        if i == 0:
          ex_deque.append(idx)
          continue
        if highs[i] > highs[i-1]:
          ex_deque.clear()

        ex_deque.append(idx)
        if len(ex_deque) == K:
          extrema.append(ex_deque.copy())

    return extrema

def getHigherHighs(data: np.array, order=5, K=2):
    '''
    Finds consecutive higher highs in price pattern.
    Must not be exceeded within the number of periods indicated by the width 
    parameter for the value to be confirmed.
    K determines how many consecutive highs need to be higher.
    '''
    # Get highs
    high_idx = argrelextrema(data, np.greater, order=5)[0]
    highs = data[high_idx]
    # Ensure consecutive highs are higher than previous highs
    extrema = []
    ex_deque = deque(maxlen=K)
    for i, idx in enumerate(high_idx):
        if i == 0:
          ex_deque.append(idx)
          continue
        if highs[i] < highs[i-1]:
          ex_deque.clear()

        ex_deque.append(idx)
        if len(ex_deque) == K:
          extrema.append(ex_deque.copy())

    return extrema

def getLowerLows(data: np.array, order=5, K=2):
    '''
    Finds consecutive lower lows in price pattern.
    Must not be exceeded within the number of periods indicated by the width 
    parameter for the value to be confirmed.
    K determines how many consecutive lows need to be lower.
    '''
    # Get lows
    low_idx = argrelextrema(data, np.less, order=order)[0]
    lows = data[low_idx]
    # Ensure consecutive lows are lower than previous lows
    extrema = []
    ex_deque = deque(maxlen=K)
    for i, idx in enumerate(low_idx):
        if i == 0:
          ex_deque.append(idx)
          continue
        if lows[i] > lows[i-1]:
          ex_deque.clear()

        ex_deque.append(idx)
        if len(ex_deque) == K:
          extrema.append(ex_deque.copy())

    return extrema

    
def detect_trendline(df, window=2):
    # Define the rolling window
    roll_window = window
    # Create new columns for the linear regression slope and y-intercept
    df['slope'] = np.nan
    df['intercept'] = np.nan

    for i in range(window, len(df)):
        x = np.array(range(i-window, i))
        y = df['Close'][i-window:i]
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        df.at[df.index[i], 'slope'] = m
        df.at[df.index[i], 'intercept'] = c

    # Create a boolean mask for trendline support
    mask_support = df['slope'] > 0

    # Create a boolean mask for trendline resistance
    mask_resistance = df['slope'] < 0

    # Create new columns for trendline support and resistance
    df['support'] = np.nan
    df['resistance'] = np.nan

    # Populate the new columns using the boolean masks
    df.loc[mask_support, 'support'] = df['Close'] * df['slope'] + df['intercept']
    df.loc[mask_resistance, 'resistance'] = df['Close'] * df['slope'] + df['intercept']

    return df

def calculate_slope_and_intercept(x1, y1, x2, y2):
    """
    Calculates the slope (m) and y-intercept (b) for a line.

    Args:
        x1: Timestamp object for the first x-coordinate.
        y1: Float for the first y-coordinate.
        x2: Timestamp object for the second x-coordinate.
        y2: Float for the second y-coordinate.

    Returns:
        A tuple: (slope, intercept), or None if the run is zero.
    """

    # Calculate the rise (change in y)
    rise = y2 - y1

    # Calculate the run (change in x)
    run = (x2 - x1).total_seconds()

    if run == 0:
        return None  # Slope and intercept are undefined (vertical line)
    else:
        # Calculate the slope (m)
        slope = rise / run

        # Calculate the intercept (b) using y = mx + b  => b = y - mx
        # Convert x1 timestamp to seconds since the start time.
        x1_seconds = (x1 - x1).total_seconds() # this is always 0.
        x2_seconds = (x2-x1).total_seconds() # this is the run
        
        intercept1 = y1 - slope * x1_seconds
        intercept2 = y2 - slope * x2_seconds

        #intercept1 and intercept2 should be the same, if not, there is an error.
        # to prevent floating point errors, take the average.
        intercept = (intercept1 + intercept2) / 2

        #return ([rise, run], intercept)
        return (slope, intercept)

def draw_cont_line(x1, y1, x2, y2, num_points=100):
    """
    Draws a continuous line based on two given points using y = mx + b.

    Args:
        x1: Timestamp object for the first x-coordinate.
        y1: Float for the first y-coordinate.
        x2: Timestamp object for the second x-coordinate.
        y2: Float for the second y-coordinate.
        num_points: Number of points to generate for the line.
    """
    result = calculate_slope_and_intercept(x1, y1, x2, y2)

    if result:
        slope, intercept = result
        #slope = slope[0]/slope[1]

        # Generate x-values (timestamps) for the line
        start_time = min(x1, x2)
        end_time = max(x1, x2)
        time_diff = (end_time - start_time) / (num_points - 1)
        # forward line
        x1_line = [start_time - i * time_diff for i in range(num_points)]
        # backward line
        x2_line = [end_time + i * time_diff for i in range(num_points)]

        # Convert timestamps to seconds relative to start time for calculations
        x1_line_seconds = [(t - start_time).total_seconds() for t in x1_line]
        x2_line_seconds = [(t - start_time).total_seconds() for t in x2_line]

        # Calculate y-values for the line
        #y_line = [slope * seconds + intercept for seconds in x_line_seconds]
        y1_line = [slope * seconds + y1 for seconds in x1_line_seconds]
        y2_line = [slope * seconds + y1 for seconds in x2_line_seconds]
        #breakpoint()

        return [x1_line+x2_line, y1_line+y2_line]
    else:
        print("Cannot draw a line (vertical line).")
        return []

def search_falling_wedge(dates, close, LH, LL):
    lines = []
    for idx in range(min(len(LH), len(LL))):
        x1h, y1h = dates[LH[idx]][0], close[LH[idx]][0]
        x2h, y2h = dates[LH[idx]][1], close[LH[idx]][1]

        x1l, y1l = dates[LL[idx]][0], close[LL[idx]][0]
        x2l, y2l = dates[LL[idx]][1], close[LL[idx]][1]

        slopeh, inter = calculate_slope_and_intercept(x1h, y1h, x2h, y2h)
        slopel, inter = calculate_slope_and_intercept(x1l, y1l, x2l, y2l)
        # upper slope needs to be more slanted than lower slope
        # at least twice as much
        #if slopeh < slopel:
        if slopeh / slopel > 2:
            if args.debug:
                print('LH: ', slopeh)
                print('LL: ', slopel)
            xh_line, yh_line = draw_cont_line(x1h, y1h, x2h, y2h)
            xl_line, yl_line = draw_cont_line(x1l, y1l, x2l, y2l)
            lines.append([[xh_line, yh_line], [xl_line, yl_line]])

    return lines 


parser = ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')

parser.add_argument('-o', '--order', help="default 5", type=int, default=5)
parser.add_argument('-p', '--peaks', action="store_true")
parser.add_argument('-m', '--movement', action="store_true")
parser.add_argument('-c', '--confirm', action="store_true")
parser.add_argument('-fw', '--falling_wedge', action="store_true")
parser.add_argument('-mag7', '--mag7', action="store_true")
parser.add_argument('-test', '--test', action="store_true")
parser.add_argument('-tick', '--ticker', type=str)
parser.add_argument('-sp500', '--sp500', action="store_true")
parser.add_argument('-plta', '--plot_all', action="store_true", default=False)
parser.add_argument('-dg', '--debug', action="store_true", default=False)
args = parser.parse_args()

MAG_7 = ['AAPL', 'AMZN', 'GOOGL', 'NVDA', 'MSFT', 'META', 'TSLA']
#yfObj = yf.Ticker(ticker)
#data = yfObj.history(start='2010-01-01', end='2010-07-01')

data = {}

if args.test:
    #data = yf.download(ticker, start="2023-07-21", end="2024-05-17", interval='1d', multi_level_index=False)
    data['AAPL'] = pd.read_csv('AAPL_2023-07-21_2024-05-17_1d',index_col=0, parse_dates=True)

if args.mag7:
    for ticker in MAG_7:
        data[ticker] = yf.download(ticker, period="12mo", interval='1d', multi_level_index=False)

if args.ticker:
    data[args.ticker] = yf.download(args.ticker, period="3mo", interval='1d', multi_level_index=False)
    
if args.sp500:
    sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    tickers = sp500['Symbol'].tolist()
    for ticker in tickers[:100]:
        data[ticker] = yf.download(ticker, period="3mo", interval='1d', multi_level_index=False)

order = args.order 
K = 2

if args.peaks:
    plot_max_min(order)
    exit()

legend_elements = [
    Line2D([0], [0], color=colors[0], label='Close'),
    Line2D([0], [0], color=colors[0], label=f'Order {order}'),
]

if args.movement:
    movementLine2D = [
      Line2D([0], [0], color=colors[1], label='Higher Highs'),
      Line2D([0], [0], color=colors[2], label='Higher Lows'),
      Line2D([0], [0], color=colors[3], label='Lower Lows'),
      Line2D([0], [0], color=colors[4], label='Lower Highs'),
    ]
    legend_elements.extend(movementLine2D)

if args.falling_wedge:
    legend_elements.append(Line2D([0], [0], color='black', label='Falling wedge'))

if args.confirm:
    confirmLine2D = [
      Line2D([0], [0], color='w',  marker='^',
             markersize=10,
             markerfacecolor=colors[1],
             label='Higher High Confirmation'),
      Line2D([0], [0], color='w',  marker='^',
             markersize=10,
             markerfacecolor=colors[2],
             label='Higher Lows Confirmation'),
      Line2D([0], [0], color='w',  marker='v',
             markersize=10,
             markerfacecolor=colors[3],
             label='Lower Lows Confirmation'),
      Line2D([0], [0], color='w',  marker='v',
             markersize=10,
             markerfacecolor=colors[4],
             label='Lower Highs Confirmation')
    ]
    legend_elements.extend(confirmLine2D)

for ticker, df in data.items():
    close = df['Close'].values
    dates = df.index

    hh = getHigherHighs(close, order, K)
    hl = getHigherLows(close, order, K)
    ll = getLowerLows(close, order, K)
    lh = getLowerHighs(close, order, K)


    if args.movement:
        _ = [plt.plot(dates[i], close[i], c=colors[1]) for i in hh]
        _ = [plt.plot(dates[i], close[i], c=colors[2]) for i in hl]
        _ = [plt.plot(dates[i], close[i], c=colors[3]) for i in ll]
        _ = [plt.plot(dates[i], close[i], c=colors[4]) for i in lh]


    if args.debug:
        print(ticker)

    if args.falling_wedge:
        lines = search_falling_wedge(dates, close, lh, ll)
        if lines:
            plt.figure(figsize=(15, 8))
            plt.plot(df['Close'])
            plt.xlabel('Date')
            plt.ylabel('Price ($)')
            plt.title(f'{ticker} Closing Price')
            for line in lines:
                xh_line, yh_line = line[0]
                xl_line, yl_line = line[1]
                plt.plot(xh_line, yh_line, c='black')
                plt.plot(xl_line, yl_line, c='black')
            plt.legend(handles=legend_elements)
            plt.show()
            

    if args.confirm:
        _ = [plt.scatter(dates[i[-1]] + timedelta(order), close[i[-1]], 
            c=colors[1], marker='^', s=100) for i in hh]
        _ = [plt.scatter(dates[i[-1]] + timedelta(order), close[i[-1]], 
            c=colors[2], marker='^', s=100) for i in hl]
        _ = [plt.scatter(dates[i[-1]] + timedelta(order), close[i[-1]], 
            c=colors[3], marker='v', s=100) for i in ll]
        _ = [plt.scatter(dates[i[-1]] + timedelta(order), close[i[-1]],
            c=colors[4], marker='v', s=100) for i in lh]

    if args.plot_all:
        pass
        #plt.legend(handles=legend_elements)
        #plt.show()
