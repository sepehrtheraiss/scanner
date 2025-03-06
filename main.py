# main.py
# controls the flow of program (ETL)
# fetch data -> transform -> plot
from argparse import ArgumentParser
from fetch import *
from trends import *
from plot import *
import json

parser = ArgumentParser(
                    prog='scanner',
                    description='Scans stocks for a pattern plus extra criteria',
                    epilog='let$ make $ome $$$')

# EXTRACT
parser.add_argument('-mag7', '--mag7', action="store_true")
parser.add_argument('-test', '--test', action="store_true")
parser.add_argument('-tick', '--ticker', type=str)
parser.add_argument('-sp500', '--sp500', action="store_true")
parser.add_argument('-f', '--file', type=str)
# LOAD
parser.add_argument('-showPlt', '--showPlt', action="store_true")
parser.add_argument('-savePlt', '--savePlt', action="store_true")
parser.add_argument('-onlyBreakout', '--onlyBreakout', action="store_true") # show&save only if breakout pattern has occured
parser.add_argument('-saveDataHistory', '--saveDataHistory', action="store_true")
parser.add_argument('-saveTickers', '--saveTickers', action="store_true")
# TRANSFORM
parser.add_argument('-o', '--order', help="default 5", type=int, default=5)
parser.add_argument('-p', '--period', help="default 3mo", type=str, default='3mo')
parser.add_argument('-i', '--interval', help="default 1d", type=str, default='1d')
parser.add_argument('-peaks', '--peaks', action="store_true")
parser.add_argument('-tr', '--trendline', action="store_true")
parser.add_argument('-c', '--confirm', action="store_true")
parser.add_argument('-fw', '--falling_wedge', action="store_true")
parser.add_argument('-sr', '--support_resistance', action="store_true")
parser.add_argument('-dg', '--debug', action="store_true")

args = parser.parse_args()

# FECTH
fetch = Fetch(args.period, args.interval)
data = {}

if args.test:
    data = fetch.test_data()

if args.mag7:
    data = fetch.mag_7()

if args.ticker:
   data = fetch.ticker(args.ticker)
    
if args.sp500:
    data = fetch.sp500() 

if args.file:
    data = fetch.file(args.file)

# TRANSFORM
# PLOT
order = args.order 
tickers = {} 
pattern = 'Falling Wedge' 
tickers['Falling Wedge'] = []
for ticker, df in data.items():

    plot = Plot(ticker, df, args.period, args.interval, ochl='Close', x=15, y=8)

    if args.peaks:
        min_idx, max_idx = get_min_max_idx(df, order) 
        plot.plot_min_max(min_idx, max_idx)

    close = df['Close'].values
    hh = get_higher_highs(close, order)
    hl = get_higher_lows(close, order)
    ll = get_lower_lows(close, order)
    lh = get_lower_highs(close, order)
    if args.trendline:
        plot.plot_trendlines(hh, hl, ll, lh)

    lines = []
    if args.falling_wedge:
        dates = df.index 
        lines = get_falling_wedge_lines(dates, close, lh, ll)
    if args.onlyBreakout:
        plot.plot_breakout(lines, 'Falling Wedge')
        #print('plot breakout')
        #print(lines)
        if lines and args.savePlt:
            plot.savePlot()
        if lines and args.showPlt:
            plot.showPlot()
        if lines and args.saveTickers:
            tickers['Falling Wedge'].append(ticker)
    else:
        plot.plot_breakout(lines, 'Falling Wedge')
        if args.savePlt:
            plot.savePlot()
        if args.showPlt:
            plot.showPlot()

if args.saveTickers:
    with open(f'{pattern}.json', 'w') as file:
        json.dump(tickers, file, indent=4)
