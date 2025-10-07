# plot.py
# plots lines, scatter, savesplot
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

class Plot:
    def __init__(self, ticker, df, period, interval, ochl='Close', order=6, x=15, y=8):
        self.ticker = ticker
        self.df = df
        self.period = period
        self.interval = interval
        self.fig = plt.figure(figsize=(x, y),facecolor='darkblue')
        #self.fig.set_facecolor('xkcd:salmon')
        plt.style.use('dark_background')
        self.ochl = ochl 
        plt.title(f'{ticker}-{period}-{interval} {ochl} Price order {order}')
        self.colors = {
            'support' : 'yellow',
            'resistance': 'red',
            'HH' : 'blue',
            'HL' : 'gray',
            'LL' : 'purple',
            'LH' : 'orange',
            'breakout' : 'red',
            'peak-up' : 'blue',
            'peak-down' :'purple',
             ochl: 'white',
        }
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        self.legend_elements = [
            Line2D([0], [0], color=self.colors[ochl], label=ochl),
        ]
        self.trendLine2D = [
          Line2D([0], [0], color=self.colors['HH'], label='Higher Highs'),
          Line2D([0], [0], color=self.colors['HL'], label='Higher Lows'),
          Line2D([0], [0], color=self.colors['LL'], label='Lower Lows'),
          Line2D([0], [0], color=self.colors['LH'], label='Lower Highs'),
        ]
        self.confirmLine2D = [
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
        
    def savePlot(self):
        plt.legend(handles=self.legend_elements)
        self.fig.savefig(f'plots/{self.ticker}-{self.period}-{self.interval}')

    def showPlot(self):
        plt.plot(self.df[self.ochl], color=self.colors[self.ochl], zorder=0)
        plt.legend(handles=self.legend_elements)
        plt.show()

    def plot_min_max(self, min_idx, max_idx):
        df = self.df

        plt.scatter(df.iloc[max_idx].index, df.iloc[max_idx][self.ochl],
            label='Maxima', s=100, color=self.colors['peak-up'], marker='^')
        plt.scatter(df.iloc[min_idx].index, df.iloc[min_idx][self.ochl],
            label='Minima', s=100, color=self.colors['peak-down'], marker='v')

        plt.plot(df.iloc[max_idx].index, df.iloc[max_idx]['Close'],
               label='Maxima', color=colors[1])
        plt.plot(df.iloc[min_idx].index, df.iloc[min_idx]['Close'],
               label='Minima', color=colors[2])

        plt.title(f'Local Maxima and Minima for {self.ticker}')

    def plot_trendlines(self, hh, hl, ll, lh):
        dates = self.df.index
        self.legend_elements.extend(self.trendLine2D)
        _ = [plt.plot(dates[i], self.df[self.ochl].iloc[i], c=self.colors['HH']) for i in hh]
        _ = [plt.plot(dates[i], self.df[self.ochl].iloc[i], c=self.colors['HL']) for i in hl]
        _ = [plt.plot(dates[i], self.df[self.ochl].iloc[i], c=self.colors['LL']) for i in ll]
        _ = [plt.plot(dates[i], self.df[self.ochl].iloc[i], c=self.colors['LH']) for i in lh]
    
    def plot_breakout(self, lines, pattern):
        self.legend_elements.append(Line2D([0], [0], color=self.colors['breakout'], label=pattern))
        for line in lines:
            xh_line, yh_line = line[0]
            xl_line, yl_line = line[1]
            plt.plot(xh_line, yh_line, color=self.colors['breakout'])
            plt.plot(xl_line, yl_line, color=self.colors['breakout'])
"""
    def plot_support_resistance(self, support, resistance, n=5):

        plt.scatter(self.df.iloc[support].index, self.df.iloc[support][self.ohcl],
            label='support', s=100, color=self.colors['support'], marker='^')

        plt.scatter(self.df.iloc[resistance].index, self.df.iloc[resistance][self.ohcl],
            label='resistance', s=100, color=self.colors['resistance'], marker='v')

        dates = self.df.index

        s1 = support[-1] 
        s2 = self.df[s1] + timedelta(days=30) 
        plt.plot([dates[s1],s2], [df.iloc[s1]['Close'], self.df.iloc[s1]['Close']],
            label='support', color='black')

        r1 = resistance[-1] 
        r2 = dates[r1] + timedelta(days=30) 
        plt.plot([dates[r1],r2], [df.iloc[resistance[-1]]['Close'],df.iloc[resistance[-1]]['Close']],
            label='resistance', color='black')

"""





