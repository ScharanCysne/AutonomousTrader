from datetime import datetime 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import backtrader as bt

from Strategy import Strategy
from Momentum import Momentum

def daily_variations(ticker):
    stock = pd.read_csv(f"input/{ticker}.csv", sep="\t", index_col='<DATE>', parse_dates=True)['<CLOSE>'].rename(ticker)
    stock = stock.groupby('<DATE>').tail(1)
    return stock

def momentum(closes):
    returns = np.log(closes)
    x = np.arange(len(returns))
    slope, _, rvalue, _, _ = linregress(x, returns)
    return ((1 + slope) ** 252) * (rvalue ** 2)  # annualize slope and multiply by R^2

# Load Data and create concatenated pandas
tickers = ["BBAS3","BBDC4","ITUB4","PETR4","VALE3"]
portfolio = [daily_variations(ticker) for ticker in tickers]

stocks = pd.concat([stock for stock in portfolio],axis=1, sort=True)
stocks = stocks.loc[:,~stocks.columns.duplicated()]
stocks = stocks.dropna()
# Verify if Ok
print(stocks)

# Calculate momentum of last 90 days
momentums = stocks.copy(deep=True)
for ticker in tickers:
    momentums[ticker] = stocks[ticker].rolling(window=90).apply(momentum, raw=False)
    momentums[ticker] = stocks[ticker].fillna(0)

# Select best momentum
bests = momentums.max().sort_values(ascending=False).index[:5]

# Create Cerebro
cerebro = bt.Cerebro(stdstats=False)
cerebro.broker.set_coc(True)

for ticker in tickers:
    df = daily_variations(ticker)
    if len(df) > 100: # data must be long enough to compute 100 day SMA
        cerebro.adddata(bt.feeds.PandasData(dataname=df, plot=False))

cerebro.addobserver(bt.observers.Value)
cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.0)
cerebro.addanalyzer(bt.analyzers.Returns)
cerebro.addanalyzer(bt.analyzers.DrawDown)
cerebro.addstrategy(Strategy)
results = cerebro.run()

cerebro.plot(iplot=False)[0][0]
print(f"Sharpe: {results[0].analyzers.sharperatio.get_analysis()['sharperatio']:.3f}")
print(f"Norm. Annual Return: {results[0].analyzers.returns.get_analysis()['rnorm100']:.2f}%")
print(f"Max Drawdown: {results[0].analyzers.drawdown.get_analysis()['max']['drawdown']:.2f}%")