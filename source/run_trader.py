import mt5b3 as b3
import pandas as pd
import time
import numpy as np

from trader.DummyTrader import DummyTrader

if b3.connect():
    print('Status: Connected to B3 Exchange')
else:
    print('Status: Something went wrong!')

# Create Trader and define assets
trader = DummyTrader()
assets = ['PETR4','VALE3','BBDC4','ITUB4','BBAS3']

# sets Backtest options 
prestart = b3.date(2020,1,20)
start = b3.date(2020,1,21)
end = b3.date(2020,11,30)
capital = 100000
results_file = 'data_equity_file'
verbose = False             # Use True if you want debug information for your Trader 
period=b3.DAILY             # it may be b3.INTRADAY (one minute interval)

#sets the backtest setup
bts = b3.backtest.set(assets, prestart, start, end, period, capital, results_file, verbose)
if b3.backtest.checkBTS(bts): # check if the backtest setup is ok!
    print('Backtest Setup: Ok! \n')
else:
    print('Backtest Setup: Something went wrong! \n')

# Running the backtest
df = b3.backtest.run(trader, bts)   
# run calls the Trader. setup and trade (once for each bar)

# Print the results
# print(df)

# Evaluate Trader Bot
b3.backtest.evaluate(df)