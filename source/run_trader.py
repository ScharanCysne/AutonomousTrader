import mt5b3 as b3
import pandas as pd
import time
import numpy as np

from trader.DummyTrader import DummyTrader

connected=b3.connect()
if connected:
    print('Ok!! It is connected to B3 exchange!!')
else:
    print('Something went wrong! It is NOT connected to B3!!')

# Create Trader
trader=DummyTrader()
assets=['PETR4','VALE3']

# sets Backtest options 
prestart=b3.date(2020,1,20)
start=b3.date(2020,1,21)
end=b3.date(2020,11,30)
capital=100000
results_file='data_equity_file.csv'
verbose=False           # Use True if you want debug information for your Trader 
#sets the backtest setup
period=b3.DAILY 
 # it may be b3.INTRADAY (one minute interval)
bts = b3.backtest.set(assets,prestart,start,end,period,capital,results_file,verbose)
if b3.backtest.checkBTS(bts): # check if the backtest setup is ok!
    print('Backtest Setup is Ok!')
else:
    print('Backtest Setup is NOT Ok!')

# Running the backtest
df = b3.backtest.run(trader,bts)   
# run calls the Trader. setup and trade (once for each bar)

#print the results
#print(df)

# Evaluate Trader Bot
b3.backtest.evaluate(df)