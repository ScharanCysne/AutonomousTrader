import numpy.random as rand
import mt5b3 as b3
import time
import pandas as pd
import numpy as np


'''
    Mono Asset Trader
'''
class MonoAssetTrader(b3.Trader):
    def trade(self,bts,dbars):
        assets=dbars.keys()
        asset=list(assets)[0]
        orders=[]
        bars=dbars[asset]
        curr_shares=b3.backtest.getShares(bts,asset)
        # number of shares that you can buy
        free_shares=b3.backtest.getAfforShares(bts,dbars,asset)
        rsi=b3.tech.rsi(bars)
        if rsi>=70:   
            order=b3.buyOrder(asset,free_shares)
        else:
            order=b3.sellOrder(asset,curr_shares)
        if rsi>=70 and free_shares>0: 
            order=b3.buyOrder(asset,free_shares)
        elif  rsi<70 and curr_shares>0:
            order=b3.sellOrder(asset,curr_shares)
        if order!=None:
                orders.append(order)
        return orders    
