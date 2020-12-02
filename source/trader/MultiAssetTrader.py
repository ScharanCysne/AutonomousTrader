import numpy.random as rand
import mt5b3 as b3
import time
import pandas as pd
import numpy as np


'''
    Multi Asset Trader
'''
class MultiAssetTrader(b3.Trader):
    def trade(self,bts,dbars):
        assets=dbars.keys()
        orders=[]
        for asset in assets:
            bars=dbars[asset]
            curr_shares=b3.backtest.getShares(bts,asset)
            money=b3.backtest.getBalance(bts)/len(assets) # divide o saldo em dinheiro igualmente entre os ativos
            # number of shares that you can buy of asset 
            free_shares=b3.backtest.getAfforShares(bts,dbars,asset,money)
            rsi=b3.tech.rsi(bars)
            if rsi>=70 and free_shares>0: 
                order=b3.buyOrder(asset,free_shares)
            elif  rsi<70 and curr_shares>0:
                order=b3.sellOrder(asset,curr_shares)
            else:
                order=None
            if order!=None:
                orders.append(order)
        return orders    
