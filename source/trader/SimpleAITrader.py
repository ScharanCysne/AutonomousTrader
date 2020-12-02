import numpy.random as rand
import mt5b3 as b3
import time
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.preprocessing import KBinsDiscretizer


'''
    Multi Asset Trader
''' 
class SimpleAITrader(b3.Trader):

    def setup(self,dbars):
        assets=list(dbars.keys())
        if len(assets)!=1:
            print('Error, this trader is supposed to deal with just one asset')
            return None
        bars=dbars[assets[0]]
        timeFrame=10 # it takes into account the last 10 bars
        horizon=1 # it project the closing price for next bar
        target='close' # name of the target column
        ds=b3.ai_utils.bars2Dataset(bars,target,timeFrame,horizon)

        X=b3.ai_utils.fromDs2NpArrayAllBut(ds,['target'])
        discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform') 

        ds['target']=b3.ai_utils.discTarget(discretizer,ds['target'])
        Y=b3.ai_utils.fromDs2NpArray(ds,['target'])

        clf = tree.DecisionTreeClassifier()

        clf = clf.fit(X, Y)
        self.clf=clf

    def trade(self,bts,dbars):
            assets=dbars.keys()
            orders=[]
            timeFrame=10 # it takes into account the last 10 bars
            horizon=1 # it project the closing price for next bar
            target='close' # name of the target column
            for asset in assets:
                curr_shares=b3.backtest.getShares(asset)
                money=b3.backtest.getBalance()/len(assets) # divide o saldo em dinheiro igualmente entre os ativos
                free_shares=b3.backtest.getAfforShares(asset,money,dbars)
                # get new information (bars), transform it in X
                bars=dbars[asset]
                #remove irrelevant info
                del bars['time']
                # convert from bars to dataset
                ds=b3.ai_utils.bars2Dataset(bars,target,timeFrame,horizon)
                # Get X fields
                X=b3.ai_utils.fromDs2NpArrayAllBut(ds,['target'])

                # predict the result, using the latest info
                p=self.clf.predict([X[-1]])
                if p==2:
                    #buy it
                    order=b3.buyOrder(asset,free_shares)
                elif p==0:
                    #sell it
                    order=b3.sellOrder(asset,curr_shares)
                else:
                    order=None
                if order!=None:
                    orders.append(order)
            return orders    