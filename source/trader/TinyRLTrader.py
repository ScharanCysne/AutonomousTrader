import mt5b3 as b3
import numpy as np
import numpy.random as rand

class TinyRLTrader(b3.Trader):
    def __init__(self):
        self.theta = [
            1.22017626, 
            0.28236144, 
            0.90629668, 
            0.43899471, 
            0.93361272, 
            0.62647585, 
            0.64547422, 
            1.82452996, 
            0.07376972, 
            1.15452448
        ]

        self.stocks = []
        self.rets = {}
        self.prices = {}
        self.means = {}
        self.stds = {}

    '''
        Receives daily bars for each stock for setup
    '''
    def setup(self, dbars):
        for key, value in dbars.items() :
            self.stocks.append(key)
            self.prices[key] = value.close.values.tolist()
            self.means[key] = np.mean(self.prices[key])
            self.stds[key] = np.std(self.prices[key])
            self.rets[key] = (value.close.diff()[1:].values.tolist() - self.means[key])/self.stds[key]


    '''
        Receives daily bars and current 
    '''
    def trade(self, ops, dbars):

        orders = [] 
        assets = ops['assets']
        limit_cap = ops['capital']/len(assets)
        self.capital = ops['capital']

        for key, value in dbars.items() :
            self.prices[key].append(value.close.values.tolist()[-1])
            self.rets[key] = np.concatenate([self.rets[key], [value.close.values.tolist()[-1] - value.close.values.tolist()[-2]]]) 
            self.rets[key][-1] = (self.rets[key][-1] - self.means[key])/self.stds[key]

        for asset in assets:
            Ft = self.positions(self.rets[asset], self.theta)
            position = limit_cap * Ft
            number_of_shares = int(round(position/self.prices[asset][-1]) - ops[f"shares_{asset}"])
        
            # Make orders
            if number_of_shares > 0:     
                order = b3.buyOrder(asset, number_of_shares)
            else:
            	order = b3.sellOrder(asset, np.abs(number_of_shares))
        
            orders.append(order)

        return orders

    '''
        Ft(x^t.theta) position on the asset
    '''
    def positions(self, x, theta):
        M = len(theta) - 2
        T = len(x)
        Ft = np.zeros(T)

        for t in range(M, T):
            xt = np.concatenate([[1], x[t - M:t], [Ft[t - 1]]])
            Ft[t] = np.tanh(np.dot(theta, xt))
        return Ft[-1]
    
    def ending(self, dbars):
        print(f'Portfolio final balance: {self.capital}')