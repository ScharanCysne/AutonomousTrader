import mt5b3 as b3
import numpy.random as rand


'''
    Dummy Trader
'''
class DummyTrader(b3.Trader):
    def __init__(self):
        pass

    def setup(self,dbars):
        print('just getting started!')

    def trade(self,ops,dbars):
        orders=[] 
        assets=ops['assets']
        for asset in assets:
            if rand.randint(2)==1:     
                order=b3.buyOrder(asset,100)
            else:
            	order=b3.sellOrder(asset,100)
            orders.append(order)
        return orders
    
    def ending(self,dbars):
        print('Ending stuff')