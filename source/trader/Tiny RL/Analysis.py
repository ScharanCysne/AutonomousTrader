import math
import pandas as pd
import matplotlib.pyplot as plt

class Analysis:
    def __init__(self, stock):
        self.stock = stock
        self.analysis()    


    def analysis(self):
        # Standard Deviation/Volatility
        self.std = self.stock.std()
        self.MA(402)
        self.EMA(402)
        self.MACD(402)
        self.StochasticOscillator(402)


    def MA(self, t):
        # Common values: 15, 20, 30, 50, 100, and 200 days.
        # 1 Day == 402
        self.stock[f'MA{math.floor(t/402)}'] = self.stock["<CLOSE>"].rolling(window=t,min_periods=0).mean()
        

    def EMA(self, t):
        # Common values: 10, 20, 30 and 50 days.
        # 1 Day == 402
        self.stock[f'EMA{math.floor(t/402)}'] = self.stock["<CLOSE>"].ewm(span=t,adjust=False).mean()


    def MACD(self, t):
        # MACD = 12-Period EMA − 26-Period EMA
        # 1 Day == 402
        stock_copy = self.stock.copy()     # Dont want to polute stock dataframe
        stock_copy['EMA12'] = stock_copy["<CLOSE>"].ewm(span=12*t,adjust=False).mean()
        stock_copy['EMA26'] = stock_copy["<CLOSE>"].ewm(span=26*t,adjust=False).mean()

        # Not suro is 100% correct - need to check
        self.stock["MACD"] = stock_copy['EMA12'] - stock_copy['EMA26']


    def StochasticOscillator(self, t):
        # %K - Slow Stochastic Oscillator
        # %D - Fast Stochastic Oscillator

        stock_copy = self.stock.copy()     # Dont want to polute stock dataframe
        stock_copy["L14"] = stock_copy["<LOW>"].rolling(window=14*t,min_periods=0).min()
        stock_copy["H14"] = stock_copy["<HIGH>"].rolling(window=14*t,min_periods=0).max()
    
        self.stock['%K'] = 100*(stock_copy["<CLOSE>"] - stock_copy["<LOW>"])/(stock_copy["<HIGH>"] - stock_copy["<LOW>"]) 
        self.stock['%D'] = self.stock['%K'].rolling(window=3,min_periods=0).mean()
        