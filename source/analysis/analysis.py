'''
    Classe de Análise de Ativos

    Métricas calculadas:
        - Standard Deviation
        - Moving Average
        - Exponential Moving Average
        - Moving Average Convergence Divergence
        - Stochastic Oscillator

'''
import math
import pandas as pd
import matplotlib.pyplot as plt

class Analysis:
    def __init__(self, stock):
        self.stock = pd.read_csv(stock)
        self.analysis()    


    def analysis(self):
        # Standard Deviation/Volatility
        self.std = self.stock.std()
        #self.histogram_variation() # Normal
        #self.histogram_volume()    # Qui-Quadrado
        self.MA(1)
        self.EMA(1)
        self.MACD(1)
        self.StochasticOscillator(1)


    def histogram_variation(self):
        # Histogram of the data
        n, bins, patches = plt.hist(100*(self.stock["<CLOSE>"]-self.stock["<OPEN>"])/self.stock["<OPEN>"], 50, facecolor='b', alpha=0.75)

        plt.xlabel('Variation per Minute (%)')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of Variation in Price of {self.name} per Minute')
        plt.grid(True)
        plt.show()


    def histogram_volume(self):
        # Histogram of the data
        n, bins, patches = plt.hist(self.stock["<VOL>"], 5000, facecolor='b', alpha=0.75)

        plt.xlabel('Variation per Minute')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of Volume Traded of {self.name} per Minute')
        plt.axis([0,300000,0,6000])
        plt.grid(True)
        plt.show()


    def MA(self, t):
        # Common values: 15, 20, 30, 50, 100, and 200 days.
        self.stock[f'MA5'] = self.stock["<CLOSE>"].rolling(window=5*t,min_periods=0).mean()
        self.stock[f'MA8'] = self.stock["<CLOSE>"].rolling(window=8*t,min_periods=0).mean()
        self.stock[f'MA13'] = self.stock["<CLOSE>"].rolling(window=13*t,min_periods=0).mean()
        

    def EMA(self, t):
        # Common values: 10, 20, 30 and 50 days.
        self.stock[f'EMA20'] = self.stock["<CLOSE>"].ewm(span=20*t,adjust=False).mean()


    def MACD(self, t):
        # MACD = 12-Period EMA − 26-Period EMA
        stock_copy = self.stock.copy()     # Dont want to polute stock dataframe
        stock_copy['EMA12'] = stock_copy["<CLOSE>"].ewm(span=12*t,adjust=False).mean()
        stock_copy['EMA26'] = stock_copy["<CLOSE>"].ewm(span=26*t,adjust=False).mean()

        # Not suro is 100% correct - need to check
        self.stock["MACD"] = stock_copy['EMA12'] - stock_copy['EMA26']


    def RSI(self):
        pass


    def StochasticOscillator(self, t):
        # %K - Slow Stochastic Oscillator
        # %D - Fast Stochastic Oscillator

        stock_copy = self.stock.copy()     # Dont want to polute stock dataframe
        stock_copy["L14"] = stock_copy["<LOW>"].rolling(window=14*t,min_periods=0).min()
        stock_copy["H14"] = stock_copy["<HIGH>"].rolling(window=14*t,min_periods=0).max()
    
        self.stock['%K'] = 100*(stock_copy["<CLOSE>"] - stock_copy["<LOW>"])/(stock_copy["<HIGH>"] - stock_copy["<LOW>"]) 
        self.stock['%D'] = self.stock['%K'].rolling(window=3,min_periods=0).mean()