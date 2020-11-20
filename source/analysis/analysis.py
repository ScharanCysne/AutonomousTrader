'''
    Classe de Análise de Ativos

    Métricas calculadas:
        - Standard Deviation
        - Moving Average
        - Exponential Moving Average
        - Moving Average Convergence Divergence
        - Stochastic Oscillator
        - Relative Strength Index

    Análises frequentistas apontam variação diária semelhante a curva normal.
'''
import math
import pandas as pd
import matplotlib.pyplot as plt

class Analysis:
    def __init__(self, stock=""):
        self.name = stock
        self.stock = pd.read_csv(f"./../data/{stock}.csv", sep="\t")
        self.daily = self.daily_variations()
        self.analysis()    


    def daily_variations(self):
        pass


    def analysis(self):
        # Standard Deviation/Volatility
        self.std = self.stock.std()
        #self.histogram_variation() # Normal
        #self.histogram_volume()    # Qui-Quadrado
        self.MA(402)
        self.EMA(402)
        self.MACD(402)
        self.StochasticOscillator(402)


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
        
    def summary(self):
        print(self.stock)
        print(f"Standard Deviation for {self.name}: \n {self.std}")