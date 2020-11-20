'''
    Classe de Análise de Ativos

    Métricas calculadas:
        - Voltatilidade
        - Média Móvel Curta (5 dias)
        - Média Móvel Longa (15 dias)
        - Sharpe Ratio
        - MACD
        - RSI (Relative Strength Index)

    Análises frequentistas apontam variação diária semelhante a curva normal.
'''
import math
import pandas as pd
import matplotlib.pyplot as plt

class Analysis:
    def __init__(self, stock=""):
        self.name = stock
        self.stock = pd.read_csv(f"./../data/{stock}.csv", sep="\t")
        self.analysis()    


    def analysis(self):
        # Standard Deviation/Volatility
        self.std = self.stock.std()
        #self.histogram_variation() # Normal
        #self.histogram_volume()    # Qui-Quadrado
        self.MA(402)


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
        # Common values: 15, 20, 30, 50, 100, and 200 days.
        # 1 Day == 402
        pass

    def MACD(self, t):
        # Common values: 15, 20, 30, 50, 100, and 200 days.
        # 1 Day == 402
        pass

    def RSI(self):
        pass

    def StochasticOscillator(self):
        pass

    def summary(self):
        print(f"Standard Deviation for {self.name}: \n {self.std}")