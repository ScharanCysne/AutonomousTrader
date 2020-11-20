import pandas as pd 
import matplotlib.pyplot as plt

from analysis.analysis import Analysis

# Portfolio Construction
stocks = ["BBAS3", "BBDC4", "ITUB4", "PETR4", "VALE3"]
portfolio = [Analysis(stock) for stock in stocks]

# Summary
for stock in portfolio:
    stock.summary()