import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from Analysis import Analysis
from TinyRLAgent import Agent

#stocks = ["BBAS3", "BBDC4", "ITUB4", "PETR4", "VALE3"]
#portfolio = np.array([pd.read_csv(f"input/{stock}.csv", sep="\t") for stock in stocks], dtype=object)
#rets = [stock['<CLOSE>'].diff()[1:] for stock in portfolio]

# Load stock to trains TinyRL
stock = pd.read_csv("input/ITUB4.csv", sep="\t")
stock = stock.dropna()
close = np.array(stock['<CLOSE>'].copy())

# Separating Train and Test Samples
N = int(0.7 * len(close))            # Number of Training Samples
P = len(close) - N                  # Number of Test Samples
close_train = close[:N]             # Training Samples
close_test = close[-P:]             # Test Samples

# Hyper-parameters
epochs = 2000
learning_rate = 0.3
commission = 0.0025
capital = 10000

# Reinforcement Learning Agent
analyst = Analysis(stock)
tinyrl = Agent(capital, analyst)
# Train Tiny RL in training samples
theta, sharpes = tinyrl.train(close_train, epochs=2000, M=8, commission=0.0025, learning_rate=0.3)

# Check results from training and testing
train_returns = tinyrl.returns(tinyrl.positions(close_train, theta), close_test, 0.0025)
test_returns = tinyrl.returns(tinyrl.positions(close_test, theta), close_test, 0.0025)


# Plot Sharpe Ration Convergence
plt.figure()
plt.plot(sharpes)
plt.xlabel('Epoch Number')
plt.ylabel('Sharpe Ratio')
plt.show()

# Plot Training Returns
plt.figure()
plt.plot((train_returns).cumsum(), label="Reinforcement Learning Model", linewidth=1)
plt.plot(close_train.cumsum(), label="Buy and Hold", linewidth=1)
plt.xlabel('Ticks')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.title("RL Model vs. Buy and Hold - Training Data")
plt.show()

# Plot Test Returns
plt.figure()
plt.plot((test_returns).cumsum(), label="Reinforcement Learning Model", linewidth=1)
plt.plot(close_test.cumsum(), label="Buy and Hold", linewidth=1)
plt.xlabel('Ticks')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.title("RL Model vs. Buy and Hold - Test Data")
plt.show()