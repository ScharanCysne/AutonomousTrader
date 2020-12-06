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
rets = np.array(stock['<CLOSE>'].diff()[1:])

# Separating Train and Test Samples
N = int(0.7 * len(rets))            # Number of Training Samples
P = len(rets) - N                  # Number of Test Samples
x_train = rets[:N]             # Training Samples
x_test = rets[-P:]             # Test Samples

std = np.std(x_train)
mean = np.mean(x_train)

x_train = (x_train - mean) / std    
x_test = (x_test - mean) / std

# Hyper-parameters
epochs = 2000
learning_rate = 0.3
commission = 0.0025
capital = 10000

# Reinforcement Learning Agent
tinyRL = Agent(epochs, learning_rate, commission, capital)
# Train Tiny RL in training samples
theta, sharpes = tinyRL.train(x_train)

# Check results from training and testing
train_returns = tinyRL.returns(tinyRL.positions(x_train, theta), x_train, 0.0025)
test_returns = tinyRL.returns(tinyRL.positions(x_test, theta), x_test, 0.0025)


# Plot Sharpe Ration Convergence
plt.figure()
plt.plot(sharpes)
plt.xlabel('Epoch Number')
plt.ylabel('Sharpe Ratio')
plt.show()

# Plot Training Returns
plt.figure()
plt.plot((train_returns).cumsum(), label="Reinforcement Learning Model", linewidth=1)
plt.plot(x_train.cumsum(), label="Buy and Hold", linewidth=1)
plt.xlabel('Ticks')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.title("RL Model vs. Buy and Hold - Training Data")
plt.show()

# Plot Test Returns
plt.figure()
plt.plot((test_returns).cumsum(), label="Reinforcement Learning Model", linewidth=1)
plt.plot(x_test.cumsum(), label="Buy and Hold", linewidth=1)
plt.xlabel('Ticks')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.title("RL Model vs. Buy and Hold - Test Data")
plt.show()