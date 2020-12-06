import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

'''
    Sharpe Ratio
'''
def sharpe_ratio(rets):
    return rets.mean() / rets.std()

'''
    Ft(x^t.theta) position on the asset
'''
def positions(x, theta):
    M = len(theta) - 2
    T = len(x)
    Ft = np.zeros(T)
    for t in range(M, T):
        xt = np.concatenate([[1], x[t - M:t], [Ft[t - 1]]])
        Ft[t] = np.tanh(np.dot(theta, xt))
    return Ft

'''
    Returns on current asset allocation
'''
def returns(Ft, x, delta):
    T = len(x)
    rets = Ft[0:T - 1] * x[1:T] - delta * np.abs(Ft[1:T] - Ft[0:T - 1])
    return np.concatenate([[0], rets])

'''
    Gradient Ascent to maximize Sharpe Ratio
'''
def gradient(x, theta, delta):
    Ft = positions(x, theta)
    R = returns(Ft, x, delta)
    T = len(x)
    M = len(theta) - 2
    
    A = np.mean(R)
    B = np.mean(np.square(R))
    S = A / np.sqrt(B - A ** 2)

    dSdA = S * (1 + S ** 2) / A
    dSdB = -S ** 3 / 2 / A ** 2
    dAdR = 1. / T
    dBdR = 2. / T * R
    
    grad = np.zeros(M + 2)  # initialize gradient
    dFpdtheta = np.zeros(M + 2)  # for storing previous dFdtheta
    
    for t in range(M, T):
        xt = np.concatenate([[1], x[t - M:t], [Ft[t-1]]])
        dRdF = -delta * np.sign(Ft[t] - Ft[t-1])
        dRdFp = x[t] + delta * np.sign(Ft[t] - Ft[t-1])
        dFdtheta = (1 - Ft[t] ** 2) * (xt + theta[-1] * dFpdtheta)
        dSdtheta = (dSdA * dAdR + dSdB * dBdR[t]) * (dRdF * dFdtheta + dRdFp * dFpdtheta)
        grad = grad + dSdtheta
        dFpdtheta = dFdtheta

        
    return grad, S

'''
    Training Reinforment 
'''
def train(x, epochs=2000, M=8, commission=0.0025, learning_rate = 0.3):
    theta = np.random.rand(M + 2)
    sharpes = np.zeros(epochs) # store sharpes over time
    for i in range(epochs):
        print(f"Training Epoch {i}")
        grad, sharpe = gradient(x, theta, commission)
        theta = theta + grad * learning_rate

        sharpes[i] = sharpe
    
    print("Finished Training")
    return theta, sharpes

#stocks = ["BBAS3", "BBDC4", "ITUB4", "PETR4", "VALE3"]
#portfolio = np.array([pd.read_csv(f"input/{stock}.csv", sep="\t") for stock in stocks], dtype=object)
#rets = [stock['<CLOSE>'].diff()[1:] for stock in portfolio]

# Load stock to trains TinyRL
stock = pd.read_csv("input/BBDC4.csv", sep="\t")
rets = stock['<CLOSE>'].diff()[1:]
x = np.array(rets)

N = int(4*len(x)/5)         # Number of Training Samples
P = len(x) - N              # Number of Test Samples
x_train = x[-(N+P):-P]      # Training Samples
x_test = x[-P:]             # Test Samples
std = np.std(x_train)
mean = np.mean(x_train)

x_train = (x_train - mean) / std    
x_test = (x_test - mean) / std

# Train Tiny RL in training samples
np.random.seed(0)
theta, sharpes = train(x_train, epochs=2000, M=8, commission=0.0025, learning_rate=0.3)

# Plot Sharpe Ration Convergence
plt.figure()
plt.plot(sharpes)
plt.xlabel('Epoch Number')
plt.ylabel('Sharpe Ratio')
plt.show()

# Plot Training Returns
train_returns = returns(positions(x_train, theta), x_train, 0.0025)
plt.figure()
plt.plot((train_returns).cumsum(), label="Reinforcement Learning Model", linewidth=1)
plt.plot(x_train.cumsum(), label="Buy and Hold", linewidth=1)
plt.xlabel('Ticks')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.title("RL Model vs. Buy and Hold - Training Data")
plt.show()

# Plot Test Returns
test_returns = returns(positions(x_test, theta), x_test, 0.0025)
plt.figure()
plt.plot((test_returns).cumsum(), label="Reinforcement Learning Model", linewidth=1)
plt.plot(x_test.cumsum(), label="Buy and Hold", linewidth=1)
plt.xlabel('Ticks')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.title("RL Model vs. Buy and Hold - Test Data")
plt.show()