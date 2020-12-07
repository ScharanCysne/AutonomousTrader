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
        grad, sharpe = gradient(x, theta, commission)
        theta = theta + grad * learning_rate
        sharpes[i] = sharpe
        print(f"Epoch {i+1} finished: Sharpe = {sharpe}")        
    
    print("finished training")
    return theta, sharpes

#stocks = ["BBAS3", "BBDC4", "ITUB4", "PETR4", "VALE3"]
#portfolio = np.array([pd.read_csv(f"input/{stock}.csv", sep="\t") for stock in stocks], dtype=object)
#rets = [stock['<CLOSE>'].diff()[1:] for stock in portfolio]

name = "VALE3"

# Load stock to trains TinyRL
stock = pd.read_csv(f"input/{name}.csv", sep="\t")
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
'''
# Train Tiny RL in training samples
np.random.seed(0)
theta, sharpes = train(x_train, epochs=200, M=8, commission=0.0025, learning_rate=0.3)

with open('fittest.txt', 'w') as f:
    print('Fittest Thetas Data.\n', file=f)
    print(f"Thetas optimized {name}: {theta}", file=f)
    print(f"Sharpes associated: {sharpes[-1]}\n", file=f)

# Plot Sharpe Ration Convergence
plt.figure()
plt.plot(sharpes)
plt.xlabel('Epoch Number')
plt.ylabel('Sharpe Ratio')
plt.show()
'''

theta = [0.00636868, 0.99213169, 0.96710875, 1.03625064, 0.90010168, 0.93073166, 0.02889846, 0.0408688,  0.18620258, 1.4629834]

# Plot Training Returns
train_returns = returns(positions(x_train, theta), x_train, 0.0025)
plt.figure()
plt.plot(np.multiply((train_returns).cumsum(),std), label="Reinforcement Learning Model", linewidth=1)
plt.plot(np.multiply(x_train.cumsum(),std), label="Buy and Hold", linewidth=1)
plt.xlabel('Ticks')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.title("RL Model vs. Buy and Hold - Training Data")
plt.show()

# Plot Test Returns
test_returns = returns(positions(x_test, theta), x_test, 0.0025)
plt.figure()
plt.plot(np.multiply((test_returns).cumsum(),std), label="Reinforcement Learning Model", linewidth=1)
plt.plot(np.multiply(x_test.cumsum(),std), label="Buy and Hold", linewidth=1)
plt.xlabel('Ticks')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.title("RL Model vs. Buy and Hold - Test Data")
plt.show()

'''
    r(t) = z(t)/z(t-1) - 1
    1.09713744 0.60594239 0.59348084 0.54156889 0.46499409 0.59890041 0.51110019 0.82695253 0.83205858 0.84740365
    1.10547124 0.60358536 0.59118186 0.54112218 0.46525614 0.59642542 0.51784977 0.82749494 0.82801721 0.85438729
    1.10148825 0.60214377 0.59195473 0.54417474 0.466907   0.59521412 0.51975879 0.82717124 0.8254497  0.85528494
    1.09312483 0.60564405 0.58744487 0.54238416 0.47314902 0.6016878 0.51622663 0.82144101 0.82993403 0.85242226

    r(t) = (z(t) - z(t-1))/std(r)
    0.00636868 0.99213169 0.96710875 1.03625064 0.90010168 0.93073166 0.02889846 0.0408688  0.18620258 1.4629834
'''