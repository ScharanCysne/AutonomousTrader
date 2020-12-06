import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

from Analysis import Analysis

'''
    Reinforcement Learning Trading Agent
    
    optimization parameters: theta(t) -> F(t) = tanh(theta(t)^T.x(t))
    
    F(t) indicates position in Asset (-1 to 1) at time t.

    TODO: Optimize parameters M, T
'''
class Agent:
    def __init__(self, epochs=1000, learning_rate=0.3, commission=0, capital=1000):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.commission = commission
        self.capital = capital

    '''
        Sharpe Ratio
    '''
    def sharpe_ratio(self, rets):
        return rets.mean() / rets.std()

    '''
        Ft(x^t.theta) position on the asset
    '''
    def positions(self, x, theta):
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
    def returns(self, Ft, x, delta):
        T = len(x)
        rets = Ft[0:T - 1] * x[1:T] - delta * np.abs(Ft[1:T] - Ft[0:T - 1])
        return np.concatenate([[0], rets])

    '''
        Gradient Ascent to maximize Sharpe Ratio
    '''
    def gradient(self, x, theta, delta):
        Ft = self.positions(x, theta)
        R = self.returns(Ft, x, delta)
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
    def train(self, x, epochs=2000, M=8, commission=0.0025, learning_rate = 0.3):
        theta = np.random.rand(M + 2)
        sharpes = np.zeros(epochs) # store sharpes over time
        for i in range(epochs):
            print(f"Training Epoch {i}")
            grad, sharpe = self.gradient(x, theta, commission)
            theta = theta + grad * learning_rate
            sharpes[i] = sharpe
        
        print("Finished Training")
        return theta, sharpes




