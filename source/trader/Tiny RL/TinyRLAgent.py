import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import math

from Analysis import Analysis

'''
    Reinforcement Learning Trading Agent
    
    optimization parameters: theta(t) -> F(t) = tanh(theta(t)^T.x(t))
    
    F(t) indicates position in Asset (-1 to 1) at time t.

    TODO: Optimize parameters M, T
'''
class Agent:
    def __init__(self, epochs=1000, learning_rate=0.3, commission=0, capital=1000, prices):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.commission = commission
        self.capital = capital
        self.prices = prices

    '''
        Sharpe Ratio
    '''
    def sharpe_ratio(self, rets):
        return rets.mean() / rets.std()

    '''
        Ft(x^t.theta) position on the asset
    '''
    def positions(self, xs, thetas):
        M = [len(theta) - 2 for theta in thetas]
        T = [len(x) for x in xs]
        Fts = [np.zeros(T[i]) for x in xs]
        for i in range(len(xs)):
            for t in range(M[i], T[i]):
                xt = np.concatenate([[1], xs[i][t - M:t], [Fts[i][t - 1]]])
                Fts[i][t] = np.tanh(np.dot(theta[i], xt))
        return Fts

    '''
        Returns on current asset allocation
    '''
    def returns(self, Fts, xs, delta, time):
        rets_parc = [Fts[i][0:len(xs[i]) - 1] * xs[i][1:len(xs[i])] - delta * np.abs(Fts[i][1:len(xs[i])] - Fts[i][0:len(xs[i]) - 1]) for i in range(len(xs))]
        dist = [np.mean(rets_parc[i])/np.sum(np.mean(rets_parc)) for i in range(len(xs))]
        mi = [math.floor(self.capital*dist[i]/price[i][time]) for i in range(len(xs))] 
        rets = np.dot(rets_parc,mi)
        return np.concatenate([[0], rets])

    '''
        Gradient Ascent to maximize Sharpe Ratio
    '''
    def gradient(self, xs, theta, delta, time):
        Ft = self.positions(xs, theta)
        R = self.returns(Ft, xs, delta, time)
        Ss = np.zeros(len(xs))
        grads = np.zeros(len(xs))
        for i in range(len(xs)):
            T = len(xs[i])
            M = len(thetas[i])
        
            A = np.mean(R[i])
            B = np.mean(np.square(R[i]))
            S = A / np.sqrt(B - A ** 2)

            dSdA = S * (1 + S ** 2) / A
            dSdB = -S ** 3 / 2 / A ** 2
            dAdR = 1. / T
            dBdR = 2. / T * R
        
            grad = np.zeros(M + 2)  # initialize gradient
            dFpdtheta = np.zeros(M + 2)  # for storing previous dFdtheta
        
            for t in range(M, T):
                xt = np.concatenate([[1], xs[i][t - M:t], [Fts[i][t-1]]])
                dRdF = -delta * np.sign(Fts[i][t] - Fts[i][t-1])
                dRdFp = xs[i][t] + delta * np.sign(Fts[i][t] - Fts[i][t-1])
                dFdtheta = (1 - Fts[i][t] ** 2) * (xt + theta[i][-1] * dFpdtheta)
                dSdtheta = (dSdA * dAdR + dSdB * dBdR[t]) * (dRdF * dFdtheta + dRdFp * dFpdtheta)
                grad = grad + dSdtheta
                dFpdtheta = dFdtheta

            Ss[i] = S
            grads[i] = grad
        return grads, Ss

    '''
        Training Reinforment 
    '''
    def train(self, xs, epochs=2000, M=8, commission=0.0025, learning_rate = 0.3):
        thetas = [np.random.rand(M + 2) for x in xs]
        sharpes = np.zeros(epochs) # store sharpes over time
        for i in range(epochs):
            print(f"Training Epoch {i}")
            grads, sharpes = self.gradient(xs, thetas, commission, i)
            thetas = thetas + grads * learning_rate
            sharpes[i] = sharpe
        
        print("Finished Training")
        return theta, sharpes




