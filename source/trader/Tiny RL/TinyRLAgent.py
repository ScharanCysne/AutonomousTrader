import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import math

'''
    Reinforcement Learning Trading Agent
    
    optimization parameters: theta(t) -> F(t) = tanh(theta(t)^T.x(t))
    
    F(t) indicates position in Asset (-1 to 1) at time t.

    TODO: Optimize parameters M, T
'''
class Agent:
    def __init__(self, epochs, learning_rate, commission, capital, prices):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.commission = commission
        self.capital = capital
        self.prices = prices
        self.M = 8

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
        Fts = [np.zeros(T[i]) for i in range(len(xs))]
        for i in range(len(xs)):
            for t in range(M[i], T[i]):
                xt = np.concatenate([[1], xs[i][t - M[i]:t], [Fts[i][t - 1]]])
                Fts[i][t] = np.tanh(np.dot(thetas[i], xt))
        return Fts

    '''
        Returns on current asset allocation
    '''
    def returns(self, Fts, xs, prices, delta):
        rets_parc = [Fts[i][0:len(xs[i]) - 1] * xs[i][1:len(xs[i])] - delta * np.abs(Fts[i][1:len(xs[i])] - Fts[i][0:len(xs[i]) - 1]) for i in range(len(xs))]
        dist = rets_parc
        
        # Calculating portfolio distribuiton        
        output = 0
        for arr in dist:
            output = np.add(output, np.abs(arr[:len(xs[0])]))
        for i in range(len(dist)):
            dist[i] = np.concatenate([[0], np.divide(dist[i], output)])
            dist[i][np.isnan(dist[i])] = 0.2
            dist[i] = dist[i][:len(prices[0])]
        # Calculating capital invested each time
        mi = [np.floor(self.capital*np.divide(dist[i],prices[i])) for i in range(len(xs))] 
        
        rets = [0] * len(rets_parc)
        for i in range(len(rets_parc)):
            rets[i] = rets_parc[i]*mi[i]
        
        return rets

    '''
        Gradient Ascent to maximize Sharpe Ratio
    '''
    def gradient(self, xs, thetas, delta):
        Fts = self.positions(xs, thetas)
        Rs = self.returns(Fts, xs, self.prices, delta)
        Ss = np.zeros(len(xs))
        grads = []

        for i in range(len(xs)):
            T = len(xs[i])

            A = np.mean(Rs[i])
            B = np.mean(np.square(Rs[i]))
            S = A / np.sqrt(B - A ** 2)

            dSdA = S * (1 + S ** 2) / A
            dSdB = -S ** 3 / 2 / A ** 2
            dAdR = 1. / T
            dBdR = 2. / T * Rs[i]
        
            grad = np.zeros(self.M + 2)          # initialize gradient
            dFpdtheta = np.zeros(self.M + 2)     # for storing previous dFdtheta
        
            for t in range(self.M, T):
                xt = np.concatenate([[1], xs[i][t - self.M:t], [Fts[i][t-1]]])
                dRdF = -delta * np.sign(Fts[i][t] - Fts[i][t-1])
                dRdFp = xs[i][t] + delta * np.sign(Fts[i][t] - Fts[i][t-1])
                dFdtheta = (1 - Fts[i][t] ** 2) * (xt + thetas[i][-1] * dFpdtheta)
                dSdtheta = (dSdA * dAdR + dSdB * dBdR[t]) * (dRdF * dFdtheta + dRdFp * dFpdtheta)
                grad = grad + dSdtheta
                dFpdtheta = dFdtheta

            Ss[i] = S
            grads.append(grad)
        return grads, Ss


    '''
        Training Reinforment 
    '''
    def train(self, xs):
        thetas = [np.random.rand(self.M + 2) for x in xs]
        sharpes = []                # store sharpes over time
        for i in range(self.epochs):
            grads, sharp = self.gradient(xs, thetas, self.commission)
            thetas = thetas + np.multiply(grads, self.learning_rate)
            sharpes.append(sharp)
            print(f"Epoch {i+1} finished: Sharpes = {sharpes[i][0]}, {sharpes[i][1]}, {sharpes[i][2]}, {sharpes[i][3]}, {sharpes[i][4]}")        

        print("Finished Training")
        return thetas, sharpes