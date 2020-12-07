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
    def returns(self, Fts, xs, delta):
        rets_parc = [Fts[i][0:len(xs[i]) - 1] * xs[i][1:len(xs[i])] - delta * np.abs(Fts[i][1:len(xs[i])] - Fts[i][0:len(xs[i]) - 1]) for i in range(len(xs))]
        output = 0
        dist = rets_parc
        for arr in dist:
            output = np.add(output, arr)
        for i in range(len(dist)):
            dist[i] = np.divide(dist[i], output)
            np.nan_to_num(dist[i],  0.2)
        mi = [math.floor(self.capital*dist[i]/self.prices[i]) for i in range(len(xs))] 
        rets = rets_parc*mi
        
        return np.concatenate([[0], rets])

    '''
        Gradient Ascent to maximize Sharpe Ratio
    '''
    def gradient(self, xs, thetas, delta):
        Fts = self.positions(xs, thetas)
        R = self.returns(Fts, xs, delta)
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
                dFdtheta = (1 - Fts[i][t] ** 2) * (xt + thetas[i][-1] * dFpdtheta)
                dSdtheta = (dSdA * dAdR + dSdB * dBdR[t]) * (dRdF * dFdtheta + dRdFp * dFpdtheta)
                grad = grad + dSdtheta
                dFpdtheta = dFdtheta

            Ss[i] = S
            grads[i] = grad
        return grads, Ss


    '''
        Training Reinforment 
    '''
    def train(self, xs):
        thetas = [np.random.rand(self.M + 2) for x in xs]
        sharpes = np.zeros(self.epochs) # store sharpes over time
        for i in range(self.epochs):
            print(f"Training Epoch {i}")
            grads, sharpes = self.gradient(xs, thetas, self.commission)
            thetas = thetas + grads * self.learning_rate
            sharpes[i] = sharpes
            print(f"Epoch {i+1} finished: Sharpe = {sharpes}")        

        print("Finished Training")
        return thetas, sharpes