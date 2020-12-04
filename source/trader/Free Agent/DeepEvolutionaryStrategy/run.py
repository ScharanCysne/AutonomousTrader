import numpy as np
import pandas as pd

from Agent import Agent
from Model import Model

# Deep Evolutionary Strategy

# Load data
df = pd.read_csv('input/ITUB4.csv', sep="\t")
x = df['<CLOSE>'].values.tolist()

N = int(len(x)/5)           # Number of Training Samples
P = len(x) - N              # Number of Test Samples
x_train = x[:N]      # Training Samples
x_test = x[-P:]             # Test Samples

### Hyper-Parameters 
input_size = 30
layer_size = 500
output_size = 3

# Market parameters
capital = 10000
max_buy = 5
max_sell = 5

# Simulation parameters
iterations = 500
checkpoint = 10

# Fit and Evaluation
model = Model(input_size, layer_size, output_size)
agent = Agent(model, x_train, capital, max_buy, max_sell)
agent.fit(iterations, checkpoint)
agent.buy()
