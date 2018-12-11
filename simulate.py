import tensorflow as tf
import pandas as pd
import numpy as np
import random
from collections import deque
import pickle
import math

bitstamp = pd.read_csv('bitstampUSD_1-min_data_2012-01-01_to_2018-06-27.csv')
bitstamp.drop(labels='Timestamp', axis=1, inplace=True)
bitstamp = bitstamp[2500000:]
bitstamp = np.array(bitstamp)

import tensorflow.keras as k
from tensorflow.keras.layers import *
from tensorflow.keras.models import model_from_json

with open('model','r') as f:
    json = f.readline()

model = model_from_json(json)

class Trader:
    def __init__(self, max_turns=60*24, fee_rate=.0005):
        self.action_space = [i for i in range(5)]
        self.max_turns = max_turns
        self.fee_rate = fee_rate
        self.reset()
        
    def reset(self):
        self.t = 0
        self.cash = 100000
        self.coins = 0
        self.time = random.randrange(60*24*60, len(bitstamp) - self.max_turns)
        self.done = False
        s = self.t+self.time, self.coins, self.cash
        return s

    def step(self, action):
        if self.t == self.max_turns:
            self.done = True
        price_open = bitstamp[self.time + self.t][0]
        price_close = bitstamp[self.time + self.t][3]
        prev_asset = price_open * self.coins + self.cash
        percent = (action-2)*.2
        if percent < 0:
            coins_sold = self.coins * (-percent)
            self.cash += coins_sold * price_open * (1-self.fee_rate)
            self.coins -= coins_sold

        elif percent > 0:
            money_spent = self.cash * percent 
            self.coins += money_spent * (1-self.fee_rate) / price_open
            self.cash -= money_spent
            
        r = price_close * self.coins + self.cash - prev_asset
        s = self.t+self.time, self.coins, self.cash
        self.asset = price_close * self.coins + self.cash
        if self.asset < 50000:
            self.done = True
            
        self.t += 1
        return s, r, self.done

class Simulator():
    def __init__(self,
                 model=model,
                 n_episodes=100, n_win_ticks=195, 
                 gamma=.9999, epsilon=.01, alpha=0.01, alpha_decay=0.01, 
                 batch_size=64):
        self.env = Trader(max_turns=len(bitstamp)-60*24*60-1)
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.n_episodes = n_episodes
        self.n_win_ticks = n_win_ticks
        self.batch_size = batch_size

        self.model = model
        #global graph
        #graph = tf.get_default_graph()
        
    def preprocess_state(self, s):
        time, coins, cash = s
        unzip = np.append(bitstamp[time-60*24*30 : time].flatten(), [coins, cash])
        state = unzip.reshape(1,60*24*30*7+2) # 1 batch of input
        return state                                                        
    def choose_action(self, state, epsilon):
        state = self.preprocess_state(state)
        #with graph.as_default():
        pred = self.model.predict(state)
        return random.choice(self.env.action_space) if (np.random.random() <= epsilon) else np.argmax(pred[0])

    def run(self):
        state = self.env.reset()
        done = False
        rewards = 0
        t=0
        while not done:
            action = self.choose_action(state, self.epsilon)
            next_state, reward, done = self.env.step(action)
            state = next_state
            rewards += reward
            t += 1
            print(self.env.asset, rewards)

agent = Simulator()
agent.run()