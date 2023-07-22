#import gym
#import json
#import datetime as dt

# from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
#import talib as ta
from env.StockTradingEnv import StockTradingEnv
import torch

import pandas as pd

df = pd.read_csv('./data/train_stock.csv')
df.Tic = df.Tic.astype('category').cat.codes
df.Tic = (df.Tic - min(df.Tic)) / (max(df.Tic) - min(df.Tic))
df[['Open', 'High', 'Low', "Close", 'Volume']] = df[['Open', 'High', 'Low', "Close", 'Volume']].astype('float64')
df = df.sort_values('Date')
df.dropna(inplace=True)
df = df.sort_values('Date')
df = df.reset_index(drop=True)

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: StockTradingEnv(df)])
#env = StockTradingEnv(df)

torch.autograd.set_detect_anomaly(True)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=2000)

obs = env.reset()
for i in range(2000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
