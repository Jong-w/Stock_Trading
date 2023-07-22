import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np


MAX_ACCOUNT_BALANCE = 782243419
MAX_NUM_SHARES = 782243419
MAX_SHARE_PRICE = 5000
MAX_STEPS = 200000

INITIAL_ACCOUNT_BALANCE = 10000


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""

    def __init__(self, df):
        super(StockTradingEnv, self).__init__()

        self.df = df
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(7, 6), dtype=np.float16)

    def _next_observation(self):
        # Get the stock data points for the last 5 days and scale to between 0-1
        frame = np.array([
            self.df['Tic'][[self.current_step, self.current_step+2000, self.current_step+4000, self.current_step + 6000, self.current_step+8000, self.current_step+10000]],
            self.df['Open'][[self.current_step, self.current_step + 2000, self.current_step + 4000, self.current_step +
                             6000, self.current_step + 8000, self.current_step + 10000]].values / max(self.df.Open),
            self.df['High'][[self.current_step, self.current_step + 2000, self.current_step + 4000, self.current_step +
                             6000, self.current_step + 8000, self.current_step+10000]].values / max(self.df.High),
            self.df['Low'][[self.current_step, self.current_step + 2000, self.current_step + 4000, self.current_step +
                            6000, self.current_step + 8000, self.current_step+10000]].values / max(self.df.Low),
            self.df['Close'][[self.current_step, self.current_step + 2000, self.current_step + 4000, self.current_step +
                              6000, self.current_step + 8000, self.current_step+10000]].values / max(self.df.Close),
            self.df['Volume'][[self.current_step, self.current_step + 2000, self.current_step + 4000, self.current_step +
                              6000, self.current_step + 8000, self.current_step+10000]].values / max(self.df.Volume)
        ])

        # Append additional data and scale each value to between 0-1
        # print(self.current_step)
        # print(frame.shape)
        obs1 = np.append(frame, [[
            self.balance / MAX_ACCOUNT_BALANCE, #key1
            self.max_net_worth / MAX_ACCOUNT_BALANCE,
            self.cost_basis / MAX_SHARE_PRICE,
            self.shares_held / MAX_NUM_SHARES, # key2
            self.total_shares_sold / MAX_NUM_SHARES,
            self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
        ]], axis=0)
        obs =obs1
        return obs

    def _take_action(self, action):
        # Set the current price to a random price within the time step
        current_price = random.uniform(self.df.loc[self.current_step, "Open"], self.df.loc[self.current_step, "Close"]) + 1e-7

        action_type = action[0]
        amount = action[1] + 1e-7

        if action_type <= 1:
            # Buy amount % of balance in shares
            total_possible = int(self.balance / current_price)
            shares_bought = int(total_possible * amount)
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * current_price

            self.balance -= additional_cost
            self.cost_basis = (prev_cost + additional_cost) / (self.shares_held + shares_bought + 1e-7)
            self.shares_held += shares_bought

        elif action_type <= 2:
            # Sell amount % of shares held
            shares_sold = int(self.shares_held * amount)
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * current_price

        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth


        if self.shares_held == 0:
            self.cost_basis = 0

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1

        if self.current_step > len(self.df.loc[:, 'Open'].values) - 10001:
            self.current_step = 0

        if self.max_net_worth >= MAX_ACCOUNT_BALANCE:
            self.max_net_worth = MAX_ACCOUNT_BALANCE
        if self.balance >= MAX_ACCOUNT_BALANCE:
            self.balance = MAX_ACCOUNT_BALANCE
        if self.cost_basis >= MAX_SHARE_PRICE:
            self.cost_basis = MAX_SHARE_PRICE
        if self.shares_held >= MAX_NUM_SHARES:
            self.shares_held = MAX_NUM_SHARES
        if self.total_shares_sold >= MAX_NUM_SHARES:
            self.total_shares_sold = MAX_NUM_SHARES
        if self.total_sales_value >= (MAX_NUM_SHARES * MAX_SHARE_PRICE):
            self.total_sales_value = (MAX_NUM_SHARES * MAX_SHARE_PRICE)

        delay_modifier = (self.current_step / MAX_STEPS)

        reward = self.balance * delay_modifier
        done = (self.balance == MAX_ACCOUNT_BALANCE) or (self.max_net_worth == MAX_ACCOUNT_BALANCE) or (self.net_worth <= 0) \
               or (self.cost_basis == MAX_SHARE_PRICE) or (self.shares_held == MAX_NUM_SHARES) or (self.total_shares_sold == MAX_NUM_SHARES) \
               or (self.total_sales_value == (MAX_NUM_SHARES * MAX_SHARE_PRICE))



        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        # Set the current step to a random point within the data frame
        self.current_step = 0
        #self.current_step = random.randint(0, len(self.df.loc[:, 'Open'].values) - 2000)

        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        print(f'Step: {self.current_step}')
        #print(f'Balance: {self.balance}')
        #print(f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        #print(f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')
