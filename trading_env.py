import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gym import spaces

class TradingEnvironment(gym.Env):
    def __init__(self, data_path='data/formatted-result/dataset.csv'):
        super(TradingEnvironment, self).__init__()
        
        # Load the dataset
        self.data = pd.read_csv(data_path)
        self.current_step = 0
        self.total_steps = len(self.data)
        
        # Compute SL/TP adjustments before training
        self._apply_sl_tp_refactor()
        
        # Define action space (multipliers for SL & TP adjustments)
        self.action_space = spaces.Box(
            low=np.array([0.5, 0.5, 0.5, 0.5]),
            high=np.array([2.0, 2.0, 2.0, 2.0]),
            dtype=np.float32
        )
        
        # Define observation space (delta, premium, strike_price, bought_price)
        self.observation_space = spaces.Box(
            low=np.array([-1, 0, 0, 0]),
            high=np.array([1, 100, 1000, 1000]),
            dtype=np.float32
        )
    
    def _apply_sl_tp_refactor(self):
        """Adjusts SL and TP for failed trades based on Premium/Delta ratio."""
        self.data['premium_to_delta'] = np.where(
            self.data['delta'] != 0, self.data['premium'] / self.data['delta'], np.nan)
        
        failed_trades = self.data[self.data['status'] == 'failed']
        failed_median = failed_trades['premium_to_delta'].median()
        
        def adjust_sl_tp(row):
            if row['status'] == 'successful':
                return row['sl'], row['tp']
            
            sl, tp = row['sl'], row['tp']
            ratio = row['premium_to_delta']
            
            if ratio > failed_median:
                sl *= 1.1  # Increase SL margin
                tp *= 0.9  # Bring TP closer
            else:
                sl *= 0.9  # Tighten SL
                tp *= 1.1  # Expand TP
            
            return sl, tp
        
        self.data[['new_sl', 'new_tp']] = self.data.apply(adjust_sl_tp, axis=1, result_type='expand')
    
    def reset(self):
        self.current_step = 0
        return self._get_observation()
    
    def _get_observation(self):
        current_data = self.data.iloc[self.current_step]
        return np.array([
            float(current_data['delta']),
            float(current_data['premium']),
            float(current_data['strike_price']),
            float(current_data['bought_price'])
        ], dtype=np.float32)
    
    def step(self, action):
        action = np.array(action).flatten()
        
        if len(action) < 4:
            print("Warning: Action length incorrect, defaulting to safe values.")
            action = np.tile(action, 4)[:4]  # Repeat values to match expected length
        
        trade_data = self.data.iloc[self.current_step]
        reward = self._calculate_reward(action, trade_data)
        
        self.current_step += 1
        done = self.current_step >= self.total_steps
        obs = self._get_observation() if not done else None
        
        return obs, reward, done, {}
    
    def _calculate_reward(self, action, trade_data):
        try:
            sl_mult, tp_mult = action[:2] if trade_data['option_type'].lower() == 'call' else action[2:]
        except ValueError:
            print("Error: Invalid action length, using default multipliers.")
            sl_mult, tp_mult = 1.0, 1.0
        
        adjusted_sl = trade_data['new_sl'] * sl_mult
        adjusted_tp = trade_data['new_tp'] * tp_mult
        
        try:
            mark_prices = eval(trade_data['mark_prices'].replace(' ', ''))
            if not isinstance(mark_prices, list):
                mark_prices = [mark_prices]
        except:
            return -0.1
        
        for price in mark_prices:
            if price >= adjusted_tp:
                return (adjusted_tp - trade_data['bought_price']) / trade_data['bought_price']
            if price <= adjusted_sl:
                return (adjusted_sl - trade_data['bought_price']) / trade_data['bought_price']
        
        final_price = mark_prices[-1]
        return (final_price - trade_data['bought_price']) / trade_data['bought_price']