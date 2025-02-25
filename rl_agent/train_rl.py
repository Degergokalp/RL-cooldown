import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from collections import Counter
from helpers.csv_formatter import format_csv

# Mapping for cooldown durations (in trades)
COOLDOWN_MAPPING = {0: 4, 1: 6, 2: 8}

# -------------------------
# Custom Gymnasium Environment: Dynamic Cooldown Trading (Discrete Action Space)
# -------------------------
class DynamicCooldownTradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    
    def __init__(self, csv_file, window=10):
        super(DynamicCooldownTradingEnv, self).__init__()
        self.df = pd.read_csv(csv_file)
        self.n = len(self.df)
        self.current_index = 0
        self.cooldown_counter = 0  # trades to skip
        self.executed_trades = []  # list of profits from executed trades
        self.window = window  # look-back window for distribution features
        # This list will record triggered cooldown strategies as tuples (threshold, cooldown_duration)
        self.triggered_strategies = []
        
        # Use Discrete(24): each action decodes into (decision, threshold, cooldown_index)
        self.action_space = spaces.Discrete(24)
        # Observation: 6 continuous features
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
    
    def reset(self, **kwargs):
        self.current_index = 0
        self.cooldown_counter = 0
        self.executed_trades = []
        self.triggered_strategies = []
        return self._get_observation(), {}
    
    def _get_observation(self):
        if self.current_index < self.n:
            current_profit = self.df.loc[self.current_index, "Profit USDT"]
            sl_value = self.df.loc[self.current_index, "SL"] if "SL" in self.df.columns else 0.0
        else:
            current_profit = 0.0
            sl_value = 0.0
        
        recent = self.executed_trades[-self.window:] if self.executed_trades else []
        neg_trades = [p for p in recent if p < 0]
        pos_trades = [p for p in recent if p > 0]
        avg_neg = np.mean(neg_trades) if neg_trades else 0.0
        avg_pos = np.mean(pos_trades) if pos_trades else 0.0
        mean_recent = np.mean(recent) if recent else 0.0
        std_recent = np.std(recent) if recent else 0.0

        obs = np.array([current_profit, sl_value, avg_neg, avg_pos, mean_recent, std_recent], dtype=np.float32)
        return obs
    
    def step(self, action):
        info = {}
        terminated = False
        truncated = False
        
        # If in cooldown period, skip trade
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.current_index += 1
            if self.current_index >= self.n:
                terminated = True
            return self._get_observation(), 0.0, terminated, truncated, info

        # Decode action: 
        # decision = action // 12  (0 = execute, 1 = skip)
        # threshold = (action % 12) // 3   (values 0 to 3)
        # cooldown_index = (action % 12) % 3 (values 0 to 2)
        decision = action // 12
        remainder = action % 12
        threshold = remainder // 3
        cooldown_index = remainder % 3
        cooldown_duration = COOLDOWN_MAPPING[cooldown_index]
        
        if decision == 0:
            profit = self.df.loc[self.current_index, "Profit USDT"]
            reward = profit
            self.executed_trades.append(profit)
            # Check if at least 3 trades executed and then apply dynamic cooldown if condition met
            if len(self.executed_trades) >= 3:
                last_three = self.executed_trades[-3:]
                count_positive = sum(1 for p in last_three if p > 0)
                if count_positive >= threshold:
                    self.cooldown_counter = cooldown_duration - 1
                    self.triggered_strategies.append((threshold, cooldown_duration))
        else:
            # Skip trade immediately, applying cooldown
            self.cooldown_counter = cooldown_duration - 1
            reward = 0.0
            self.triggered_strategies.append((threshold, cooldown_duration))
        
        self.current_index += 1
        if self.current_index >= self.n:
            terminated = True
        
        return self._get_observation(), reward, terminated, truncated, info

# -------------------------
# Baseline Metrics Calculation and Profit Distribution Analysis
# -------------------------
def compute_baseline_metrics(csv_file):
    df = pd.read_csv(csv_file)
    total_profit = df["Profit USDT"].sum()
    win_rate = (df["Profit USDT"] > 0).mean() * 100
    total_positive = df.loc[df["Profit USDT"] > 0, "Profit USDT"].sum()
    total_negative = abs(df.loc[df["Profit USDT"] < 0, "Profit USDT"].sum())
    profit_factor = total_positive / total_negative if total_negative > 0 else np.nan
    distribution = {
        "mean_profit": df["Profit USDT"].mean(),
        "std_profit": df["Profit USDT"].std(),
        "min_profit": df["Profit USDT"].min(),
        "max_profit": df["Profit USDT"].max()
    }
    return total_profit, win_rate, profit_factor, distribution

format_csv()
csv_file = "data/formatted-result/filtered_trades.csv"
baseline_profit, baseline_win_rate, baseline_profit_factor, profit_distribution = compute_baseline_metrics(csv_file)

print("Baseline Metrics (No Cooldown Strategy):")
print("Total Profit USDT:", baseline_profit)
print("Win Rate (%):", baseline_win_rate)
print("Profit Factor:", baseline_profit_factor)
print("Profit Distribution:", profit_distribution)

# -------------------------
# Evaluation Function
# -------------------------
def evaluate_trading_strategy(env, model, n_eval_episodes=10):
    episode_rewards = []
    win_rates = []
    profit_factors = []
    all_triggered_strategies = []
    
    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0.0
        executed = []
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            if reward != 0:
                executed.append(reward)
        episode_rewards.append(episode_reward)
        all_triggered_strategies.extend(env.unwrapped.triggered_strategies)
        if executed:
            ep_win_rate = (sum(1 for r in executed if r > 0) / len(executed)) * 100
            pos_sum = sum(r for r in executed if r > 0)
            neg_sum = abs(sum(r for r in executed if r < 0))
            ep_pf = pos_sum / neg_sum if neg_sum > 0 else np.nan
        else:
            ep_win_rate = 0
            ep_pf = np.nan
        win_rates.append(ep_win_rate)
        profit_factors.append(ep_pf)
    
    avg_reward = np.mean(episode_rewards)
    avg_win_rate = np.mean(win_rates)
    avg_profit_factor = np.nanmean(profit_factors)
    return avg_reward, avg_win_rate, avg_profit_factor, all_triggered_strategies

# -------------------------
# Run Multiple Trainings and Pick the Best
# -------------------------
num_runs = 20
best_run = None
best_profit = -np.inf
all_results = []

for run in range(num_runs):
    print(f"\n----- Run {run+1} of {num_runs} -----")
    env = DynamicCooldownTradingEnv(csv_file)
    env = Monitor(env)
    model = DQN("MlpPolicy", env, verbose=0)
    # Train for a fixed number of timesteps (adjust as needed)
    model.learn(total_timesteps=10000)
    
    eval_profit, eval_win_rate, eval_profit_factor, triggered_strats = evaluate_trading_strategy(env, model, n_eval_episodes=10)
    
    result = {
        "run": run+1,
        "avg_profit": eval_profit,
        "win_rate": eval_win_rate,
        "profit_factor": eval_profit_factor,
        "triggered_strats": triggered_strats
    }
    all_results.append(result)
    
    print(f"Run {run+1}: Profit: {eval_profit:.2f}, Win Rate: {eval_win_rate:.2f}%, Profit Factor: {eval_profit_factor:.2f}")
    
    if eval_profit > best_profit:
        best_profit = eval_profit
        best_run = result

# Process the triggered strategies of the best run
strategy_counter = Counter(best_run["triggered_strats"])
if strategy_counter:
    optimal_strategy, optimal_count = max(strategy_counter.items(), key=lambda x: x[1])
else:
    optimal_strategy, optimal_count = (None, 0)

# -------------------------
# Final Report: Best Run Across 20 Runs
# -------------------------
report = f"""
---------------------------
Final Report (Best of {num_runs} Runs):
---------------------------
Baseline Strategy (No Cooldown):
- Total Profit USDT: {baseline_profit}
- Win Rate: {baseline_win_rate:.2f}%
- Profit Factor: {baseline_profit_factor:.2f}
- Profit Distribution: Mean = {profit_distribution['mean_profit']:.2f}, Std = {profit_distribution['std_profit']:.2f}, Min = {profit_distribution['min_profit']:.2f}, Max = {profit_distribution['max_profit']:.2f}

Best RL Dynamic Cooldown Strategy (Run {best_run["run"]}):
- Average Total Profit USDT per episode: {best_run["avg_profit"]:.2f}
- Average Win Rate: {best_run["win_rate"]:.2f}%
- Average Profit Factor: {best_run["profit_factor"]:.2f}

Cooldown Strategy Distribution (Best Run):
"""
for (threshold, cooldown_duration), count in strategy_counter.items():
    report += f"\n  - {threshold} in 3 and {cooldown_duration} cooldown: triggered {count} times"
    
if optimal_strategy is not None:
    report += f"\n\nMost Optimal Strategy Type: {optimal_strategy[0]} in 3 and {optimal_strategy[1]} cooldown (triggered {optimal_count} times)"
else:
    report += "\n\nNo cooldown strategies were triggered."
    
report += "\n---------------------------"
print(report)
