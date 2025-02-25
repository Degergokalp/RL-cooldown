RL-cooldown: Reinforcement Learning for Market Analysis

This project applies Reinforcement Learning to TradingView backtesting exports, enabling dynamic “cooldown” strategies to filter trades and improve profitability.

Overview

Data Input: You provide a CSV of your TradingView backtest results (e.g., trades) in the data/list-of-trades/ directory, named trades.csv.
Preprocessing: The script automatically formats this CSV to remove unwanted rows (like “Entry” trades) and saves the result as filtered_trades.csv in data/formatted-result/.
RL Training: The script then runs a reinforcement learning agent multiple times (e.g., 20 times), each time training on your historical trades. It selects the best run based on average profit per episode.
Reports: Finally, it prints a report to the console comparing a baseline (no cooldown) to the RL-based cooldown strategy.
Directory Structure

RL-MAT/
├── data/
│   ├── list-of-trades/
│   │   └── trades.csv          # User's TradingView backtest export
│   ├── formatted-result/
│   │   └── filtered_trades.csv # Auto-generated filtered data
├── helpers/
│   ├── __init__.py
│   └── csv_formatter.py        # (Optional) CSV formatting logic
├── rl_agent/
│   └── train_rl.py             # Main RL training script
├── venv/                       # (Optional) Python virtual environment
├── .gitignore
├── requirements.txt
└── README.md                   # This README file
Getting Started

Install Dependencies
Make sure you have Python 3.8+ installed. Create and activate a virtual environment (optional but recommended), then install the required packages:
pip install -r requirements.txt
Place Your Trades CSV
Export your backtest results from TradingView (or similar) as a CSV.
Rename the file to trades.csv.
Put it in data/list-of-trades/.
Run the Training Script
From the project root (i.e., RL-MAT/ directory), run:
python -m rl_agent.train_rl
or, if you prefer, navigate to rl_agent/ and run:
python train_rl.py
This will:
Format the CSV (remove “Entry” trades, etc.).
Train the RL agent 20 times.
Evaluate each run and pick the best-performing policy.
Print a final report to the console.
Check Results
The final output in your console will show baseline metrics (no cooldown) and the RL agent’s performance, along with which cooldown strategies were triggered and how often.
A typical snippet might look like:
Final Report (Best of 20 Runs):
...
RL Dynamic Cooldown Strategy:
- Average Total Profit USDT per episode: 31020.16
- Average Win Rate: 56.76%
- Average Profit Factor: 4.57
...
How It Works

format_csv(): Removes rows with "Entry" from the original trades.csv and saves the cleaned data to filtered_trades.csv.
DynamicCooldownTradingEnv: A custom Gymnasium environment that processes trades sequentially and applies cooldowns based on the RL agent’s actions.
DQN Training: Uses Stable-Baselines3’s DQN to learn a policy that decides whether to execute or skip trades and how to trigger cooldown periods.
Multiple Runs: The script runs the training loop multiple times (default is 20) to handle RL randomness. It picks the “best” run based on the highest average profit.
Customization

Tweak Timesteps: Adjust model.learn(total_timesteps=10000) in train_rl.py to train longer or shorter.
Change the Number of Runs: Modify num_runs = 20 if you want more or fewer runs.
Cooldown Mapping: In train_rl.py, you can change COOLDOWN_MAPPING (e.g., 4, 6, 8 trades) to your preferred cooldown lengths.
CSV Format: If your CSV has different columns, you might need to edit the environment code to read the correct profit column or to skip different row types.
Troubleshooting

ModuleNotFoundError: Make sure you’re running the script from the project root with python -m rl_agent.train_rl so Python recognizes the local helpers package.
Different Results on Each Run: RL training is stochastic. You can set random seeds or run multiple times to get a more stable best result.
CSV Issues: Check that your trades.csv has the expected columns (e.g., “Type”, “Profit USDT”, etc.). The script might need adjustments if your backtest CSV uses different headings.
License

This project is provided “as is” without warranty under the MIT License. Feel free to adapt and modify for your own use.