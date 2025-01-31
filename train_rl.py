# Train PPO Model
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from trading_env import TradingEnvironment
import matplotlib.pyplot as plt
import torch
env = DummyVecEnv([lambda: TradingEnvironment()])

model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=0.00005,
    n_steps=8192,
    batch_size=256,
    n_epochs=20,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.005,
    policy_kwargs=dict(
        net_arch=[512, 256, 128],
        activation_fn=torch.nn.ReLU
    )
)

model.learn(total_timesteps=500000)
model.save("trading_model")

# Generate Trading Report
from trading_report import TradingReport
reporter = TradingReport()
report = reporter.generate_report(model)
reporter.print_report(report)
reporter.save_report_to_csv(report, 'trading_report.csv')

# Generate and Save Graphs
plt.figure(figsize=(10, 5))
plt.hist(reporter.data['premium_to_delta'], bins=30, alpha=0.7, label='Premium/Delta Ratio')
plt.axvline(reporter.data['premium_to_delta'].median(), color='red', linestyle='dashed', label='Median')
plt.xlabel("Premium/Delta Ratio")
plt.ylabel("Count")
plt.legend()
plt.title("Distribution of Premium/Delta Ratio")
plt.savefig("premium_delta_distribution.png")
print("\n✅ Report and graphs saved successfully.")
