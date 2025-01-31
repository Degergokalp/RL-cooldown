import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from stable_baselines3.common.vec_env import DummyVecEnv
from trading_env import TradingEnvironment

class TradingReport:
    def __init__(self, data_path='data/formatted-result/dataset.csv'):
        self.data = pd.read_csv(data_path)
        
    def calculate_original_rr(self, row) -> Tuple[float, bool]:
        """Calculate R:R ratio and success for original trade"""
        bought_price = row['bought_price']
        sl = row['sl']
        tp = row['tp']
        mark_prices = eval(row['mark_prices'])
        is_call = row['option_type'].lower() == 'call'
        
        # Calculate potential R:R
        if is_call:
            risk = bought_price - sl
            reward = tp - bought_price
        else:
            risk = sl - bought_price
            reward = bought_price - tp
            
        rr_ratio = abs(reward / risk) if risk != 0 else 0
        
        # Determine if trade was successful
        for price in mark_prices:
            if is_call:
                if price <= sl:  # Stop loss hit
                    return -rr_ratio, False
                if price >= tp:  # Take profit hit
                    return rr_ratio, True
            else:
                if price >= sl:  # Stop loss hit
                    return -rr_ratio, False
                if price <= tp:  # Take profit hit
                    return rr_ratio, True
        
        # If no SL/TP hit, calculate final R:R based on last price
        final_price = mark_prices[-1]
        if is_call:
            actual_return = (final_price - bought_price) / (bought_price - sl)
        else:
            actual_return = (bought_price - final_price) / (sl - bought_price)
        
        return actual_return, actual_return > 0

    def calculate_optimized_rr(self, row, sl_mult, tp_mult) -> Tuple[float, bool]:
        """Calculate R:R ratio and success for optimized trade"""
        bought_price = float(row['bought_price'])
        original_sl = float(row['sl'])
        original_tp = float(row['tp'])
        mark_prices = eval(row['mark_prices'])
        is_call = row['option_type'].lower() == 'call'
        
        # Convert multipliers to float if they're numpy arrays
        sl_mult = float(sl_mult)
        tp_mult = float(tp_mult)
        
        # Calculate adjusted SL and TP
        if is_call:
            adjusted_sl = bought_price - (bought_price - original_sl) * sl_mult
            adjusted_tp = bought_price + (original_tp - bought_price) * tp_mult
        else:
            adjusted_sl = bought_price + (original_sl - bought_price) * sl_mult
            adjusted_tp = bought_price - (bought_price - original_tp) * tp_mult
        
        # Calculate potential R:R
        if is_call:
            risk = bought_price - adjusted_sl
            reward = adjusted_tp - bought_price
        else:
            risk = adjusted_sl - bought_price
            reward = bought_price - adjusted_tp
            
        rr_ratio = abs(float(reward) / float(risk)) if abs(risk) > 1e-10 else 0
        
        # Determine if trade was successful
        for price in mark_prices:
            if is_call:
                if price <= adjusted_sl:
                    return -rr_ratio, False
                if price >= adjusted_tp:
                    return rr_ratio, True
            else:
                if price >= adjusted_sl:
                    return -rr_ratio, False
                if price <= adjusted_tp:
                    return rr_ratio, True
        
        # If no SL/TP hit, calculate final R:R based on last price
        final_price = mark_prices[-1]
        if is_call:
            actual_return = (final_price - bought_price) / (bought_price - adjusted_sl)
        else:
            actual_return = (bought_price - final_price) / (adjusted_sl - bought_price)
        
        return actual_return, actual_return > 0

    def generate_report(self, model=None):
        """Generate comprehensive trading report"""
        total_trades = len(self.data)
        
        # Original performance metrics
        original_results = [self.calculate_original_rr(row) for _, row in self.data.iterrows()]
        original_rr = [r[0] for r in original_results]
        original_success = [r[1] for r in original_results]
        
        # Calculate optimized performance if model provided
        if model:
            env = DummyVecEnv([lambda: TradingEnvironment()])
            obs = env.reset()
            optimized_results = []
            
            for i in range(total_trades):
                action, _ = model.predict(obs)
                action = action.flatten()  # Flatten the action array
                row = self.data.iloc[i]
                
                # Get appropriate multipliers based on option type
                if row['option_type'].lower() == 'call':
                    sl_mult = action[0]
                    tp_mult = action[1]
                else:
                    sl_mult = action[2]
                    tp_mult = action[3]
                
                optimized_rr = self.calculate_optimized_rr(row, sl_mult, tp_mult)
                optimized_results.append(optimized_rr)
                if i < total_trades - 1:
                    obs, _, _, _ = env.step(action)
        
        # Generate report
        report = {
            "Original Strategy": {
                "Total Trades": total_trades,
                "Successful Trades": sum(original_success),
                "Failed Trades": total_trades - sum(original_success),
                "Win Rate": f"{(sum(original_success)/total_trades)*100:.2f}%",
                "Total RR Gained": sum(rr for rr in original_rr if rr > 0),
                "Total RR Lost": abs(sum(rr for rr in original_rr if rr < 0)),
                "Net RR": sum(original_rr),
                "Average RR per Trade": sum(original_rr)/total_trades
            }
        }
        
        if model:
            optimized_rr = [r[0] for r in optimized_results]
            optimized_success = [r[1] for r in optimized_results]
            
            report["Optimized Strategy"] = {
                "Total Trades": total_trades,
                "Successful Trades": sum(optimized_success),
                "Failed Trades": total_trades - sum(optimized_success),
                "Win Rate": f"{(sum(optimized_success)/total_trades)*100:.2f}%",
                "Total RR Gained": sum(rr for rr in optimized_rr if rr > 0),
                "Total RR Lost": abs(sum(rr for rr in optimized_rr if rr < 0)),
                "Net RR": sum(optimized_rr),
                "Average RR per Trade": sum(optimized_rr)/total_trades
            }
        
        return report

    def print_report(self, report):
        """Print formatted report"""
        for strategy_name, metrics in report.items():
            print(f"\n{'='*20} {strategy_name} {'='*20}")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"{metric}: {value:.2f}")
                else:
                    print(f"{metric}: {value}")

    def save_report_to_csv(self, report, filename='trading_report.csv'):
        """Save the report to a CSV file"""
        # Convert report to DataFrame
        report_data = []
        for strategy, metrics in report.items():
            for metric, value in metrics.items():
                report_data.append({
                    'Strategy': strategy,
                    'Metric': metric,
                    'Value': value
                })
        
        df = pd.DataFrame(report_data)
        
        # Save to CSV
        df.to_csv(filename, index=False)
        print(f"\nReport saved to {filename}") 