import pandas as pd

def format_csv(file_path='data/list-of-trades/trades.csv'):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Display the original DataFrame shape
    print(f"Original DataFrame shape: {df.shape}")

    # Remove rows where Type is 'Entry'
    df_filtered = df[~df['Type'].str.contains('Entry', na=False)]

    # Display the new DataFrame shape
    print(f"Filtered DataFrame shape: {df_filtered.shape}")

    # Save the filtered DataFrame to a new CSV file
    df_filtered.to_csv('data/formatted-result/filtered_trades.csv', index=False)

    print("Filtered data saved to 'filtered_trades.csv'.")