import os
import pandas as pd

csv_folder = "data/sample-backtest-results"  # Update this path

csv_files = [f for f in os.listdir(csv_folder) if f.endswith(".csv")]
dataframes = []

for file in csv_files:
    file_path = os.path.join(csv_folder, file)
    try:
        # First, let's check the file structure
        with open(file_path, 'r') as f:
            first_line = f.readline()
            num_columns_header = len(first_line.split(','))
            print(f"\nAnalyzing {file}:")
            print(f"Number of columns in header: {num_columns_header}")
        
        # Try reading with error handling
        df = pd.read_csv(file_path, on_bad_lines='warn')
        print(f"Successfully read {file} with {df.shape[1]} columns")
        dataframes.append(df)
    except Exception as e:
        print(f"Error reading {file}: {str(e)}")
        continue

if dataframes:
    merged_df = pd.concat(dataframes, ignore_index=True)
    output_file = "dataset.csv"
    merged_df.to_csv(output_file, index=False)
    print(f"\n✅ Merged {len(dataframes)} CSV files into {output_file}")
else:
    print("\n❌ No files were successfully processed")
