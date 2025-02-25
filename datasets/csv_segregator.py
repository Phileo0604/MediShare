import pandas as pd

# Load the dataset
file_path = "datasets/gait-in-parkinsons-disease/demographics.xls"  # Change this to your actual file path
df = pd.read_csv(file_path)

# Calculate split sizes
split1 = len(df) // 3
split2 = 2 * (len(df) // 3)

# Split dataset into three parts
df_part1 = df.iloc[:split1]   # First 1/3
df_part2 = df.iloc[split1:split2]  # Second 1/3
df_part3 = df.iloc[split2:]   # Last 1/3

# Save each part as separate CSV files
df_part1.to_csv("gait_parkinsons1.csv", index=False)
df_part2.to_csv("gait_parkinsons2.csv", index=False)
df_part3.to_csv("gait_parkinsons3.csv", index=False)

print("Dataset split into three parts successfully!")
