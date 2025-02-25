import pandas as pd

# Load the dataset from an Excel file
file_path = "datasets/gait-in-parkinsons-disease/demographics.xls"  # Update with your actual file path
df = pd.read_excel(file_path)

# Separate the dataset by the "Study" column values
df_ga = df[df['Study'] == 'Ga']
df_ju = df[df['Study'] == 'Ju']
df_si = df[df['Study'] == 'Si']

# Save each group as separate Excel files (xlsx format)
df_ga.to_excel("gait_parkinsons_Ga.xlsx", index=False)
df_ju.to_excel("gait_parkinsons_Ju.xlsx", index=False)
df_si.to_excel("gait_parkinsons_Si.xlsx", index=False)

print("Dataset separated by Ga, Ju, and Si successfully!")
