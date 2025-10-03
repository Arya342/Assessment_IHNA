import pandas as pd

# Replace with your actual file path
file_path = "c:/Users/MT078/Documents/Assessment_IHNA/assessment_ ihna.xlsx"

# Read the Excel file
df = pd.read_excel(file_path)

# Display the first few rows
print(df.head())