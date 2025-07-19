import pandas as pd

# Read the CSV file
df = pd.read_csv('enhanced_ui_preference_dataset.csv')  # Replace 'input.csv' with your file name

# Take the first 10 lines
first_10_rows = df.head(10)

# Write them to a new CSV file
first_10_rows.to_csv('10_first_line.csv', index=False)
