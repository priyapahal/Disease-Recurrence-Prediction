import pandas as pd


file_path = "/Users/admin/Documents/Disease-Recurrence-Prediction/data/raw/data.csv"

df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nData info:")
print(df.info())
