import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
file_path = "/Users/admin/Documents/Disease-Recurrence-Prediction/data/raw/data.csv"

df = pd.read_csv(file_path)

# Drop unnecessary column if exists
if "Unnamed: 32" in df.columns:
    df = df.drop(columns=["Unnamed: 32"])

# Convert diagnosis to numeric
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

print("\nDataset cleaned successfully.")

# Separate features and target
X = df.drop(columns=["diagnosis", "id"])
y = df["diagnosis"]

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\nTrain/Test split completed.")
print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)

# Feature scaling (important for ML)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed.")

# Save processed data (optional for later use)
processed_path = "/Users/admin/Documents/Disease-Recurrence-Prediction/data/processed/"

import os
os.makedirs(processed_path, exist_ok=True)

pd.DataFrame(X_train_scaled).to_csv(processed_path + "X_train.csv", index=False)
pd.DataFrame(X_test_scaled).to_csv(processed_path + "X_test.csv", index=False)
y_train.to_csv(processed_path + "y_train.csv", index=False)
y_test.to_csv(processed_path + "y_test.csv", index=False)

print("\nPreprocessing completed. Files saved in processed folder.")
