import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Full dataset path
file_path = "/Users/admin/Documents/Disease-Recurrence-Prediction/data/raw/data.csv"

# Load dataset
df = pd.read_csv(file_path)

print("\n===== Basic Dataset Info =====")
print(df.shape)
print(df.columns)

# Drop unnecessary column if exists
if "Unnamed: 32" in df.columns:
    df = df.drop(columns=["Unnamed: 32"])

# Check missing values
print("\n===== Missing Values =====")
print(df.isnull().sum())

# Target variable distribution
print("\n===== Diagnosis Distribution =====")
print(df["diagnosis"].value_counts())

# Convert diagnosis to numeric (M=1, B=0)
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

# Summary statistics
print("\n===== Summary Statistics =====")
print(df.describe())

# Plot distribution of diagnosis
plt.figure()
sns.countplot(x="diagnosis", data=df)
plt.title("Disease Recurrence Risk Distribution (0=Low, 1=High)")
plt.savefig("/Users/admin/Documents/Disease-Recurrence-Prediction/dashboard/risk_distribution.png")
plt.show()

# Correlation heatmap (top 10 features only for clarity)
plt.figure(figsize=(18, 14))  

corr = df.corr()

sns.heatmap(
    corr,
    annot=False,
    cmap="coolwarm",
    xticklabels=True,
    yticklabels=True
)

plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title("Feature Correlation Heatmap")

plt.tight_layout()  
plt.savefig("/Users/admin/Documents/Disease-Recurrence-Prediction/dashboard/correlation_heatmap.png", dpi=300)
plt.show()

print("\nEDA completed. Charts saved in dashboard folder.")
