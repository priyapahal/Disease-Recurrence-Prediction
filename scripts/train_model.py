import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load processed data
base_path = "/Users/admin/Documents/Disease-Recurrence-Prediction/data/processed/"

X_train = pd.read_csv(base_path + "X_train.csv")
X_test = pd.read_csv(base_path + "X_test.csv")
y_train = pd.read_csv(base_path + "y_train.csv")
y_test = pd.read_csv(base_path + "y_test.csv")

# Convert target to 1D array
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Create and train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("\nModel training completed.")

# Make predictions
y_pred = model.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
