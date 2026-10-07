import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Load prepared dataset
df = pd.read_csv("data/processed/sleep_quality_prepared.csv")


# Target
y = df["sleep_quality_category"]


# Train-test split
y_train, y_test = train_test_split(
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Find majority class using training data only
majority_class = y_train.value_counts().idxmax()


# Predict the majority class for every test sample
y_pred_baseline = [majority_class] * len(y_test)


# Calculate baseline accuracy on the test set
accuracy = accuracy_score(y_test, y_pred_baseline)


print("Majority Class Baseline")
print("-----------------------")
print("Majority Class:", majority_class)
print("Baseline Accuracy:", accuracy)