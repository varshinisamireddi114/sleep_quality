import pandas as pd


# Load raw dataset
df = pd.read_csv("data/raw/sleep_health_dataset.csv")


# Dataset shape
print("Dataset shape:")
print(df.shape)


# Column names
print("\nColumns:")

for i, column in enumerate(df.columns, start=1):
    print(i, "->", column)


# First 5 rows
print("\nFirst 5 rows:")
print(df.head())


# Missing values
print("\nMissing values:")
print(df.isnull().sum())