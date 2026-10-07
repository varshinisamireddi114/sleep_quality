import pandas as pd


# Load original dataset
df = pd.read_csv("data/raw/sleep_health_dataset.csv")


# Scores near the Poor/Average boundary
near_lower_boundary = df[
    (df["sleep_quality_score"] >= 3.5) &
    (df["sleep_quality_score"] <= 4.5)
]


# Scores near the Average/Good boundary
near_upper_boundary = df[
    (df["sleep_quality_score"] >= 6.5) &
    (df["sleep_quality_score"] <= 7.5)
]


print("Scores near Poor/Average boundary (3.5 - 4.5):")
print("Count:", len(near_lower_boundary))


print("\nScores near Average/Good boundary (6.5 - 7.5):")
print("Count:", len(near_upper_boundary))


# Exact scores around the lower boundary
print("\nExact scores around lower boundary:")

print(
    df[
        (df["sleep_quality_score"] >= 3.8) &
        (df["sleep_quality_score"] <= 4.2)
    ]["sleep_quality_score"]
    .value_counts()
    .sort_index()
)


# Exact scores around the upper boundary
print("\nExact scores around upper boundary:")

print(
    df[
        (df["sleep_quality_score"] >= 6.8) &
        (df["sleep_quality_score"] <= 7.2)
    ]["sleep_quality_score"]
    .value_counts()
    .sort_index()
)