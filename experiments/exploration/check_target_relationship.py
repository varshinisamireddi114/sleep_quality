import pandas as pd


# Load dataset
df = pd.read_csv("data/raw/sleep_health_dataset.csv")


# Numeric features used by the model
features = [
    "age",
    "bmi",
    "sleep_duration_hrs",
    "rem_percentage",
    "deep_sleep_percentage",
    "sleep_latency_mins",
    "wake_episodes_per_night",
    "caffeine_mg_before_bed",
    "alcohol_units_before_bed",
    "screen_time_before_bed_mins",
    "exercise_day",
    "steps_that_day",
    "nap_duration_mins",
    "stress_score",
    "work_hours_that_day",
    "heart_rate_resting_bpm"
]


target = "sleep_quality_score"


# Calculate Pearson correlations
correlations = df[features + [target]].corr(
    numeric_only=True
)[target]


# Remove target itself and sort by absolute correlation
correlations = correlations.drop(target).sort_values(
    key=abs,
    ascending=False
)


print("Correlation with Sleep Quality Score")
print("-------------------------------------")
print(correlations)