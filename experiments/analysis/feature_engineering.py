import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report
)


df = pd.read_csv("data/processed/sleep_quality_prepared.csv")


features = [
    "age",
    "gender",
    "occupation",
    "bmi",
    "country",
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
    "chronotype",
    "heart_rate_resting_bpm"
]

target = "sleep_quality_category"


# Create derived features
df["sleep_efficiency"] = (
    df["sleep_duration_hrs"]
    / (df["sleep_duration_hrs"] + df["sleep_latency_mins"] / 60)
)

df["sleep_disruption"] = (
    df["wake_episodes_per_night"] * df["sleep_latency_mins"]
)

df["work_stress"] = (
    df["work_hours_that_day"] * df["stress_score"]
)

df["screen_caffeine_load"] = (
    df["screen_time_before_bed_mins"]
    * (df["caffeine_mg_before_bed"] + 1)
)

df["activity_per_hour"] = (
    df["steps_that_day"]
    / (df["work_hours_that_day"] + 1)
)


engineered_features = [
    "sleep_efficiency",
    "sleep_disruption",
    "work_stress",
    "screen_caffeine_load",
    "activity_per_hour"
]

features = features + engineered_features


categorical_features = [
    "gender",
    "occupation",
    "country",
    "chronotype"
]


X = df[features]
y = df[target]

label_map = {
    "Poor": 0,
    "Average": 1,
    "Good": 2
}

y = y.map(label_map)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")


print("\nRandom Forest + Feature Engineering")
print("-----------------------------------")
print(f"Accuracy: {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Poor", "Average", "Good"]
    )
)