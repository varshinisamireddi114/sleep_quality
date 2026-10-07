import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder


# Load prepared dataset
df = pd.read_csv("data/processed/sleep_quality_prepared.csv")


# Model features
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


X = df[features]
y = df[target]


# Categorical features
categorical_features = [
    "gender",
    "occupation",
    "country",
    "chronotype"
]


# Preprocessing
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


# Transform features
X_processed = preprocessor.fit_transform(X)


# Train Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_processed, y)


# Get feature names after preprocessing
feature_names = preprocessor.get_feature_names_out()


# Feature importance
importance = pd.Series(
    model.feature_importances_,
    index=feature_names
).sort_values(ascending=False)


print("Top 20 Feature Importances")
print("--------------------------")
print(importance.head(20))