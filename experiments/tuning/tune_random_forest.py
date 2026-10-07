import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score


# Load dataset
df = pd.read_csv("data/processed/sleep_quality_prepared.csv")


# Features
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


# Numeric and categorical columns
numeric_features = [
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

categorical_features = [
    "gender",
    "occupation",
    "country",
    "chronotype"
]


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical",
         OneHotEncoder(handle_unknown="ignore"),
         categorical_features)
    ],
    remainder="passthrough"
)


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Different Random Forest settings
models = [
    {
        "name": "Baseline",
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_leaf": 1
    },
    {
        "name": "Tuned 1",
        "n_estimators": 300,
        "max_depth": 20,
        "min_samples_leaf": 1
    },
    {
        "name": "Tuned 2",
        "n_estimators": 300,
        "max_depth": 25,
        "min_samples_leaf": 2
    },
    {
        "name": "Tuned 3",
        "n_estimators": 400,
        "max_depth": 25,
        "min_samples_leaf": 2
    }
]


print("Random Forest Tuning")
print("====================")


for settings in models:

    model = RandomForestClassifier(
        n_estimators=settings["n_estimators"],
        max_depth=settings["max_depth"],
        min_samples_leaf=settings["min_samples_leaf"],
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
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print()
    print(settings["name"])
    print("Parameters:", settings)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")