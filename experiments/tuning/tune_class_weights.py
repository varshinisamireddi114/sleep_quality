import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report


df = pd.read_csv("data/processed/sleep_quality_prepared.csv")

features = [
    "age", "gender", "occupation", "bmi", "country",
    "sleep_duration_hrs", "rem_percentage", "deep_sleep_percentage",
    "sleep_latency_mins", "wake_episodes_per_night",
    "caffeine_mg_before_bed", "alcohol_units_before_bed",
    "screen_time_before_bed_mins", "exercise_day", "steps_that_day",
    "nap_duration_mins", "stress_score", "work_hours_that_day",
    "chronotype", "heart_rate_resting_bpm"
]

target = "sleep_quality_category"

X = df[features]
y = df[target]

categorical_features = [
    "gender",
    "occupation",
    "country",
    "chronotype"
]

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

weights = [
    {
        "name": "Balanced",
        "class_weight": "balanced"
    },
    {
        "name": "Good x2",
        "class_weight": {
            "Poor": 1,
            "Average": 1,
            "Good": 2
        }
    },
    {
        "name": "Good x3",
        "class_weight": {
            "Poor": 1,
            "Average": 1,
            "Good": 3
        }
    },
    {
        "name": "Good x4",
        "class_weight": {
            "Poor": 1,
            "Average": 1,
            "Good": 4
        }
    }
]

print("Class Weight Experiment")
print("=======================")

for setting in weights:

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight=setting["class_weight"],
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
    print(setting["name"])
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print(
        classification_report(
            y_test,
            y_pred,
            labels=["Poor", "Average", "Good"],
            digits=4
        )
    )