import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score


# Load data
df = pd.read_csv("data/processed/sleep_quality_prepared.csv")

target = "sleep_quality_category"


# Current 20 features
all_features = [
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


# Remove the weakest features based on
# the feature importance experiment
reduced_features = [
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
    "screen_time_before_bed_mins",
    "steps_that_day",
    "nap_duration_mins",
    "stress_score",
    "work_hours_that_day",
    "chronotype",
    "heart_rate_resting_bpm"
]


# Even smaller set containing the strongest features
strong_features = [
    "sleep_duration_hrs",
    "stress_score",
    "wake_episodes_per_night",
    "rem_percentage",
    "sleep_latency_mins",
    "work_hours_that_day",
    "steps_that_day",
    "bmi",
    "deep_sleep_percentage",
    "screen_time_before_bed_mins",
    "age",
    "heart_rate_resting_bpm"
]


categorical_features = [
    "gender",
    "occupation",
    "country",
    "chronotype"
]


X_train_full, X_test_full, y_train, y_test = train_test_split(
    df[all_features],
    df[target],
    test_size=0.20,
    random_state=42,
    stratify=df[target]
)


feature_sets = [
    ("All 20 features", all_features),
    ("Reduced 17 features", reduced_features),
    ("Strong 12 features", strong_features)
]


print("Feature Selection Experiment")
print("============================")


for name, features in feature_sets:

    categorical = [
        feature
        for feature in categorical_features
        if feature in features
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical
            )
        ],
        remainder="passthrough"
    )

    X_train = X_train_full[features]
    X_test = X_test_full[features]

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
    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro"
    )

    print()
    print(name)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")