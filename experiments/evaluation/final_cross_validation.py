from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sleep_quality_prepared.csv"
)

FEATURES = [
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
    "heart_rate_resting_bpm",
]

CATEGORICAL_FEATURES = [
    "gender",
    "occupation",
    "country",
    "chronotype",
]

TARGET = "sleep_quality_category"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = FEATURES + [TARGET]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    X = df[FEATURES]
    y = df[TARGET]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            )
        ],
        remainder="passthrough",
    )

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "macro_f1": "f1_macro",
    }

    results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
    )

    accuracy_scores = results["test_accuracy"]
    balanced_accuracy_scores = results["test_balanced_accuracy"]
    macro_f1_scores = results["test_macro_f1"]

    print("\nFINAL 5-FOLD CROSS-VALIDATION")
    print("=============================")

    print(f"\nTotal samples: {len(df)}")
    print("Number of folds: 5")

    print("\nAccuracy:")
    print(accuracy_scores)
    print(f"Mean: {accuracy_scores.mean():.4f}")
    print(f"Std:  {accuracy_scores.std():.4f}")

    print("\nBalanced Accuracy:")
    print(balanced_accuracy_scores)
    print(f"Mean: {balanced_accuracy_scores.mean():.4f}")
    print(f"Std:  {balanced_accuracy_scores.std():.4f}")

    print("\nMacro F1:")
    print(macro_f1_scores)
    print(f"Mean: {macro_f1_scores.mean():.4f}")
    print(f"Std:  {macro_f1_scores.std():.4f}")


if __name__ == "__main__":
    main()