from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sleep_quality_prepared.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "sleep_quality_model.pkl"
)


# Features used by the final model
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

TARGET = "sleep_quality_category"

CATEGORICAL_FEATURES = [
    "gender",
    "occupation",
    "country",
    "chronotype",
]


def main():
    # Check dataset
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {DATA_PATH}"
        )

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Validate required columns
    required_columns = FEATURES + [TARGET]
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    X = df[FEATURES]
    y = df[TARGET]

    # Preprocessing
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

    # Final Random Forest
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    # Complete pipeline
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    # Train on all available data
    pipeline.fit(X, y)

    # Save everything needed by the application
    model_data = {
        "model": pipeline,
        "features": FEATURES,
        "target": TARGET,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model_data, MODEL_PATH)

    print("Final model trained successfully.")
    print("Training samples:", len(X))
    print("Features:", len(FEATURES))
    print("Model: Balanced Random Forest")
    print("Trees:", 200)
    print(f"Saved as: {MODEL_PATH}")


if __name__ == "__main__":
    main()