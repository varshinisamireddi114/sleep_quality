from pathlib import Path

import pandas as pd


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "sleep_health_dataset.csv"
PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "sleep_quality_prepared.csv"
)


def categorize_sleep_quality(score):
    """Convert sleep quality score into three quality categories."""
    if score <= 4:
        return "Poor"
    elif score <= 7:
        return "Average"
    return "Good"


def main():
    # Load original dataset
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_DATA_PATH}"
        )

    df = pd.read_csv(RAW_DATA_PATH)

    # Validate required column
    if "sleep_quality_score" not in df.columns:
        raise ValueError(
            "Required column 'sleep_quality_score' is missing."
        )

    # Create target column
    df["sleep_quality_category"] = df["sleep_quality_score"].apply(
        categorize_sleep_quality
    )

    # Display category distribution
    print("Sleep Quality Category Distribution:")
    print(df["sleep_quality_category"].value_counts())

    # Display first 10 rows
    print("\nFirst 10 rows:")
    print(
        df[
            ["sleep_quality_score", "sleep_quality_category"]
        ].head(10)
    )

    # Ensure processed-data directory exists
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save prepared dataset
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print("\nPrepared dataset saved successfully!")
    print(f"Saved to: {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    main()