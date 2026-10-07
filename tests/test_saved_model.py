import joblib
import pandas as pd


MODEL_PATH = "models/sleep_quality_model.pkl"

EXPECTED_FEATURES = [
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


def load_model():
    model_data = joblib.load(MODEL_PATH)

    assert "model" in model_data
    assert "features" in model_data

    return model_data["model"], model_data["features"]


def create_sample():
    return {
        "age": 25,
        "gender": "Female",
        "occupation": "Student",
        "bmi": 22.0,
        "country": "India",
        "sleep_duration_hrs": 7.0,
        "rem_percentage": 20.0,
        "deep_sleep_percentage": 20.0,
        "sleep_latency_mins": 20.0,
        "wake_episodes_per_night": 2,
        "caffeine_mg_before_bed": 100.0,
        "alcohol_units_before_bed": 0.0,
        "screen_time_before_bed_mins": 60.0,
        "exercise_day": 1,
        "steps_that_day": 7000,
        "nap_duration_mins": 20.0,
        "stress_score": 5.5,
        "work_hours_that_day": 8.0,
        "chronotype": "Morning",
        "heart_rate_resting_bpm": 70.0,
    }


def test_saved_model_structure():
    pipeline, features = load_model()

    assert pipeline is not None
    assert features == EXPECTED_FEATURES


def test_saved_model_prediction():
    pipeline, features = load_model()

    sample = create_sample()
    input_data = pd.DataFrame([sample])[features]

    prediction = pipeline.predict(input_data)

    assert len(prediction) == 1
    assert prediction[0] in {"Poor", "Average", "Good"}