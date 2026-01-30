import pandas as pd
import joblib
import numpy as np

MODEL_PATH = "models/isolation_forest_sensor_optimized.pkl"
SCALER_PATH = "models/scaler.pkl"
SELECTOR_PATH = "models/feature_selector.pkl"
INPUT_PATH = "data/sample_input.csv"


def load_artifacts():
    """Load trained model and preprocessing artifacts."""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    selector = joblib.load(SELECTOR_PATH)
    return model, scaler, selector


def load_input_data(path: str) -> pd.DataFrame:
    """Load new sensor data for inference."""
    return pd.read_csv(path)


def preprocess_input(data: pd.DataFrame, scaler, selector) -> np.ndarray:
    """
    Apply the same preprocessing used during training:
    - Feature selection
    - Scaling
    """
    selected_features = selector.transform(data)
    scaled_features = scaler.transform(selected_features)
    return scaled_features


def run_inference(model, processed_data: np.ndarray) -> pd.DataFrame:
    """Run anomaly detection."""
    scores = model.decision_function(processed_data)
    predictions = model.predict(processed_data)

    return pd.DataFrame({
        "anomaly_score": scores,
        "anomaly_flag": predictions
    })


def main():
    print("Loading artifacts...")
    model, scaler, selector = load_artifacts()

    print("Loading input data...")
    input_data = load_input_data(INPUT_PATH)

    print("Preprocessing input data...")
    processed_data = preprocess_input(input_data, scaler, selector)

    print("Running inference...")
    results = run_inference(model, processed_data)

    print("Inference complete. Sample output:")
    print(results.head())


if __name__ == "__main__":
    main()
