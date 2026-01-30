import os
import pandas as pd
import numpy as np
import joblib

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


# Paths
DATA_PATH = "data/train_data.csv"
MODEL_DIR = "models"

MODEL_PATH = os.path.join(MODEL_DIR, "isolation_forest_sensor_optimized.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
SELECTOR_PATH = os.path.join(MODEL_DIR, "feature_selector.pkl")


def load_data(path: str) -> pd.DataFrame:
    """Load raw dataset."""
    return pd.read_csv(path)


def select_sensor_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select only sensor measurement columns."""
    sensor_cols = [col for col in df.columns if "SensorMeasure" in col]
    return df[sensor_cols]


def feature_optimization(sensor_df: pd.DataFrame):
    """Remove low-variance sensor features."""
    selector = VarianceThreshold(threshold=0.01)
    reduced_features = selector.fit_transform(sensor_df)
    return reduced_features, selector


def scale_features(features: np.ndarray):
    """Standardize features."""
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, scaler


def train_model(features: np.ndarray) -> IsolationForest:
    """Train optimized Isolation Forest model."""
    model = IsolationForest(
        n_estimators=150,
        max_samples=0.8,
        contamination=0.01,
        max_features=0.7,
        random_state=42
    )
    model.fit(features)
    return model


def save_artifacts(model, scaler, selector):
    """Save trained artifacts locally."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(selector, SELECTOR_PATH)


def main():
    print("Loading dataset...")
    data = load_data(DATA_PATH)

    print("Selecting sensor features...")
    sensor_data = select_sensor_features(data)

    print("Optimizing features...")
    reduced_features, selector = feature_optimization(sensor_data)

    print("Scaling features...")
    scaled_features, scaler = scale_features(reduced_features)

    print("Training Isolation Forest...")
    model = train_model(scaled_features)

    print("Saving model artifacts...")
    save_artifacts(model, scaler, selector)

    print("Training completed successfully.")


if __name__ == "__main__":
    main()
