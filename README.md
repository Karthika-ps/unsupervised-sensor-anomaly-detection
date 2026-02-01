# Unsupervised Sensor Anomaly Detection

This project implements an unsupervised anomaly detection pipeline on
industrial IoT sensor data using Isolation Forest. The system learns
normal operating behavior from multivariate sensor measurements and
flags abnormal patterns without relying on labeled failure data.

The project is designed with a production mindset, including feature
optimization, stable model tuning, and reusable inference artifacts.

---

## Problem Statement

In many real-world industrial systems, labeled anomaly or failure data
is rare or unavailable. Traditional supervised approaches are often
not feasible.

This project addresses the problem by using an unsupervised learning
approach that:
- Learns normal sensor behavior
- Detects deviations as potential anomalies
- Avoids label leakage entirely

---

## Dataset

- Source: NASA CMAPSS Turbofan Engine Sensor Dataset
- Domain: Industrial / IoT sensor data
- Samples: ~20,000
- Sensors: 21 (reduced to 11 after feature optimization)

**Note:** Remaining Useful Life (RUL) labels are available in the dataset
but are intentionally **not used** to preserve the unsupervised nature
of the problem.

---

## Dataset Access

The dataset used in this project is publicly available on Kaggle:

**NASA CMAPSS Turbofan Engine Sensor Dataset**  
https://www.kaggle.com/datasets/nasa/cmaps

Due to size constraints, the dataset is not included in this repository.

To run training locally:
1. Download the dataset from Kaggle
2. Extract `FD001.csv`
3. Place it in the following path:
data/train_data.csv
---

## Approach

### 1. Sensor Feature Selection
- Only raw sensor measurements are used
- Identifiers, operating settings, and labels are excluded

### 2. Feature Optimization
- Variance-based feature selection removes near-constant sensors
- Reduces noise and improves anomaly score stability

### 3. Feature Scaling
- Standardization ensures consistent feature scales
- Improves Isolation Forest performance

### 4. Unsupervised Modeling
- Isolation Forest is used for anomaly detection
- No labeled data is required

### 5. Model Optimization
- Hyperparameters are tuned using score stability
- Avoids supervised metrics such as accuracy or F1

---

## Project Structure
unsupervised-sensor-anomaly-detection/
│
├── data/
│ └── sample_input.csv
│
├── models/
│ ├── isolation_forest_sensor_optimized.pkl
│ ├── scaler.pkl
│ └── feature_selector.pkl
│
├── train.py
├── inference.py
├── requirements.txt
└── README.md

---

## Usage

This project separates training and inference logic to reflect
production-style workflows.

### Training

The model is trained using unsupervised learning on sensor data.
During training:
- Low-variance sensors are removed
- Features are standardized
- Isolation Forest is optimized using score stability

---

## Model Artifacts

Trained model artifacts are not committed to the repository due to
file size limitations.

To generate the model locally, run:

```bash
python train.py
```
---
## Key Takeaways

- Demonstrates unsupervised anomaly detection without label leakage
- Applies feature optimization to improve model robustness
- Uses stability-based tuning instead of supervised metrics
- Produces reusable artifacts for deployment and inference
- Applicable to industrial, IoT, and system monitoring use cases

---

## Future Improvements

- Time-aware anomaly detection
- Streaming or batch inference pipelines
- Monitoring and alerting integration
- Containerized deployment using Docker

---

## Docker (WSL)

This project was containerized and tested using Docker Engine on Linux (WSL).
Docker Desktop is not required.

Inference can be run via:
docker build -t sensor-anomaly-inference .
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models sensor-anomaly-inference
