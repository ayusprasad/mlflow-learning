# 📊 MLflow Experiment Tracking & Hyperparameter Tuning

A hands-on laboratory demonstrating machine learning experiment management, metric logging, model artifact preservation, and hyperparameter tuning sweeps using **MLflow** and **scikit-learn**.

---

## 🌟 Key Features
- **Metric & Artifact Logging:** Tracks training accuracy, confusion matrices, and parameter logs across runs.
- **Hyperparameter Sweeps (`src/hypertune1.py`):** Grid and random search logging with real-time MLflow comparison.
- **Interactive Web UI:** Browse and compare runs on the built-in MLflow UI dashboard.
- **Dockerized:** Instant container serving of the MLflow Tracking Server.

---

## 🚀 Quickstart

### 1. View MLflow UI with Docker
```bash
docker build -t mlflow-server .
docker run -p 5000:5000 mlflow-server
```
Open [http://localhost:5000](http://localhost:5000) to view runs and artifacts.

---

### 2. Run Experiments Locally
```bash
pip install -r requirements.txt

# Run sample training experiment
python src/file1.py

# Run hyperparameter tuning sweep
python src/hypertune1.py

# Launch MLflow UI
mlflow ui
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000).