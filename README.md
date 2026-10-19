# MLflow Experiment Tracking Lab

[![CI](https://github.com/ayusprasad/mlflow-learning/actions/workflows/ci.yml/badge.svg)](https://github.com/ayusprasad/mlflow-learning/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![MLflow](https://img.shields.io/badge/Experiment%20tracking-MLflow-0194E2?logo=mlflow&logoColor=white)
![scikit--learn](https://img.shields.io/badge/Models-scikit--learn-F7931E?logo=scikitlearn&logoColor=white)

An experiment-tracking sandbox that trains scikit-learn classifiers and records model parameters, metrics, and artifacts with MLflow.

## Included experiments

- Wine classification with a local MLflow tracking server.
- DagsHub-backed tracking examples for hosted experiments.
- Breast-cancer classification with grid-search tuning.
- A committed confusion matrix image as a sample report; runtime run databases and model artifacts are generated locally.

## Run locally

### Prerequisites

- Python 3.11 or newer
- Optional: DagsHub account and token for hosted tracking

### Install

~~~bash
git clone https://github.com/ayusprasad/mlflow-learning.git
cd mlflow-learning
python -m venv .venv
python -m pip install -r requirements.txt
~~~

Start a local MLflow server in one terminal:

~~~bash
mlflow server --host 127.0.0.1 --port 5000
~~~

Run a local experiment in another terminal:

~~~bash
python src/file1.py
~~~

The DagsHub scripts require an authorized tracking URI and credentials in the environment. Copy .env.example to .env and set the required values locally. Never commit tokens.

## Quality checks

~~~bash
pytest -q
ruff check --select E4,E7,E9,F src tests
python -m compileall -q src tests
~~~

CI uses the built-in Wine dataset for a local model smoke test. It does not contact DagsHub.
