from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import pandas as pd
import mlflow
import dagshub
import os
from mlflow.tracking import MlflowClient

# Initialize DagsHub
dagshub.init(repo_owner='ayusprasad', repo_name='mlflow-learning', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/ayusprasad/mlflow-learning.mlflow")

# Set up your remote MLflow server (replace with your server details)
REMOTE_TRACKING_URI = "http://your-remote-server:5000"
remote_client = MlflowClient(REMOTE_TRACKING_URI)

# Load the Breast Cancer dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the RandomForestClassifier model
rf = RandomForestClassifier(random_state=42)

# Defining the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20, 30]
}

# Applying GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Set experiment for both servers
mlflow.set_experiment('breast-cancer-rf-hp')

# Create a function to log to both servers
def log_to_both_servers(params, metrics, artifacts, tags, model, run_name=None):
    # Start run on DagsHub
    with mlflow.start_run(run_name=run_name) as dagshub_run:
        # Log to DagsHub
        if params:
            mlflow.log_params(params)
        if metrics:
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
        if artifacts:
            for artifact in artifacts:
                mlflow.log_artifact(artifact)
        if tags:
            mlflow.set_tags(tags)
        if model:
            mlflow.sklearn.log_model(model, "model")
        
        # Get the DagsHub run ID for reference
        dagshub_run_id = dagshub_run.info.run_id
        
    # Log to remote server
    remote_exp = remote_client.get_experiment_by_name('breast-cancer-rf-hp')
    if remote_exp is None:
        # Create experiment if it doesn't exist
        remote_exp_id = remote_client.create_experiment('breast-cancer-rf-hp')
    else:
        remote_exp_id = remote_exp.experiment_id
    
    # Start run on remote server
    remote_run = remote_client.create_run(remote_exp_id, run_name=run_name)
    remote_run_id = remote_run.info.run_id
    
    # Log to remote server
    if params:
        for key, value in params.items():
            remote_client.log_param(remote_run_id, key, value)
    if metrics:
        for key, value in metrics.items():
            remote_client.log_metric(remote_run_id, key, value)
    if tags:
        remote_client.set_tags(remote_run_id, tags)
    if artifacts:
        # For artifacts, you'd need to upload them to the remote server
        # This is more complex and might require using the MLflow API directly
        pass
    if model:
        # Log model to remote server
        mlflow.sklearn.save_model(model, "/tmp/model")
        remote_client.log_artifacts(remote_run_id, "/tmp/model", "model")
    
    # End the remote run
    remote_client.set_terminated(remote_run_id)
    
    return dagshub_run_id, remote_run_id

# Run the grid search
grid_search.fit(X_train, y_train)

# Log all the child runs to both servers
for i in range(len(grid_search.cv_results_['params'])):
    params = grid_search.cv_results_["params"][i]
    metrics = {"accuracy": grid_search.cv_results_["mean_test_score"][i]}
    
    # Log to both servers
    log_to_both_servers(
        params=params,
        metrics=metrics,
        artifacts=None,
        tags=None,
        model=None,
        run_name=f"grid_search_{i}"
    )

# Displaying the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Log the best model and results to both servers
dagshub_run_id, remote_run_id = log_to_both_servers(
    params=best_params,
    metrics={"accuracy": best_score},
    artifacts=[__file__],  # Add other artifacts as needed
    tags={"author": "Vikash Das", "model_type": "RandomForest"},
    model=grid_search.best_estimator_,
    run_name="best_model"
)

print(f"Best parameters: {best_params}")
print(f"Best score: {best_score}")
print(f"DagsHub Run ID: {dagshub_run_id}")
print(f"Remote Server Run ID: {remote_run_id}")