import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("Starting script...")

# Set tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")
print("MLflow tracking URI set")

# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target
print("Dataset loaded")

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
print("Train-test split completed")

# Define the params for RF model
max_depth = 100
n_estimators = 15

# Set experiment
mlflow.set_experiment('billo')
print("bagge")

try:
    with mlflow.start_run():
        print("MLflow run started")
        
        # Train model
        rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
        rf.fit(X_train, y_train)
        print("Model trained")

        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")

        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_param('max_depth', max_depth)
        mlflow.log_param('n_estimators', n_estimators)
        print("Metrics and parameters logged")

        # Creating a confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')

        # save plot
        plt.savefig("Confusion-matrix.png")
        print("Confusion matrix saved")

        # log artifacts using mlflow
        mlflow.log_artifact("Confusion-matrix.png")
        mlflow.log_artifact(__file__)
        print("Artifacts logged")

        # tags
        mlflow.set_tags({"Author": 'Vikash', "Project": "Wine Classification"})
        print("Tags set")

        # Log the model
        mlflow.sklearn.log_model(rf, "Random-Forest-Model")
        print("Model logged")

        print(f"Final accuracy: {accuracy}")

except Exception as e:
    print(f"An error occurred: {e}")