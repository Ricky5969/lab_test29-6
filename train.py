# This code train the random forest model using parameter in params.yaml file
# Apply to random forest dataset

import yaml
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def load_params(path="params.yaml"):
    with open(path, "r") as file:
        params = yaml.safe_load(file)
    return params


def main():
    # 1. Load parameters
    params = load_params()
    model_params = params["model"]

    # 2. Load dataset
    df = pd.read_csv("data/breast-cancer.csv")

    X = df.drop(columns=["target"])
    y = df["target"]

    # 3. Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state= model_params["random_state"],
        stratify=y
    )

    # 4. Start MLflow experiment
    mlflow.set_experiment("Experiment 101 to evaluate random forest for breast cancer dataset")

    with mlflow.start_run(run_name="RF_Setting_3_Full_Data"):
        # Log parameters
        mlflow.log_param("n_estimators", model_params["n_estimators"])
        mlflow.log_param("max_depth", model_params["max_depth"])
        mlflow.log_param("random_state", model_params["random_state"])

        mlflow.log_param("dataset_name", "Original sklearn breast cancer dataset")
        mlflow.log_param("dataset_size", "570 samples, 30 features")

        # 5. Train model
        model = RandomForestClassifier(
            n_estimators=model_params["n_estimators"],
            max_depth=model_params["max_depth"],
            random_state=model_params["random_state"]
        )

        model.fit(X_train, y_train)

        # 6. Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Accuracy: {accuracy:.4f}")

        # Log metric to MLflow
        mlflow.log_metric("accuracy", accuracy)

        # 7. Save model locally
        Path("models").mkdir(exist_ok=True)
        model_path = "models/model.pkl"
        joblib.dump(model, model_path)

        # Log model to MLflow
        mlflow.sklearn.log_model(model, name="model-rf")

        # Log model file as MLflow artifact
        mlflow.log_artifact(model_path)

        print("Model saved to models/model.pkl")


if __name__ == "__main__":
    main()
