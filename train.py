import argparse
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def parse_max_depth(value: str):
    if value.lower() in {"none", "null"}:
        return None
    return int(value)


def main():
    parser = argparse.ArgumentParser(description="Train and register an Iris classifier")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=parse_max_depth, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("iris-classification")

    X, y = load_iris(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.random_state, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
    )

    with mlflow.start_run() as run:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        weighted_f1 = f1_score(y_test, predictions, average="weighted")
        mlflow.log_params({
            "n_estimators": args.n_estimators,
            "max_depth": "None" if args.max_depth is None else args.max_depth,
            "random_state": args.random_state,
        })
        mlflow.log_metrics({"accuracy": accuracy, "weighted_f1": weighted_f1})

        model_path = Path("model.joblib")
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="joblib")
        signature = infer_signature(X_train, model.predict(X_train))
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="iris-model",
            signature=signature,
            input_example=X_test.iloc[:3],
            registered_model_name="iris-classifier",
            # This is the built-in tree data object produced by our own fitted model.
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        print(f"Run ID: {run.info.run_id}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Weighted F1: {weighted_f1:.4f}")
        print(f"Registered model version: {model_info.registered_model_version}")
        print(f"Model saved to {model_path.resolve()}")


if __name__ == "__main__":
    main()
