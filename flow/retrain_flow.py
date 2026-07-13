# flows/retrain_flow.py
import os
from pathlib import Path
from dotenv import load_dotenv
from prefect import flow, task
from minio import Minio
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#  ENV ----------------------------------------------------------------------------
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# MinIO/MLflow
MINIO_ENDPOINT = f"{os.getenv('MINIO_INTERFACE')}:{os.getenv('MINIO_PORT')}"
MINIO_ACCESS_KEY = os.getenv("MINIO_ROOT_USER")
MINIO_SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD")
BUCKET_NAME = "mlflow"

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL")
os.environ["AWS_ACCESS_KEY_ID"] = MINIO_ACCESS_KEY
os.environ["AWS_SECRET_ACCESS_KEY"] = MINIO_SECRET_KEY
os.environ["MLFLOW_S3_ENDPOINT_URL"] = MLFLOW_S3_ENDPOINT_URL

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("iris_experiment")

print(f"MLflow URI: {MLFLOW_URI}, MinIO URI: {MLFLOW_S3_ENDPOINT_URL}")

# Tasks -----------------------------------------------------------------------------------------------
@task
def prepare_minio():
    """Check exist bucket for MLflow."""
    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )
    if not client.bucket_exists(BUCKET_NAME):
        client.make_bucket(BUCKET_NAME)
        print(f"Bucket '{BUCKET_NAME}' created.")
    else:
        print(f"Bucket '{BUCKET_NAME}' already exists.")

@task
def train_model():
    """Create data and models, train/test split."""
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    models = [
        RandomForestClassifier(n_estimators=100),
        LogisticRegression(max_iter=300)
    ]

    return models, X_train, y_train, X_test, y_test

@task
def train_and_register(model, X_train, y_train, X_test, y_test):
    """Train model, logging and registre in MLflow."""
    model_name = type(model).__name__

    with mlflow.start_run(run_name=model_name) as run:
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_params(model.get_params())

        # log in MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name
        )

    # versionage
    client = MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"{model_name} registered as Production, accuracy={accuracy:.4f}")


# Flow ----------------------------------------------------------
@flow(name="Auto Retrain Flow")
def retrain_flow():
    prepare_minio()
    models, X_train, y_train, X_test, y_test = train_model()
    for model in models:
        train_and_register(model, X_train, y_train, X_test, y_test)
    print("All models trained and uploaded.")

# for local test
if __name__ == "__main__":
    retrain_flow()