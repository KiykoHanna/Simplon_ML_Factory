import os
import boto3
import mlflow
import mlflow.sklearn

from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#  ENV 
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"

def in_docker():
    try:
        # если есть файл /.dockerenv, значит внутри контейнера
        return os.path.exists("/.dockerenv")
    except:
        return False

if in_docker():
    MLFLOW_URI = "http://mlflow:5000"
    MINIO_URI = "http://minio:9000"
else:
    MLFLOW_URI = "http://localhost:5000"
    MINIO_URI = "http://localhost:9000"

os.environ["MLFLOW_S3_ENDPOINT_URL"] = MINIO_URI
mlflow.set_tracking_uri(MLFLOW_URI)
print(f"MLflow URI: {MLFLOW_URI}, MinIO URI: {MINIO_URI}")


#  MINIO
def prepare_minio():
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"],
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"]
        )

        buckets = [b["Name"] for b in s3.list_buckets()["Buckets"]]

        if "mlflow" not in buckets:
            s3.create_bucket(Bucket="mlflow")
            print("Bucket 'mlflow' created successfully.")
        else:
            print("Bucket 'mlflow' exist.")

    except Exception as e:
        print(f"Error MinIO: {e}")

#  TRAIN ---------------------------------------------------------------
def train_and_register():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("iris_experiment")

    # параметры модели
    # params = {"n_estimators": n_estimators}
    params = {"max_iter": 300}

    with mlflow.start_run() as run:
        # обучение
        model.fit(X_train, y_train)

        # логирование
        mlflow.log_params(params)
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)

        # лог модели + регистрация
        model_name = "iris_model"

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=model_name,
        )

    #  ALIAS ---------------------------------------------------
    client = MlflowClient()

    latest_version = client.get_latest_versions(
        model_name, stages=["None"]
    )[0].version

    AUTO_PROMOTE = True  # False для ручного переключения

    if AUTO_PROMOTE:
        client.set_registered_model_alias(model_name, "Production", latest_version)

# DATA ---------------------------------------------------------
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#  MODEL  --------------------------------------------------
# n_estimators = 100
# model = RandomForestClassifier(n_estimators=n_estimators)
model = LogisticRegression(max_iter=300)

#  RUN -------------------------------------------------------------------
if __name__ == "__main__":
    prepare_minio()
    train_and_register()