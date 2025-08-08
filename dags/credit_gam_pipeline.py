from airflow import DAG
from airflow.decorators import task
from datetime import datetime
import os, subprocess, json
import mlflow
from mlflow.tracking import MlflowClient

default_args = {"owner":"ml", "retries":0}
with DAG("credit_gam_pipeline", start_date=datetime(2025,8,1), schedule="@daily", catchup=False, default_args=default_args):
    @task
    def feature_build():
        # descarga/actualiza dataset
        cmd = ["python","/opt/airflow/src/../scripts/fetch_german_credit.py"]
        out = subprocess.run(cmd, capture_output=True, text=True)
        return out.stdout[-4000:]

    @task
    def train():
        # Ejecuta entrenamiento + logging + registro de modelo
        cmd = ["python","/opt/airflow/src/main.py","--config","/opt/airflow/configs/base.yaml"]
        out = subprocess.run(cmd, capture_output=True, text=True)
        print(out.stdout)
        return out.stdout[-4000:]

    @task
    def gate_and_promote():
        # Consulta último run y promueve a Staging/Production si supera umbrales
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI","http://mlflow:5000")
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()

        exp_name = os.environ.get("MLFLOW_EXPERIMENT_NAME","credit-gam-ranking")
        exp = client.get_experiment_by_name(exp_name)
        assert exp is not None, "Experiment not found"
        runs = client.search_runs([exp.experiment_id], order_by=["attributes.start_time DESC"], max_results=1)
        if not runs:
            return "NO_RUNS"

        run = runs[0]
        metrics = run.data.metrics
        ndcg = metrics.get("ndcg@100", 0)
        brier = metrics.get("brier", 1)
        promote = (ndcg >= float(os.environ.get("PROMOTE_NDCG", "0.85"))) and (brier <= float(os.environ.get("PROMOTE_BRIER","0.18")))

        # get latest model version for 'credit_gam'
        mv = client.get_latest_versions(name="credit_gam", stages=[])
        if not mv:
            return f"NO_MODEL_VERSION for run {run.info.run_id}"
        # assume the last is the newest
        latest = sorted(mv, key=lambda x: int(x.version))[-1]

        stage = "Production" if promote else "Staging"
        client.transition_model_version_stage(name="credit_gam", version=latest.version, stage=stage, archive_existing_versions=False)
        return f"PROMOTED:{stage}:v{latest.version}"

    @task
    def publish_report():
        # convierte report.md a PDF si hay pandoc
        if os.system("pandoc -v > /dev/null 2>&1") == 0:
            os.system("pandoc /opt/airflow/reports/report.md -o /opt/airflow/reports/report.pdf --pdf-engine=xelatex || pandoc /opt/airflow/reports/report.md -o /opt/airflow/reports/report.pdf")
            return "PDF_OK"
        return "PANDOC_MISSING"

    feature_build() >> train() >> gate_and_promote() >> publish_report()
