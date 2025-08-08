import os, argparse, yaml
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
from tabulate import tabulate
from sklearn.isotonic import IsotonicRegression
import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

from src.model import FeatureSpec, GAMTrainer, GAMPyFunc
from src.eval import evaluate_ranking
from src.plots import plot_partial_effect, plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from src.report import write_markdown, write_html

def autodetect_columns(df: pd.DataFrame, target: str):
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != target]
    cat_cols = [c for c in df.columns if c not in num_cols + [target]]
    return num_cols, cat_cols

def load_config(path: str):
    with open(path) as f:
        return yaml.safe_load(f)

def train_and_log(cfg):
    df = pd.read_csv(cfg["data_path"])
    target = cfg["target"]
    # map target to 0/1 if string
    if df[target].dtype == object:
        df[target] = (df[target].astype(str).str.lower().isin(["bad", "1", "default"])).astype(int)

    num_cols = cfg.get("numeric") or []
    cat_cols = cfg.get("categorical") or []
    if not num_cols and not cat_cols:
        num_cols, cat_cols = autodetect_columns(df, target)

    spec = FeatureSpec(numeric=num_cols, categorical=cat_cols, target=target)
    trainer = GAMTrainer(spec=spec, test_size=cfg["test_size"], random_state=cfg["seed"])
    train_df, test_df = trainer.fit(df)

    metrics = evaluate_ranking(test_df["y"].to_numpy(), test_df["p"].to_numpy(), k=100)

    # Optional calibration
    if cfg.get("calibrate", False):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(train_df["p"], train_df["y"])
        test_df["p"] = ir.transform(test_df["p"])
        metrics = evaluate_ranking(test_df["y"].to_numpy(), test_df["p"].to_numpy(), k=100)

    # Plots (limit amount)
    feat_list = (num_cols + cat_cols)[: cfg.get("max_plots", 8)]
    plots = {}
    for feat in feat_list:
        x, y, ci = trainer.partial_effect(feat)
        plots[feat] = plot_partial_effect(feat, x, y, ci)

    # Additional plots
    y_true = test_df["y"].to_numpy()
    y_pred = (test_df["p"].to_numpy() > 0.5).astype(int)
    y_prob = test_df["p"].to_numpy()
    class_names = ['Good', 'Bad']

    plots["confusion_matrix"] = plot_confusion_matrix(y_true, y_pred, class_names)
    plots["roc_curve"] = plot_roc_curve(y_true, y_prob)
    plots["precision_recall_curve"] = plot_precision_recall_curve(y_true, y_prob)

    # Top10 table
    top10 = test_df.sort_values("p", ascending=True).head(10).copy()
    top_table_md = tabulate(top10, headers="keys", tablefmt="github", showindex=False)

    # Report
    meta = {
        "n_obs": len(df),
        "n_num": len(num_cols),
        "n_cat": len(cat_cols),
        "metrics": metrics,
    }
    write_markdown("reports/report.md", meta, plots)
    write_html("reports/report.html", meta, plots, top_table_md)

    # MLflow logging
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "credit-gam-ranking")
    mlflow.set_experiment(exp_name)

    with mlflow.start_run() as run:
        mlflow.log_params({
            "model":"LogisticGAM",
            "seed": cfg["seed"],
            "test_size": cfg["test_size"],
            "numeric": ",".join(num_cols),
            "categorical": ",".join(cat_cols),
        })
        mlflow.log_metrics(metrics)
        # artifacts
        mlflow.log_artifact("reports/report.md")
        if os.path.exists("reports/report.pdf"):
            mlflow.log_artifact("reports/report.pdf")
        for p in plot_paths.values():
            mlflow.log_artifact(p, artifact_path="plots")
        # Register model as PyFunc including preprocessing
        pyfunc = GAMPyFunc(trainer.gam, trainer.spec.numeric, trainer.spec.categorical,
                           trainer.label_encoders, trainer.numeric_median)
        # Build an input example (median row)
        example = pd.DataFrame([{
            **{c: float(df[c].median()) for c in trainer.spec.numeric},
            **{c: str(df[c].astype(str).mode(dropna=True).iloc[0] if not df[c].empty else "NA_SENTINEL") for c in trainer.spec.categorical}
        }])
        signature = infer_signature(example, pyfunc.predict(None, example))
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=pyfunc,
            input_example=example,
            signature=signature,
            registered_model_name="credit_gam"
        )
        run_id = run.info.run_id
    # Return identifiers for gating
    return {"run_id": run_id, "registered_model": "credit_gam"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", default="configs/base.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    info = train_and_log(cfg)
    print("TRAIN_OK", info)

if __name__ == "__main__":
    main()
