from typing import Any, Dict

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "legal-rag-queries"

if MLFLOW_AVAILABLE:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_query_run(endpoint: str, params: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    """Log one query/chat/document call as an MLflow run. Never raises —
    tracking is observability, not a hard dependency of the pipeline; a
    logging failure (or mlflow not installed) must not break a real answer."""
    if not MLFLOW_AVAILABLE:
        return
    try:
        with mlflow.start_run(run_name=endpoint):
            mlflow.log_param("endpoint", endpoint)
            for k, v in params.items():
                if v is not None:
                    mlflow.log_param(k, v)
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and v is not None:
                    mlflow.log_metric(k, v)
    except Exception:
        pass
