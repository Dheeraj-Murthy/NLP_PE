from typing import Any, Dict

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "legal-rag-queries"
INGESTION_EXPERIMENT_NAME = "legal-rag-ingestion"
BENCHMARK_EXPERIMENT_NAME = "legal-rag-benchmarks"

if MLFLOW_AVAILABLE:
    mlflow.set_tracking_uri(TRACKING_URI)


def _experiment_id(name: str) -> str:
    """Resolves (creating if needed) an experiment id by name, without
    touching mlflow's global "active experiment" — that's process-wide
    mutable state, and both log_* functions below run in the same process
    (and even interleave: ingestion scripts and the API can share a venv/
    import graph), so one calling mlflow.set_experiment() would silently
    redirect the other's runs into the wrong experiment."""
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id
    return mlflow.create_experiment(name)


def log_query_run(endpoint: str, params: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    """Log one query/chat/document call as an MLflow run. Never raises —
    tracking is observability, not a hard dependency of the pipeline; a
    logging failure (or mlflow not installed) must not break a real answer."""
    if not MLFLOW_AVAILABLE:
        return
    try:
        with mlflow.start_run(run_name=endpoint, experiment_id=_experiment_id(EXPERIMENT_NAME)):
            mlflow.log_param("endpoint", endpoint)
            for k, v in params.items():
                if v is not None:
                    mlflow.log_param(k, v)
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and v is not None:
                    mlflow.log_metric(k, v)
    except Exception:
        pass


def log_ingestion_run(script: str, params: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    """Log one corpus-ingestion script run (ingest.py, ingest_statutes.py)
    as an MLflow run, under a separate experiment from live query traffic —
    so a graph of ingestion runs over time (corpus growth, failure rate,
    chunking yield) doesn't get mixed in with per-query request logs.
    Never raises, same reasoning as log_query_run: ingestion already
    succeeded or failed on its own merits by the time this is called, a
    tracking failure shouldn't turn a real run into a reported failure."""
    if not MLFLOW_AVAILABLE:
        return
    try:
        experiment_id = _experiment_id(INGESTION_EXPERIMENT_NAME)
        with mlflow.start_run(run_name=script, experiment_id=experiment_id):
            mlflow.log_param("script", script)
            for k, v in params.items():
                if v is not None:
                    mlflow.log_param(k, v)
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and v is not None:
                    mlflow.log_metric(k, v)
    except Exception:
        pass


def log_benchmark_run(script: str, params: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    """Log one benchmark run (benchmark/retrieval_eval.py, future
    benchmark/answer_eval.py) as an MLflow run, under its own experiment so
    repeated benchmark runs are comparable as retrieval config or the
    corpus changes, without mixing into live query or ingestion traffic.
    Never raises, same reasoning as log_query_run/log_ingestion_run."""
    if not MLFLOW_AVAILABLE:
        return
    try:
        experiment_id = _experiment_id(BENCHMARK_EXPERIMENT_NAME)
        with mlflow.start_run(run_name=script, experiment_id=experiment_id):
            mlflow.log_param("script", script)
            for k, v in params.items():
                if v is not None:
                    mlflow.log_param(k, v)
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and v is not None:
                    mlflow.log_metric(k, v)
    except Exception:
        pass
