"""Download the most recent model logged to MLflow (DagsHub) into the image.

Reads MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD
from the environment. Optionally restrict the search with MLFLOW_EXPERIMENT_NAME.
"""

import json
import os
import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

MODEL_ARTIFACT = os.getenv("MODEL_ARTIFACT", "model")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/app/artifacts"))
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "").strip()


def experiment_ids(client: MlflowClient) -> list[str]:
    if EXPERIMENT_NAME:
        exp = client.get_experiment_by_name(EXPERIMENT_NAME)
        if exp is None:
            sys.exit(f"Experiment '{EXPERIMENT_NAME}' not found on {mlflow.get_tracking_uri()}")
        return [exp.experiment_id]
    return [exp.experiment_id for exp in client.search_experiments()]


def main() -> None:
    client = MlflowClient()
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")

    runs = client.search_runs(
        experiment_ids=experiment_ids(client),
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=100,
    )
    for run in runs:
        run_id = run.info.run_id
        if any(a.path == MODEL_ARTIFACT for a in client.list_artifacts(run_id)):
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            local_path = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path=MODEL_ARTIFACT, dst_path=str(OUTPUT_DIR)
            )
            (OUTPUT_DIR / "model_info.json").write_text(
                json.dumps({"run_id": run_id, "metrics": run.data.metrics, "params": run.data.params})
            )
            print(f"Downloaded model from run {run_id} to {local_path}")
            return

    sys.exit(f"No finished run with a '{MODEL_ARTIFACT}' artifact was found")


if __name__ == "__main__":
    main()
