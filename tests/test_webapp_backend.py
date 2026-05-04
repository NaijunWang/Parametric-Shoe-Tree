from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path


def test_webapp_job_store_tracks_split_outputs() -> None:
    project_root = Path(__file__).resolve().parents[1]
    backend_dir = project_root / "webapp" / "backend"
    sys.path.insert(0, str(backend_dir))
    try:
        pipeline_runner = importlib.import_module("pipeline_runner")
    finally:
        sys.path.remove(str(backend_dir))

    store = pipeline_runner.JobStore()
    store.create_job("job-1", "scan.obj")
    job = store.get_job("job-1")

    assert job["split_for_print"] is False
    assert job["split_stl_paths"] is None
    assert job["split_report_path"] is None
    assert "split_for_print" in inspect.signature(pipeline_runner.run_generate_job).parameters
