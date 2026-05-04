from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from pipeline_runner import JobStore, run_generate_job, run_measure_job

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)

app = FastAPI(title="Custom Shoe Tree Generator", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("/tmp/custom_shoe_tree_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

store = JobStore()


@app.post("/api/upload")
async def upload_scan(scan: UploadFile = File(...)) -> dict[str, Any]:
    if not scan.filename or not scan.filename.lower().endswith(".obj"):
        raise HTTPException(status_code=400, detail="Please upload an OBJ file.")

    job_id = str(uuid.uuid4())
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Preserve original filename so scan_id stays clean
    scan_path = job_dir / scan.filename
    content = await scan.read()
    scan_path.write_bytes(content)

    store.create_job(job_id, str(scan_path))
    asyncio.create_task(run_measure_job(job_id, scan_path, store))

    return {"job_id": job_id, "status": "measuring"}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str) -> dict[str, Any]:
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


class GenerateRequest(BaseModel):
    measurements: dict[str, Any]
    allowance_mm: float = 3.0
    split_for_print: bool = False


@app.post("/api/generate/{job_id}")
async def generate(job_id: str, body: GenerateRequest) -> dict[str, Any]:
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] not in ("measured",):
        raise HTTPException(status_code=400, detail=f"Cannot generate from status '{job['status']}'.")

    store.update_job(job_id, status="generating", progress=0, message="Queued…")
    asyncio.create_task(run_generate_job(job_id, body.measurements, body.allowance_mm, body.split_for_print, store))
    return {"job_id": job_id, "status": "generating"}


@app.get("/api/download/{job_id}/stl")
async def download_stl(job_id: str) -> FileResponse:
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail="STL not ready yet.")
    stl_path = Path(job["stl_path"])
    if not stl_path.exists():
        raise HTTPException(status_code=404, detail="STL file missing on disk.")
    return FileResponse(
        str(stl_path),
        media_type="application/octet-stream",
        filename=stl_path.name,
    )


@app.get("/api/download/{job_id}/obj")
async def download_obj(job_id: str) -> FileResponse:
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail="OBJ not ready yet.")
    obj_path = Path(job["obj_path"])
    if not obj_path.exists():
        raise HTTPException(status_code=404, detail="OBJ file missing on disk.")
    return FileResponse(
        str(obj_path),
        media_type="text/plain",
        filename=obj_path.name,
        headers={"Access-Control-Allow-Origin": "*"},
    )


@app.get("/api/download/{job_id}/split/{part}")
async def download_split_stl(job_id: str, part: str) -> FileResponse:
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail="Split STL files are not ready yet.")

    split_paths = job.get("split_stl_paths") or {}
    if part not in {"heel-tabs", "toe-sockets"}:
        raise HTTPException(status_code=404, detail="Unknown split STL part.")
    split_path_value = split_paths.get(part)
    if not split_path_value:
        raise HTTPException(status_code=404, detail="Split STL file was not generated for this job.")

    split_path = Path(split_path_value)
    if not split_path.exists():
        raise HTTPException(status_code=404, detail="Split STL file missing on disk.")
    return FileResponse(
        str(split_path),
        media_type="application/octet-stream",
        filename=split_path.name,
    )
