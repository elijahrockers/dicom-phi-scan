"""FastAPI server for DICOM PHI scanning."""

import tempfile
from pathlib import Path

import anthropic
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from .agent import run_agent, run_direct_scan
from .models import ScanReport

app = FastAPI(
    title="DICOM PHI Screening Agent",
    description="Agentic pipeline for scanning DICOM datasets for PHI",
    version="0.1.0",
)

client = anthropic.Anthropic()


@app.post("/scan", response_model=ScanReport)
async def scan_dicom(
    file: UploadFile = File(...),
    mode: str = "agent",
):
    """Upload and scan a DICOM file for PHI.

    Args:
        file: DICOM file upload.
        mode: 'agent' for Claude-orchestrated scan, 'direct' for sequential scan.
    """
    if not file.filename or not file.filename.lower().endswith((".dcm", ".dicom")):
        raise HTTPException(400, "File must be a DICOM file (.dcm)")

    with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        if mode == "agent":
            report = run_agent(tmp_path, client)
        else:
            report = run_direct_scan(tmp_path, client)
        return report
    except Exception as e:
        raise HTTPException(500, f"Scan failed: {str(e)}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/health")
async def health():
    return {"status": "ok"}
