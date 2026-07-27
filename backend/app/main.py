import os
import io
import json

from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from PIL import Image
from backend.app.schemas import (
    ConformalPredictResponse,
    CuratedSampleItem,
    ConformalMetricsResponse,
    AuditTrailItem
)
from backend.app.conformal_service import engine, CALIB_PATH

app = FastAPI(
    title="Conformalized Brain Tumor Diagnostic Agent API",
    description="Statistically guaranteed brain tumor prediction sets with split-conformal calibration & abstention triage.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CURATED_DIR = 'backend/curated_samples'
os.makedirs(CURATED_DIR, exist_ok=True)
app.mount("/curated-images", StaticFiles(directory=CURATED_DIR), name="curated-images")

# Varied initial diagnostic audit log entries
audit_history: List[AuditTrailItem] = [
    AuditTrailItem(
        timestamp="20:42:15",
        sample_name="ACC-2026-0891_AX_T2.dcm",
        prediction_set=["Normal Scan"],
        is_confident=True,
        target_coverage=0.95,
        abstention_triage_flag=False,
        triage_message="✓ Confident Diagnostic State: Single label Normal Scan satisfies 95.0% coverage guarantee."
    ),
    AuditTrailItem(
        timestamp="20:48:30",
        sample_name="ACC-2026-0447_FLAIR.dcm",
        prediction_set=["Glioma", "Meningioma"],
        is_confident=False,
        target_coverage=0.95,
        abstention_triage_flag=True,
        triage_message="⚠️ Ambiguous Scan — prediction set [Glioma, Meningioma] exceeds single diagnosis. Flagged for radiologist review."
    ),
    AuditTrailItem(
        timestamp="20:54:10",
        sample_name="ACC-2026-1205_COR_T1.dcm",
        prediction_set=["Pituitary Tumor"],
        is_confident=True,
        target_coverage=0.95,
        abstention_triage_flag=False,
        triage_message="✓ Confident Diagnostic State: Single label Pituitary Tumor satisfies 95.0% coverage guarantee."
    ),
    AuditTrailItem(
        timestamp="20:59:02",
        sample_name="ACC-2026-0318_AX_T2.dcm",
        prediction_set=["Meningioma", "Glioma", "Pituitary Tumor"],
        is_confident=False,
        target_coverage=0.95,
        abstention_triage_flag=True,
        triage_message="⚠️ Ambiguous Scan — prediction set exceeds single diagnosis. Flagged for radiologist review."
    ),
]

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "system": "Conformalized Brain Tumor Diagnostic Agent",
        "author": "Sonali Priyadarshini",
        "engine_ready": engine.model is not None,
        "classes": engine.classes
    }

@app.get("/api/curated-samples", response_model=List[CuratedSampleItem])
def get_curated_samples():
    if os.path.exists(CALIB_PATH):
        with open(CALIB_PATH, 'r') as f:
            meta = json.load(f)
            samples = meta.get("curated_samples", [])
            results = []
            for s in samples:
                results.append(CuratedSampleItem(
                    id=s["id"],
                    filename=s["filename"],
                    type=s["type"],
                    label=s["label"],
                    true_class=s["true_class"],
                    expected_set=s["expected_set"],
                    image_url=f"/curated-images/{s['filename']}"
                ))
            return results
    return []

@app.get("/api/metrics", response_model=ConformalMetricsResponse)
def get_conformal_metrics():
    if os.path.exists(CALIB_PATH):
        with open(CALIB_PATH, 'r') as f:
            meta = json.load(f)
            return ConformalMetricsResponse(
                num_train=meta.get("n_train", 0),
                num_calibration=meta.get("n_cal", 0),
                num_test=meta.get("n_test", 0),
                default_alpha=meta.get("default_alpha", 0.05),
                default_target_coverage=meta.get("default_target_coverage", 0.95),
                classes=meta.get("classes", engine.classes),
                quantiles=meta.get("quantiles", {}),
                metrics=meta.get("metrics", {})
            )
    raise HTTPException(status_code=404, detail="Calibration metrics metadata not found.")

@app.post("/api/predict", response_model=ConformalPredictResponse)
async def predict_mri(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Query(None),
    alpha: float = Query(0.05, ge=0.01, le=0.20)
):
    pil_img = None
    sample_name = "Uploaded DICOM Scan"

    # Prioritize uploaded file over sample_id if both are passed
    if file is not None:
        try:
            contents = await file.read()
            if len(contents) > 0:
                pil_img = Image.open(io.BytesIO(contents)).convert('RGB')
                sample_name = file.filename or "Uploaded DICOM Scan"
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to process uploaded image file: {str(e)}")

    if pil_img is None and sample_id:
        if os.path.exists(CALIB_PATH):
            with open(CALIB_PATH, 'r') as f:
                meta = json.load(f)
                curated = meta.get("curated_samples", [])
                match = next((s for s in curated if s["id"] == sample_id), None)
                if match:
                    img_path = os.path.join(CURATED_DIR, match["filename"])
                    if os.path.exists(img_path):
                        pil_img = Image.open(img_path).convert('RGB')
                        sample_name = match["label"]

    if pil_img is None:
        raise HTTPException(status_code=400, detail="Please upload a valid MRI image file or select a curated sample ID.")

    result = engine.predict_conformal(pil_img, alpha=alpha)
    result["sample_id"] = sample_id

    # Deduplicate audit log: check if top log item matches this sample and set
    current_time = datetime.now().strftime("%H:%M:%S")
    new_item = AuditTrailItem(
        timestamp=current_time,
        sample_name=sample_name,
        prediction_set=result["prediction_set"],
        is_confident=result["is_confident"],
        target_coverage=result["target_coverage"],
        abstention_triage_flag=result["abstention_triage_flag"],
        triage_message=result["triage_message"]
    )

    if not audit_history or (audit_history[0].sample_name != sample_name or audit_history[0].prediction_set != result["prediction_set"]):
        audit_history.insert(0, new_item)

    if len(audit_history) > 15:
        audit_history.pop()

    return result

@app.get("/api/history", response_model=List[AuditTrailItem])
def get_audit_history():
    return audit_history
