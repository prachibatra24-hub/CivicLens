# backend/main.py
"""FastAPI entry point for CivicLens AI inference service.
It loads the YOLO model (if a checkpoint is present) at start‑up and
exposes a POST endpoint `/api/ai/detect` that accepts an image file.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import io

from .config import Settings, get_settings
from .schemas import DetectionResult
from .inference import get_model, predict_image

app = FastAPI(title="CivicLens AI Backend")

@app.get("/")
async def root():
    """Simple health check endpoint."""
    return {"message": "CivicLens AI Backend is running"}

# ---- CORS configuration (allow local frontend) ----
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once on startup (or mark as unavailable)
model = None
settings = get_settings()
model_path = Path(settings.model_path)
if model_path.is_file():
    try:
    # Ensure tables exist
    from .database import Base, engine
    Base.metadata.create_all(bind=engine)
    except Exception as exc:
        print(f"[ERROR] Failed to load YOLO model: {exc}")
        model = None
else:
    print(f"[INFO] No YOLO checkpoint found at {model_path}. Inference disabled.")

@app.post("/api/ai/detect", response_model=DetectionResult)
async def detect(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
) -> DetectionResult:
    """Accept an image, run YOLO detection, and return results.

    * Validates MIME type (must be an image).
    * Limits file size to `settings.max_upload_size` bytes.
    * Returns 503 if the model is unavailable.
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model not available")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image")
    contents = await file.read()
    if len(contents) > settings.max_upload_size:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large (>{settings.max_upload_size // 1024 // 1024} MiB)"
        )
    # Verify image can be opened
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(contents))
        img.verify()
        img = Image.open(io.BytesIO(contents))  # reopen after verify to get size
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    detections = predict_image(model, contents, settings.confidence_threshold)
    return DetectionResult(width=img.width, height=img.height, detections=detections)

# Optional endpoint that returns an annotated image
@app.post("/api/ai/detect/annotated")
async def detect_annotated(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    if not model:
        raise HTTPException(status_code=503, detail="Model not available")
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image")
    contents = await file.read()
    if len(contents) > settings.max_upload_size:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large (>{settings.max_upload_size // 1024 // 1024} MiB)"
        )
    annotated_bytes = predict_image(
        model, contents, settings.confidence_threshold, return_annotated=True
    )
    return StreamingResponse(io.BytesIO(annotated_bytes), media_type="image/png")
