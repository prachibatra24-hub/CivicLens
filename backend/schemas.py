# backend/schemas.py
"""Pydantic models for the CivicLens detection API.
These models are used for request validation and to format the JSON
response returned by `/api/ai/detect`.
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List

class DetectionItem(BaseModel):
    """Single detection result.

    * ``class_name`` – human‑readable label (e.g., "pothole").
    * ``confidence`` – probability between 0 and 1.
    * ``bbox`` – bounding box expressed as ``[x_min, y_min, width, height]`` in
      pixel coordinates (origin at top‑left corner).
    """

    class_name: str = Field(..., description="Detected class label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    bbox: List[int] = Field(..., description="[x_min, y_min, width, height] in pixels")

class DetectionResult(BaseModel):
    """Full JSON payload returned by the detection endpoint.

    * ``width`` / ``height`` – dimensions of the input image.
    * ``detections`` – list of ``DetectionItem``; can be empty when nothing is found.
    """

    width: int = Field(..., description="Image width in pixels")
    height: int = Field(..., description="Image height in pixels")
    detections: List[DetectionItem] = Field(..., description="List of detections (may be empty)")
