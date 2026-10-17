# backend/inference.py
"""Helper functions to load a YOLO model and run inference.
The functions are deliberately lightweight – they avoid any heavy
initialisation on each request and keep the model object in memory.
"""

from typing import List, Tuple, Union
from pathlib import Path
import io

from ultralytics import YOLO
from .schemas import DetectionItem

# ------------------------------------------------------------
# Model loading
# ------------------------------------------------------------

def get_model(weights_path: str) -> YOLO:
    """Load a YOLO model from a ``.pt`` checkpoint.

    Parameters
    ----------
    weights_path: str
        Absolute or relative path to the Ultralytics checkpoint.
    Returns
    -------
    YOLO
        An instantiated model ready for ``predict`` / ``val`` calls.
    """
    model = YOLO(weights_path)
    # Ensure the model runs on the appropriate device (CPU or GPU).
    # Ultralytics automatically selects CUDA if available.
    return model

# ------------------------------------------------------------
# Prediction wrapper
# ------------------------------------------------------------

def _format_detection(result) -> DetectionItem:
    """Convert a single Ultralytics result to our DetectionItem model.
    ``result.boxes`` holds tensor information: ``xywh`` (center‑based).
    We convert to ``[x_min, y_min, width, height]`` pixel coordinates.
    """
    # Ultralytics result boxes are in xyxy format after .xyxy attribute.
    # However, we will use .xywh to get centre based, then compute top‑left.
    box = result.boxes[0]
    # ``box.xywh`` returns a tensor [[x_center, y_center, w, h]]
    x_c, y_c, w, h = box.xywh[0].cpu().numpy()
    x_min = int(round(x_c - w / 2))
    y_min = int(round(y_c - h / 2))
    width = int(round(w))
    height = int(round(h))
    cls_id = int(box.cls[0].cpu().numpy())
    conf = float(box.conf[0].cpu().numpy())
    # Resolve class name from model.names dict (if present)
    class_name = str(result.names.get(cls_id, cls_id))
    return DetectionItem(class_name=class_name, confidence=conf, bbox=[x_min, y_min, width, height])


def predict_image(
    model: YOLO,
    image_bytes: bytes,
    confidence_thresh: float = 0.25,
    return_annotated: bool = False,
) -> Union[List[DetectionItem], bytes]:
    """Run inference on a raw image.

    Parameters
    ----------
    model: YOLO
        The loaded YOLO model.
    image_bytes: bytes
        Raw image data (as read from an ``UploadFile``).
    confidence_thresh: float, optional
        Minimum confidence for a detection to be kept.
    return_annotated: bool, default ``False``
        When ``True`` the function returns a PNG byte stream containing the
        image with drawn bounding boxes. When ``False`` it returns a list of
        ``DetectionItem`` objects.
    """
    # Ultralytics can accept a bytes object via ``source=io.BytesIO``.
    results = model.predict(source=io.BytesIO(image_bytes), conf=confidence_thresh, save=False)
    # ``results`` is a list with a single ``Results`` object because we passed one image.
    result = results[0]
    if return_annotated:
        # Ultralytics can paint the image and we retrieve the annotated image as bytes.
        # ``result.plot()`` draws boxes onto ``result.imgs[0]`` (numpy array).
        annotated = result.plot()  # modifies result.imgs[0] in‑place and returns the array
        # Encode to PNG bytes
        from PIL import Image
        import numpy as np
        pil_img = Image.fromarray(annotated)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return buf.getvalue()
    else:
        detections: List[DetectionItem] = []
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                # ``box`` is a single detection; we need to build a temporary result wrapper.
                # Ultralytics stores attributes per box; we mimic the structure used in _format_detection.
                # Build a lightweight object with required attributes.
                class SimpleResult:
                    def __init__(self, box):
                        self.boxes = [box]
                        self.names = model.names
                simple = SimpleResult(box)
                detections.append(_format_detection(simple))
        return detections
