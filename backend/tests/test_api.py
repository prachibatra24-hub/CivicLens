import io
from fastapi.testclient import TestClient
from pathlib import Path

from backend.main import app
from backend.config import Settings

client = TestClient(app)

def test_root_endpoint():
    response = client.get('/')
    assert response.status_code == 200
    assert response.json() == {"message": "CivicLens AI Backend is running"}

def test_detect_without_model(tmp_path: Path):
    # Point Settings to a non‑existent checkpoint to trigger 503
    settings = Settings(model_path=str(tmp_path / "nonexistent.pt"))
    # Override dependency to use our settings instance
    app.dependency_overrides[Settings] = lambda: settings
    # Minimal valid PNG bytes (1x1 pixel)
    png_bytes = (b"\x89PNG\r\n\x1a\n"
                 b"\x00\x00\x00\rIHDR"
                 b"\x00\x00\x00\x01"
                 b"\x00\x00\x00\x01"
                 b"\x08\x06\x00\x00\x00"
                 b"\x1f\x15\xc4\x89"
                 b"\x00\x00\x00\x0aIDATx\x9cc``\x00\x00\x00\x02\x00\x01"
                 b"\xe2\x5c\x9d\x1b"
                 b"\x00\x00\x00\x00IEND\xaeB`\x82")
    files = {"file": ("test.png", png_bytes, "image/png")}
    response = client.post("/api/ai/detect", files=files)
    assert response.status_code == 503
    assert response.json()["detail"] == "Model not available"
    # Cleanup override
    app.dependency_overrides.clear()
