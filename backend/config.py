# backend/config.py
"""Configuration for the CivicLens FastAPI backend.
It uses Pydantic BaseSettings so values can be overridden via environment
variables or a .env file. Default values work for local development.
"""

from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    # Path to the YOLO checkpoint (.pt). Adjust if your model lives elsewhere.
    model_path: str = Field(default="models/civiclens/run/weights/best.pt")
    # Maximum upload size – 10 MiB is a reasonable default for images.
    max_upload_size: int = Field(default=10 * 1024 * 1024)
    # Confidence threshold used for filtering detections.
    confidence_threshold: float = Field(default=0.25)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

def get_settings() -> Settings:
    """FastAPI dependency that provides a Settings instance."""
    return Settings()