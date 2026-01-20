import os
from pathlib import Path


class Config:
    """
    Small, intern-friendly config.
    In this environment `.env` may be blocked from being created; we still read env vars if present.
    """

    BASE_DIR = Path(__file__).resolve().parent

    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret")
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///face_matching.db")

    # Uploads live inside backend/uploads/
    UPLOAD_FOLDER_NAME = os.getenv("UPLOAD_FOLDER", "uploads")
    UPLOAD_FOLDER = str(BASE_DIR / UPLOAD_FOLDER_NAME)

    FACE_DISTANCE_THRESHOLD = float(os.getenv("FACE_DISTANCE_THRESHOLD", "0.6"))

    FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")


