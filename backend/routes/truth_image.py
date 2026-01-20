from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request

from models import TruthImage
from services.face_service import embedding_to_json, extract_face_embedding

logger = logging.getLogger(__name__)

truth_image_bp = Blueprint("truth_image", __name__)


@truth_image_bp.post("/truth-image")
def upload_truth_image():
    """
    Upload Truth Image:
    - saves image to uploads/truth/
    - extracts embedding
    - stores embedding in DB (overwrites by inserting new row; latest row is used)
    """
    if "file" not in request.files:
        return jsonify({"error": "Missing file field"}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    uploads_dir = Path(current_app.config["UPLOAD_FOLDER"]) / "truth"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    ext = os.path.splitext(file.filename)[1].lower() or ".jpg"
    filename = f"truth_{uuid.uuid4().hex}{ext}"
    abs_path = uploads_dir / filename
    file.save(str(abs_path))
    rel_path = str(Path("truth") / filename)

    embedding = extract_face_embedding(str(abs_path))
    if embedding is None:
        # Keep file saved for visibility/debugging.
        return jsonify({"error": "No face found in truth image"}), 400

    SessionLocal = current_app.session_local  # type: ignore[attr-defined]
    with SessionLocal() as db:
        truth = TruthImage(image_path=rel_path, embedding=embedding_to_json(embedding))
        db.add(truth)
        db.commit()
        db.refresh(truth)

    return jsonify(
        {
            "message": "Truth image uploaded",
            "truth_image": {
                "id": truth.id,
                "image_path": truth.image_path,
                "public_url": f"/uploads/{truth.image_path}",
            },
        }
    )


