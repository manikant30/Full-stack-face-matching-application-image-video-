from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request
from sqlalchemy import select

from models import TestImage, TruthImage
from services.face_service import compare_embeddings, extract_face_embedding

logger = logging.getLogger(__name__)

test_images_bp = Blueprint("test_images", __name__)

def _to_public_url(image_path: str) -> str:
    """
    DB might contain either:
    - relative path like "images/foo.jpg"
    - absolute path like "D:\\...\\uploads\\images\\foo.jpg"
    We normalize to /uploads/<relative>.
    """
    norm = image_path.replace("\\", "/")
    if "/uploads/" in norm:
        rel = norm.split("/uploads/", 1)[1]
        return f"/uploads/{rel}"
    # already relative
    return f"/uploads/{norm.lstrip('/')}"


def _get_latest_truth_embedding(db) -> list[float] | None:
    truth = db.execute(select(TruthImage).order_by(TruthImage.id.desc())).scalars().first()
    if not truth:
        return None
    return truth.embedding_as_list()


@test_images_bp.post("/test-images")
def upload_test_images():
    """
    Upload multiple test images:
    - saves each image
    - extracts face embedding
    - compares with latest truth image embedding
    - stores match_status + confidence_score
    """
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files provided"}), 400

    uploads_dir = Path(current_app.config["UPLOAD_FOLDER"]) / "images"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    SessionLocal = current_app.session_local  # type: ignore[attr-defined]
    threshold = float(current_app.config["FACE_DISTANCE_THRESHOLD"])

    results = []
    with SessionLocal() as db:
        truth_embedding = _get_latest_truth_embedding(db)

        for file in files:
            if not file or file.filename == "":
                continue

            ext = os.path.splitext(file.filename)[1].lower() or ".jpg"
            filename = f"img_{uuid.uuid4().hex}{ext}"
            abs_path = uploads_dir / filename
            file.save(str(abs_path))
            rel_path = str(Path("images") / filename)

            test_embedding = extract_face_embedding(str(abs_path))
            match = compare_embeddings(truth_embedding, test_embedding, threshold=threshold)

            row = TestImage(
                image_path=rel_path,
                match_status=match.match_status,
                confidence_score=float(match.confidence_score),
            )
            db.add(row)
            db.flush()

            results.append(
                {
                    "id": row.id,
                    "image_path": row.image_path,
                    "public_url": _to_public_url(row.image_path),
                    "match_status": row.match_status,
                    "confidence_score": row.confidence_score,
                }
            )

        db.commit()

    return jsonify({"message": "Test images processed", "results": results})


@test_images_bp.get("/results")
def get_results():
    """
    Get Results API:
    - returns all test image results and all video frame results
    This endpoint is shared by frontend `Results` component.
    """
    SessionLocal = current_app.session_local  # type: ignore[attr-defined]

    with SessionLocal() as db:
        images = db.execute(select(TestImage).order_by(TestImage.id.desc())).scalars().all()

        # Import here to avoid circular imports.
        from models import TestVideo, VideoMatch  # noqa: WPS433

        videos = db.execute(select(TestVideo).order_by(TestVideo.id.desc())).scalars().all()
        video_matches = db.execute(select(VideoMatch).order_by(VideoMatch.id.desc())).scalars().all()

    return jsonify(
        {
            "test_images": [
                {
                    "id": i.id,
                    "image_path": i.image_path,
                    "public_url": _to_public_url(i.image_path),
                    "match_status": i.match_status,
                    "confidence_score": i.confidence_score,
                    "created_at": i.created_at.isoformat(),
                }
                for i in images
            ],
            "test_videos": [
                {
                    "id": v.id,
                    "video_path": v.video_path,
                    "public_url": _to_public_url(v.video_path),
                    "created_at": v.created_at.isoformat(),
                }
                for v in videos
            ],
            "video_matches": [
                {
                    "id": m.id,
                    "video_id": m.video_id,
                    "frame_path": m.frame_path,
                    "public_url": _to_public_url(m.frame_path),
                    "match_status": m.match_status,
                    "confidence_score": m.confidence_score,
                }
                for m in video_matches
            ],
        }
    )


