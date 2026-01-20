from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request
from sqlalchemy import select

from models import TestVideo, TruthImage, VideoMatch
from services.face_service import compare_embeddings, extract_face_embedding
from services.video_service import extract_frames_one_per_second

logger = logging.getLogger(__name__)

test_videos_bp = Blueprint("test_videos", __name__)

def _to_public_url(path_value: str) -> str:
    norm = path_value.replace("\\", "/")
    if "/uploads/" in norm:
        rel = norm.split("/uploads/", 1)[1]
        return f"/uploads/{rel}"
    return f"/uploads/{norm.lstrip('/')}"


def _get_latest_truth_embedding(db) -> list[float] | None:
    truth = db.execute(select(TruthImage).order_by(TruthImage.id.desc())).scalars().first()
    if not truth:
        return None
    return truth.embedding_as_list()


@test_videos_bp.post("/test-videos")
def upload_test_videos():
    """
    Upload multiple test videos:
    - save each video to uploads/videos/
    - extract frames (1 frame/sec) into uploads/frames/<video_id>/
    - match each frame vs truth image
    - store per-frame results in video_matches
    """
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files provided"}), 400

    videos_dir = Path(current_app.config["UPLOAD_FOLDER"]) / "videos"
    frames_dir = Path(current_app.config["UPLOAD_FOLDER"]) / "frames"
    videos_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    SessionLocal = current_app.session_local  # type: ignore[attr-defined]
    threshold = float(current_app.config["FACE_DISTANCE_THRESHOLD"])

    processed = []
    with SessionLocal() as db:
        truth_embedding = _get_latest_truth_embedding(db)

        for file in files:
            if not file or file.filename == "":
                continue

            ext = os.path.splitext(file.filename)[1].lower() or ".mp4"
            filename = f"vid_{uuid.uuid4().hex}{ext}"
            abs_video_path = videos_dir / filename
            file.save(str(abs_video_path))
            rel_video_path = str(Path("videos") / filename)

            video_row = TestVideo(video_path=rel_video_path)
            db.add(video_row)
            db.flush()  # get id

            out_frames_dir = frames_dir / f"video_{video_row.id}"
            out_frames_dir.mkdir(parents=True, exist_ok=True)

            frame_results = []
            for fr in extract_frames_one_per_second(str(abs_video_path), str(out_frames_dir)):
                # Store frame paths relative to uploads/ for easy serving
                rel_frame_path = str(Path("frames") / f"video_{video_row.id}" / Path(fr.frame_path).name)
                test_embedding = extract_face_embedding(fr.frame_path)
                match = compare_embeddings(truth_embedding, test_embedding, threshold=threshold)

                m = VideoMatch(
                    video_id=video_row.id,
                    frame_path=rel_frame_path,
                    match_status=match.match_status,
                    confidence_score=float(match.confidence_score),
                )
                db.add(m)
                db.flush()
                frame_results.append(
                    {
                        "id": m.id,
                        "frame_path": m.frame_path,
                        "public_url": _to_public_url(m.frame_path),
                        "match_status": m.match_status,
                        "confidence_score": m.confidence_score,
                    }
                )

            processed.append(
                {
                    "video": {
                        "id": video_row.id,
                        "video_path": video_row.video_path,
                        "public_url": _to_public_url(video_row.video_path),
                    },
                    "frames": frame_results,
                }
            )

        db.commit()

    return jsonify({"message": "Test videos processed", "results": processed})


