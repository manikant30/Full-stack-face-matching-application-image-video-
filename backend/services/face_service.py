from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

import face_recognition
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    match_status: str  # MATCH / NO_MATCH / NO_FACE / NO_TRUTH
    confidence_score: float
    face_distance: Optional[float] = None


def extract_face_embedding(image_path: str) -> Optional[list[float]]:
    """
    Returns the first face embedding found in the image, or None if no face.
    """
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    if not encodings:
        return None
    return encodings[0].tolist()


def embedding_to_json(embedding: list[float]) -> str:
    return json.dumps(embedding)


def compare_embeddings(
    truth_embedding: Optional[list[float]],
    test_embedding: Optional[list[float]],
    threshold: float,
) -> MatchResult:
    """
    Uses face distance with a threshold of 0.6.
    Converts distance to confidence percentage:
      confidence = max(0, 1 - distance/threshold) * 100
    """
    if truth_embedding is None:
        return MatchResult(match_status="NO_TRUTH", confidence_score=0.0)
    if test_embedding is None:
        return MatchResult(match_status="NO_FACE", confidence_score=0.0)

    # face_recognition's face_distance subtracts arrays; ensure numpy arrays here.
    truth_np = np.asarray(truth_embedding, dtype=np.float64)
    test_np = np.asarray(test_embedding, dtype=np.float64)
    distance = face_recognition.face_distance([truth_np], test_np)[0]
    is_match = distance <= threshold
    confidence = max(0.0, (1.0 - (float(distance) / threshold)) * 100.0)
    if confidence > 100.0:
        confidence = 100.0

    return MatchResult(
        match_status="MATCH" if is_match else "NO_MATCH",
        confidence_score=round(confidence, 2),
        face_distance=round(float(distance), 4),
    )


