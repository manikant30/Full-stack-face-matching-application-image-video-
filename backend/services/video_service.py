from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Iterator, Optional

import cv2

logger = logging.getLogger(__name__)


@dataclass
class ExtractedFrame:
    frame_path: str
    frame_index: int
    timestamp_sec: int


def extract_frames_one_per_second(
    video_path: str,
    output_dir: str,
    max_seconds: Optional[int] = None,
) -> Iterator[ExtractedFrame]:
    """
    Extract 1 frame per second from a video and save as JPG.
    Keeps logic simple: seeks by frame index using FPS.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fps = float(fps) if fps > 0 else 25.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = int(total_frames / fps) if total_frames > 0 else 0

    seconds_to_process = duration_sec if duration_sec > 0 else 0
    if max_seconds is not None:
        seconds_to_process = min(seconds_to_process, int(max_seconds))

    # If duration is unknown, just iterate until read fails, sampling every ~fps frames.
    if seconds_to_process == 0:
        frame_step = int(round(fps))
        idx = 0
        sec = 0
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                break
            out_path = os.path.join(output_dir, f"frame_{sec:06d}.jpg")
            cv2.imwrite(out_path, frame)
            yield ExtractedFrame(frame_path=out_path, frame_index=idx, timestamp_sec=sec)
            idx += frame_step
            sec += 1
        cap.release()
        return

    for sec in range(seconds_to_process):
        frame_idx = int(sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break
        out_path = os.path.join(output_dir, f"frame_{sec:06d}.jpg")
        cv2.imwrite(out_path, frame)
        yield ExtractedFrame(frame_path=out_path, frame_index=frame_idx, timestamp_sec=sec)

    cap.release()


