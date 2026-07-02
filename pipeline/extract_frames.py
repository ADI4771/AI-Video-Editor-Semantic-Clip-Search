import cv2
import os
from pathlib import Path
from PIL import Image
from typing import List, Tuple


def extract_frames(
    video_path: str,
    output_dir: str,
    fps: float = 1.0,
) -> List[Tuple[str, float]]:

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0

    frame_interval = int(video_fps / fps)  # Extract every N-th frame
    frame_interval = max(1, frame_interval)

    extracted = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / video_fps
            frame_filename = f"frame_{frame_idx:07d}_{timestamp:.3f}.jpg"
            frame_path = output_dir / frame_filename

            # Convert BGR → RGB and save
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            pil_img.save(frame_path, "JPEG", quality=85)

            extracted.append((str(frame_path), timestamp))

        frame_idx += 1

    cap.release()
    print(f" Extracted {len(extracted)} frames from {duration:.1f}s video")
    return extracted
