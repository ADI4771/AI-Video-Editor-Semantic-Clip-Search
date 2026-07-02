import subprocess
import os
from pathlib import Path
from typing import List, Dict
from utils.video_utils import get_video_duration, clamp_time


def generate_clip(
    video_path: str,
    start: float,
    end: float,
    output_path: str,
) -> str:
    """
    Cut a single clip from video using ffmpeg (fast stream copy).
    """
    duration = end - start
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", video_path,
        "-t", str(duration),
        "-c", "copy",           # Stream copy = fast, no re-encode
        "-avoid_negative_ts", "1",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg clip error:\n{result.stderr}")
    return output_path


def generate_clips_from_segments(
    video_path: str,
    segments: List[Dict],
    output_dir: str,
) -> List[str]:

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_duration = get_video_duration(video_path)
    clip_paths = []

    for i, seg in enumerate(segments):
        start = clamp_time(seg["start"], video_duration)
        end = clamp_time(seg["end"], video_duration)

        if end - start < 0.5:
            print(f"   Skipping segment {i+1}: too short after clamping")
            continue

        out_path = str(output_dir / f"clip_{i+1:02d}_score{seg['peak_score']:.2f}.mp4")
        print(f"   Clip {i+1}: {start:.1f}s → {end:.1f}s (score: {seg['peak_score']:.3f})")
        generate_clip(video_path, start, end, out_path)
        clip_paths.append(out_path)

    return clip_paths


def merge_clips(clip_paths: List[str], output_path: str) -> str:

    if not clip_paths:
        raise ValueError("No clips to merge.")

    if len(clip_paths) == 1:
        import shutil
        shutil.copy(clip_paths[0], output_path)
        return output_path

    # Write concat list file
    list_file = str(Path(output_path).parent / "concat_list.txt")
    with open(list_file, "w") as f:
        for cp in clip_paths:
            f.write(f"file '{os.path.abspath(cp)}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_file,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-movflags", "+faststart",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg merge error:\n{result.stderr}")

    os.remove(list_file)
    print(f" Merged {len(clip_paths)} clips → {output_path}")
    return output_path
