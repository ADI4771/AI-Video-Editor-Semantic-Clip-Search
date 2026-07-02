import numpy as np
from typing import List, Dict, Tuple
from models.clip_model_loader import get_text_embedding
from utils.similarity import cosine_similarity_batch

COMMON_OBJECTS = {
    "person", "car", "dog", "cat", "bicycle", "truck",
    "bus", "bird", "chair", "bottle", "motorcycle",
    "horse", "boat", "airplane"
}

def extract_objects_from_query(query: str):
    words = query.lower().split()
    return [w for w in words if w in COMMON_OBJECTS]

def search_frames(
    query: str,
    frame_data,
    top_k: int = 5,
    similarity_threshold: float = 0.20,
):
    from models.clip_model_loader import get_text_embedding
    from utils.similarity import cosine_similarity_batch
    import numpy as np

    # Extract objects from query
    query_objects = extract_objects_from_query(query)

    # CLIP embedding
    text_embedding = get_text_embedding(query).numpy().squeeze()
    scores = cosine_similarity_batch(text_embedding, frame_data["embeddings"])

    top_indices = np.argsort(scores)[::-1]
    results = []

    for idx in top_indices:
        score = float(scores[idx])

        if score < similarity_threshold:
            break

        #  OBJECT FILTER
        if query_objects:
            frame_objects = frame_data.get("objects", [])
            if frame_objects:
                objs = frame_objects[idx]
                if not any(obj in objs for obj in query_objects):
                    continue

        if len(results) >= top_k:
            break

        results.append({
            "timestamp": frame_data["timestamps"][idx],
            "frame_path": frame_data["paths"][idx],
            "score": score,
        })

    return results

def group_into_segments(
    results: List[Dict],
    gap_threshold: float = 8.0,
    clip_padding: float = 5.0,
) -> List[Dict]:
 
    if not results:
        return []

    # Sort by timestamp for grouping
    sorted_results = sorted(results, key=lambda x: x["timestamp"])

    segments = []
    current_group = [sorted_results[0]]

    for frame in sorted_results[1:]:
        prev_ts = current_group[-1]["timestamp"]
        curr_ts = frame["timestamp"]

        if curr_ts - prev_ts <= gap_threshold:
            current_group.append(frame)
        else:
            segments.append(_build_segment(current_group, clip_padding))
            current_group = [frame]

    # Don't forget last group
    segments.append(_build_segment(current_group, clip_padding))

    # Sort by peak score so best segments come first
    segments.sort(key=lambda s: s["peak_score"], reverse=True)
    return segments


def _build_segment(group: List[Dict], padding: float) -> Dict:
    timestamps = [f["timestamp"] for f in group]
    scores = [f["score"] for f in group]
    best_idx = int(np.argmax(scores))

    return {
        "start": max(0.0, min(timestamps) - padding),
        "end": max(timestamps) + padding,
        "peak_score": max(scores),
        "frame_path": group[best_idx]["frame_path"],  # Thumbnail = best frame
        "matched_timestamps": timestamps,
    }
