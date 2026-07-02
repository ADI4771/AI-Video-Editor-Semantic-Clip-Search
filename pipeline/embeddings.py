import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Dict
from models.clip_model_loader import get_image_embedding
from pipeline.object_detection import detect_objects


def compute_frame_embeddings(
    frames,
    embeddings_dir,
    cache_file: str = "embeddings_cache.npz",
):
    from pathlib import Path
    import numpy as np
    from PIL import Image
    from models.clip_model_loader import get_image_embedding

    embeddings_dir = Path(embeddings_dir)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    cache_path = embeddings_dir / cache_file

    # Load cache
    if cache_path.exists():
        print(" Loading embeddings from cache...")
        data = np.load(cache_path, allow_pickle=True)
        return {
            "embeddings": data["embeddings"],
            "timestamps": data["timestamps"].tolist(),
            "paths": data["paths"].tolist(),
            "objects": data["objects"].tolist(),
        }

    print(f" Computing embeddings for {len(frames)} frames...")

    all_embeddings = []
    timestamps = []
    paths = []
    objects_list = []

    for i, (frame_path, timestamp) in enumerate(frames):
        try:
            pil_img = Image.open(frame_path).convert("RGB")

            # CLIP embedding
            embedding = get_image_embedding(pil_img).numpy().squeeze()

            # YOLO detection
            objects = detect_objects(frame_path)

            all_embeddings.append(embedding)
            timestamps.append(timestamp)
            paths.append(frame_path)
            objects_list.append(objects)

            if (i + 1) % 50 == 0:
                print(f"Processed {i+1}/{len(frames)} frames...")

        except Exception as e:
            print(f" Skipping frame {frame_path}: {e}")

    embeddings_array = np.array(all_embeddings, dtype=np.float32)

    # Save cache
    np.savez(
        cache_path,
        embeddings=embeddings_array,
        timestamps=np.array(timestamps),
        paths=np.array(paths),
        objects=np.array(objects_list, dtype=object),
    )

    print(f" Saved embeddings + objects → {cache_path}")

    return {
        "embeddings": embeddings_array,
        "timestamps": timestamps,
        "paths": paths,
        "objects": objects_list,
    }