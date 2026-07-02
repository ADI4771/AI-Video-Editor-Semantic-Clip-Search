import numpy as np


def cosine_similarity_batch(query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:

    query = query_embedding.squeeze()
    query = query / (np.linalg.norm(query) + 1e-8)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    normed = embeddings / norms

    return (normed @ query).astype(np.float32)
