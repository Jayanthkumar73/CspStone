"""Face recognition evaluation metrics."""

import numpy as np
from sklearn.metrics import roc_curve


def cosine_similarity(a, b):
    """Cosine similarity between two vectors or two sets of vectors."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.ndim == 1:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    # Matrix
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a_norm @ b_norm.T


def rank_n_accuracy(probe_embs, gallery_embs, probe_ids, gallery_ids, n=1):
    """Compute Rank-N identification accuracy."""
    sims = cosine_similarity(probe_embs, gallery_embs)  # [P, G]
    correct = 0
    for i in range(len(probe_ids)):
        top_n_idx = np.argsort(sims[i])[::-1][:n]
        top_n_ids = [gallery_ids[j] for j in top_n_idx]
        if probe_ids[i] in top_n_ids:
            correct += 1
    return correct / len(probe_ids)


def tar_at_far(scores, labels, far_target=0.01):
    """Compute TAR (True Accept Rate) at a given FAR (False Accept Rate)."""
    fpr, tpr, _ = roc_curve(labels, scores)
    # Find closest FAR to target
    idx = np.argmin(np.abs(fpr - far_target))
    return tpr[idx]


def compute_all_metrics(probe_embs, gallery_embs, probe_ids, gallery_ids):
    """Compute all standard FR metrics."""
    results = {}
    for n in [1, 5, 10]:
        results[f"rank{n}"] = rank_n_accuracy(
            probe_embs, gallery_embs, probe_ids, gallery_ids, n=n
        )

    # Verification metrics (all pairs)
    sims = cosine_similarity(probe_embs, gallery_embs)
    scores, labels = [], []
    for i in range(len(probe_ids)):
        for j in range(len(gallery_ids)):
            scores.append(sims[i, j])
            labels.append(1 if probe_ids[i] == gallery_ids[j] else 0)

    for far in [0.01, 0.001]:
        key = f"tar_far{str(far).replace('.', '')}"
        results[key] = tar_at_far(scores, labels, far)

    return results
