"""Retrieve geographically referenced training images using text embeddings."""

import torch
import torch.nn.functional as F


@torch.inference_mode()
def retrieve_top_k(
    text_embeddings,
    reference_embeddings,
    records,
    top_k=5,
    device="cuda",
    chunk_size=8192,
):
    """Retrieve the top-K panoramas using average similarity across views."""
    if len(records) != len(reference_embeddings):
        raise ValueError("Reference records and embeddings differ in length")

    if not records or top_k < 1 or chunk_size < 1:
        raise ValueError("Invalid reference index or retrieval settings")

    # Normalize each description embedding.
    queries = F.normalize(
        text_embeddings.float().to(device),
        dim=-1,
    )

    best_scores = None
    best_indices = None

    for start in range(0, len(reference_embeddings), chunk_size):
        chunk = F.normalize(
            reference_embeddings[start:start + chunk_size]
            .float()
            .to(device),
            dim=-1,
        )

        # Each description is compared against every reference panorama.
        similarities = queries @ chunk.T

        # Average similarity across all panorama view descriptions.
        scores = similarities.mean(dim=0)

        chunk_scores, indices = torch.topk(
            scores,
            k=min(top_k, len(chunk)),
        )

        indices += start

        if best_scores is None:
            best_scores = chunk_scores
            best_indices = indices
        else:
            all_scores = torch.cat([best_scores, chunk_scores])
            all_indices = torch.cat([best_indices, indices])

            best_scores, positions = torch.topk(
                all_scores,
                k=min(top_k, len(all_scores)),
            )

            best_indices = all_indices[positions]

    return [
        {
            "rank": rank,
            **records[index],
            "similarity": score,
        }
        for rank, (score, index) in enumerate(
            zip(best_scores.tolist(), best_indices.tolist()),
            start=1,
        )
    ]