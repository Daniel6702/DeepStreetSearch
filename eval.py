"""Evaluate text-to-image geographic retrieval on an unseen panorama."""

import argparse
from pathlib import Path

import torch

from config import SIGLIP_MODEL, DATASET_PATH
from evaluation.inference import load_model, encode_text
from evaluation.reference import build_reference_index, load_reference_index
from evaluation.retrieval import retrieve_top_k
from evaluation.metrics import evaluate_matches
from evaluation.results import save_results
from evaluation.description import PanoramaDescriber


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", default="checkpoints/deepstreetsearch.pt")
    parser.add_argument("--grid", default="outputs/grids/world.aggrid")
    parser.add_argument("--model", default=SIGLIP_MODEL)
    parser.add_argument("--reference-dataset", default=f"{DATASET_PATH}/world")

    parser.add_argument("--image", required=True)
    parser.add_argument("--latitude", type=float, required=True)
    parser.add_argument("--longitude", type=float, required=True)

    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="results/text_retrieval")
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--max-reference-images", type=int, default=5000,
                    help="Limit reference index size (0 = use all)")

    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if not -90 <= args.latitude <= 90:
        parser.error("--latitude must be between -90 and 90")
    if not -180 <= args.longitude <= 180:
        parser.error("--longitude must be between -180 and 180")

    return args


def main():
    args = parse_args()
    device = torch.device(args.device)

    output_dir = Path(args.output)
    index_path = output_dir / "reference_index.pt"

    # --------------------------------------------------
    # 1. Generate descriptions using Qwen3-VL
    # --------------------------------------------------

    print("\nGenerating panorama descriptions...")

    describer = PanoramaDescriber()
    descriptions = describer.describe(args.image)

    print("\nGenerated descriptions:\n")

    for i, description in enumerate(descriptions, start=1):
        print(f"View {i}: {description}\n")

    # Release Qwen before loading SigLIP.
    # vLLM may reserve most of the available GPU memory.
    del describer

    import gc
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --------------------------------------------------
    # 2. Load trained SigLIP model
    # --------------------------------------------------

    print("Loading trained GeoSearchModel...")

    model, processor = load_model(
        args.checkpoint,
        args.grid,
        args.model,
        device,
    )

    # --------------------------------------------------
    # 3. Load or build reference database
    # --------------------------------------------------

    if index_path.exists() and not args.rebuild_index:
        print("Loading reference index...")
        index = load_reference_index(index_path)
    else:
        print("Building reference index...")

        index = build_reference_index(
            model,
            processor,
            args.reference_dataset,
            device,
            index_path,
            max_images=args.max_reference_images or None,
        )

    # --------------------------------------------------
    # 4. Encode generated descriptions
    # --------------------------------------------------

    print("\nEncoding descriptions...")

    text_embeddings = encode_text(
        model,
        processor,
        descriptions,
        device,
    )

    # --------------------------------------------------
    # 5. Retrieve similar training images
    # --------------------------------------------------

    print("Searching reference images...")

    matches = retrieve_top_k(
        text_embeddings,
        index["embeddings"],
        index["records"],
        top_k=args.top_k,
        device=device,
    )

    # --------------------------------------------------
    # 6. Calculate geographic errors
    # --------------------------------------------------

    matches = evaluate_matches(
        matches,
        args.latitude,
        args.longitude,
    )

    # --------------------------------------------------
    # 7. Save results
    # --------------------------------------------------

    save_results(
        matches,
        args.image,
        descriptions,
        args.latitude,
        args.longitude,
        output_dir,
    )

    # --------------------------------------------------
    # 8. Display results
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("TEXT-TO-IMAGE GEOLOCATION EVALUATION")
    print("=" * 60)

    print(f"\nUnknown image: {args.image}")

    print(
        f"Ground truth: "
        f"{args.latitude:.6f}, {args.longitude:.6f}"
    )

    print("\nTop retrieved locations:\n")

    for match in matches:
        print(
            f"#{match['rank']} | "
            f"{match['latitude']:.5f}, {match['longitude']:.5f} | "
            f"Similarity: {match['similarity']:.4f} | "
            f"Error: {match['distance_km']:.2f} km"
        )

    best = matches[0]

    print("\n" + "=" * 60)
    print(
        f"FINAL GEOGRAPHIC ERROR: "
        f"{best['distance_km']:.2f} km"
    )
    print("=" * 60)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()