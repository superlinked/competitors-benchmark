import argparse
import json
import logging
import pickle
from typing import Dict, List, Union

from config import (AZURE_FULL_TEXT_RESULTS_PATH, AZURE_GT_PARAMS_RESULTS_PATH,
                    AZURE_NLQ_RESULTS_PATH, GCS_BUCKET_NAME,
                    GCS_RANKED_RESULTS_BLOB, GCS_SL_FULL_TEXT_RESULTS_BLOB,
                    GCS_SL_GT_PARAMS_RESULTS_BLOB, GCS_SL_NLQ_RESULTS_BLOB,
                    HYBRID_SEARCH_RESULTS_PATH,
                    VERTEX_AI_FULL_TEXT_RESULTS_PATH,
                    VERTEX_AI_GT_PARAMS_RESULTS_PATH,
                    VERTEX_AI_NLQ_RESULTS_PATH)
from eval import evaluate_rankings
from google.cloud import storage

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument(
        "--system",
        type=str,
        choices=[
            "sl-nlq",
            "sl-gt-params",
            "sl-full-text",
            "vertex-ai-nlq",
            "vertex-ai-gt-params",
            "vertex-ai-full-text",
            "azure-nlq",
            "azure-gt-params",
            "azure-full-text",
            "hybrid-search",
        ],
        help="System to evaluate",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="@k",
    )
    args = parser.parse_args()
    return args


def load_results_from_gcs(blob_path: str) -> Union[List[Dict], Dict]:
    """Load results from GCS.

    Args:
        blob_path: Path to the blob.

    Returns:
        Union[List[Dict], Dict]: Results.
    """
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(blob_path)
    if blob_path.endswith(".json"):
        return json.loads(blob.download_as_text(encoding="utf-8"))
    elif blob_path.endswith(".pkl"):
        return pickle.loads(blob.download_as_bytes())


def load_system_results(system: str) -> Union[List[Dict], Dict]:
    """Load system results from local or GCS.

    Args:
        system: System to load results for.

    Returns:
        Union[List[Dict], Dict]: Results.
    """
    local_paths = {
        "vertex-ai-nlq": VERTEX_AI_NLQ_RESULTS_PATH,
        "vertex-ai-gt-params": VERTEX_AI_GT_PARAMS_RESULTS_PATH,
        "vertex-ai-full-text": VERTEX_AI_FULL_TEXT_RESULTS_PATH,
        "azure-nlq": AZURE_NLQ_RESULTS_PATH,
        "azure-gt-params": AZURE_GT_PARAMS_RESULTS_PATH,
        "azure-full-text": AZURE_FULL_TEXT_RESULTS_PATH,
        "hybrid-search": HYBRID_SEARCH_RESULTS_PATH,
    }
    gcs_blobs = {
        "sl-nlq": GCS_SL_NLQ_RESULTS_BLOB,
        "sl-gt-params": GCS_SL_GT_PARAMS_RESULTS_BLOB,
        "sl-full-text": GCS_SL_FULL_TEXT_RESULTS_BLOB,
    }
    if system in local_paths:
        with open(local_paths[system], "r") as f:
            return json.load(f)
    elif system in gcs_blobs:
        return load_results_from_gcs(gcs_blobs[system])
    else:
        raise ValueError(f"Unknown system: {system}")


def load_ground_truth() -> Union[List[Dict], Dict]:
    """Load ground truth from GCS.

    Returns:
        Union[List[Dict], Dict]: Ground truth.
    """
    return load_results_from_gcs(GCS_RANKED_RESULTS_BLOB)


def format_rankings(
    rankings: Union[List[Dict], Dict], ground_truth: bool = False
) -> list:
    """Format rankings.

    Args:
        rankings: Rankings.
        ground_truth: Whether the rankings are ground truth.

    Returns:
        list: Formatted rankings.
    """
    formatted_rankings = []

    if isinstance(rankings, list):
        return rankings
    elif isinstance(rankings, dict):
        for query_id, product_ids in rankings.items():
            for rank, product_id in enumerate(product_ids, 1):
                if ground_truth:
                    formatted_rankings.append((query_id, str(product_id), rank))
                else:
                    formatted_rankings.append(
                        (query_id, str(product_id), rank, query_id.split("-")[0])
                    )

    return formatted_rankings


def main() -> None:
    args = parse_arguments()
    ground_truth = load_ground_truth()
    system_results = load_system_results(args.system)

    formatted_rankings = format_rankings(system_results)
    formatted_ground_truth = format_rankings(ground_truth, ground_truth=True)

    results, by_query_results = evaluate_rankings(
        formatted_rankings, formatted_ground_truth, args.k
    )

    logger.info(results)


if __name__ == "__main__":
    main()
