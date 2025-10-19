import json
import logging
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from config import (CLEANED_PRODUCTS_CSV, CLEANED_PRODUCTS_JSONL,
                    EMBEDDINGS_FILE, GCS_BUCKET_FULL_PATH, GCS_BUCKET_NAME,
                    INPUT_DIR, OUTPUT_DIR, QUERIES_JSON, RAW_PRODUCT_CSV,
                    SEARCH_RESULTS_JSON, SYNTHETIC_PRODUCTS_JSON,
                    SYNTHETIC_QUERIES_JSON, SYNTHETIC_RESULTS_JSON,
                    UNIQUE_CATEGORIES_JSON)
from google.cloud import storage

logger = logging.getLogger(__name__)


def load_synthetic_products() -> pd.DataFrame:
    """Load synthetic products from JSON file.

    Returns:
        pd.DataFrame: Synthetic products data.
    """
    logger.info("Loading synthetic products...")
    input_file = os.path.join(INPUT_DIR, SYNTHETIC_PRODUCTS_JSON)
    synthetic_products = pd.read_json(input_file)
    logger.info(f"Loaded {len(synthetic_products)} synthetic products")
    return synthetic_products


def load_synth_queries_results() -> Tuple[List[Dict[str, Any]], Dict[str, List[int]]]:
    """Load synthetic queries and results.

    Returns:
        Tuple[List[Dict[str, Any]], Dict[str, List[int]]]: Tuple of synthetic queries and results.
    """
    return load_queries(os.path.join(INPUT_DIR, SYNTHETIC_QUERIES_JSON)), load_results(
        os.path.join(INPUT_DIR, SYNTHETIC_RESULTS_JSON)
    )


def save_embeddings(all_embeddings: List[Dict[str, Any]], output_file: str = None):
    """Save embeddings to a JSONL file.

    Args:
        all_embeddings: List of embeddings.
        output_file: Output file.
    """
    if output_file is None:
        output_file = os.path.join(OUTPUT_DIR, EMBEDDINGS_FILE)
    with open(output_file, "w", encoding="utf-8") as f:
        for emb in all_embeddings:
            if isinstance(emb["dense_embedding"], np.ndarray):
                emb["dense_embedding"] = emb["dense_embedding"].tolist()
            f.write(json.dumps(emb) + "\n")


def save_product_data(products_df: pd.DataFrame):
    """Save product data to a CSV and JSON file.

    Args:
        products_df: Products DataFrame.
    """
    csv_output = os.path.join(OUTPUT_DIR, CLEANED_PRODUCTS_CSV)
    products_df.to_csv(csv_output, index=False)
    logger.info(f"Final data saved as CSV to {csv_output}")

    json_output = os.path.join(OUTPUT_DIR, CLEANED_PRODUCTS_JSONL)
    products_df.to_json(json_output, orient="records", lines=True)
    logger.info(f"Final data saved as JSON to {json_output}")


def load_results(results_file: str = None) -> Dict[str, List[int]]:
    """Load results from a JSON file.

    Returns:
        Dict[str, List[int]]: Mapping of query IDs to product ID lists.
    """
    if results_file is None:
        results_file = os.path.join(OUTPUT_DIR, SEARCH_RESULTS_JSON)
    logger.info(f"Loading results from {results_file}...")
    with open(os.path.join(OUTPUT_DIR, results_file), "r") as f:
        results = json.load(f)
    logger.info(f"Loaded {len(results)} results")
    return results


def load_product_data(products_file: str = None) -> pd.DataFrame:
    """Load product data.

    Args:
        products_file: Products file.

    Returns:
        pd.DataFrame: Products DataFrame.
    """
    if products_file is None:
        products_file = os.path.join(OUTPUT_DIR, CLEANED_PRODUCTS_JSONL)
    logger.info(f"Loading product data from {products_file}...")

    products_list = []
    with open(products_file, "r") as f:
        for line in f:
            products_list.append(json.loads(line.strip()))

    products_df = pd.DataFrame(products_list)
    logger.info(f"Loaded {len(products_df)} products")
    return products_df


def load_queries(queries_file: str = None) -> List[Dict[str, Any]]:
    """Load queries.

    Args:
        queries_file: Queries file.

    Returns:
        List[Dict[str, Any]]: List of query dictionaries.
    """
    if queries_file is None:
        queries_file = os.path.join(OUTPUT_DIR, QUERIES_JSON)
    logger.info(f"Loading queries from {queries_file}...")

    with open(os.path.join(OUTPUT_DIR, queries_file), "r") as f:
        queries = json.load(f)

    logger.info(f"Loaded {len(queries)} queries")
    return queries


def load_search_results(results_file: str = None) -> Dict[str, List[int]]:
    """Load search results.

    Args:
        results_file: Results file.

    Returns:
        Dict[str, List[int]]: Mapping of query IDs to product ID lists.
    """
    if results_file is None:
        results_file = os.path.join(OUTPUT_DIR, SEARCH_RESULTS_JSON)
    logger.info(f"Loading search results from {results_file}...")

    with open(os.path.join(OUTPUT_DIR, results_file), "r") as f:
        results = json.load(f)

    logger.info(f"Loaded search results for {len(results)} queries")
    return results


def load_raw_products() -> pd.DataFrame:
    """Load the original product data.

    Args:
        None

    Returns:
        pd.DataFrame: Raw product data from CSV file.
    """
    logger.info("Loading product data...")
    input_file = os.path.join(INPUT_DIR, RAW_PRODUCT_CSV)
    products = pd.read_csv(input_file, delimiter="\t")
    logger.info(f"Loaded {len(products)} products")
    return products


def load_unique_categories() -> Dict[str, List[str]]:
    """Load unique categories.

    Returns:
        Dict[str, List[str]]: Dictionary of unique categories.
    """
    return json.load(open(UNIQUE_CATEGORIES_JSON))


def upload_to_gcs(data, blob_name):
    """Upload data to GCS.

    Args:
        data: Data to upload (dict, list, or string).
        blob_name: Blob name.
    """
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(GCS_BUCKET_FULL_PATH + blob_name)

    # Convert data to JSON string if it's a dict or list
    if isinstance(data, (dict, list)):
        data_str = json.dumps(data, indent=2)
        content_type = "application/json"
    else:
        data_str = str(data)
        content_type = "text/plain"

    blob.upload_from_string(data_str, content_type=content_type)

    return None
