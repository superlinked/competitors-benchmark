#!/usr/bin/env python3
"""
Azure Cognitive Search Ranking

This module performs semantic search using Azure Cognitive Search with
configurable filtering and parameter-based constraints.
"""

import argparse
import json
import logging
from typing import Any, Dict, List, Optional

from config import (AZURE_SEARCH_API_KEY, AZURE_SEARCH_ENDPOINT,
                    AZURE_SEARCH_INDEX_NAME, FULL_TEXT_OUTPUT_RESULTS,
                    GCS_BUCKET_NAME, GCS_QUERIES_BLOB,
                    GT_PARAMS_OUTPUT_RESULTS, NLQ_OUTPUT_RESULTS,
                    NLQ_PARAMS_QUERIES, SEMANTIC_CONFIGURATION_NAME)
from google.cloud import storage

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description="Azure Cognitive Search Ranking")
    parser.add_argument(
        "--params",
        type=str,
        default="none",
        choices=["nlq", "gt", "none"],
        help="Which params to use to filter the results",
    )
    args = parser.parse_args()
    return args


def load_queries_from_file() -> List[Dict[str, Any]]:
    """
    Load queries from JSON file.

    Returns:
        List of query dictionaries
    """
    with open(NLQ_PARAMS_QUERIES, "r", encoding="utf-8") as f:
        queries = json.load(f)
    return queries


def load_queries_from_bucket() -> List[Dict[str, Any]]:
    """
    Load queries from Google Cloud Storage bucket.
    Returns:
        List of query dictionaries

    Raises:
        Exception: If loading from bucket fails
    """
    try:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(GCS_QUERIES_BLOB)
        text = blob.download_as_text(encoding="utf-8")
        queries = json.loads(text)

        logger.info(f"Loaded {len(queries)} queries from bucket")
        return queries

    except Exception as e:
        logger.error(f"Failed to load queries from bucket: {e}")
        raise


def build_filter(query_params: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    Build Azure Search filter expression from query parameters.

    Args:
        query_params: Dictionary containing query parameters

    Returns:
        Filter expression string or None if no filters
    """
    if not query_params:
        return None

    filters: List[str] = []

    # Numeric constraints
    price_max = query_params.get("price_max")
    if isinstance(price_max, (int, float)) and price_max > 0:
        filters.append(f"price le {float(price_max)}")

    rating_min = query_params.get("rating_min")
    if isinstance(rating_min, (int, float)) and rating_min > 0:
        filters.append(f"average_rating ge {float(rating_min)}")

    rating_count_min = query_params.get("rating_count_min")
    if isinstance(rating_count_min, (int, float)) and rating_count_min > 0:
        filters.append(f"rating_count ge {int(rating_count_min)}")

    # Collection includes
    for field in ["color", "material", "style"]:
        raw_val = query_params.get(field)
        values: List[str] = []

        if isinstance(raw_val, list):
            values = [str(v).strip() for v in raw_val if v]
        elif isinstance(raw_val, str) and raw_val.strip():
            values = [raw_val.strip()]

        if values:
            escaped = [v.replace("'", "''") for v in values]
            inner = " or ".join([f"t eq '{v}'" for v in escaped])
            filters.append(f"{field}/any(t: {inner})")

    # Collection negations
    for neg_key, field in [
        ("color_negated", "color"),
        ("material_negated", "material"),
        ("style_negated", "style"),
    ]:
        raw_val = query_params.get(neg_key)
        values: List[str] = []

        if isinstance(raw_val, list):
            values = [str(v).strip() for v in raw_val if v]
        elif isinstance(raw_val, str) and raw_val.strip():
            values = [raw_val.strip()]

        if values:
            escaped = [v.replace("'", "''") for v in values]
            inner = " or ".join([f"t eq '{v}'" for v in escaped])
            filters.append(f"not {field}/any(t: {inner})")

    if not filters:
        return None

    return " and ".join(filters)


def create_search_client() -> SearchClient:
    """
    Create Azure Search client.

    Returns:
        Configured SearchClient instance

    Raises:
        ValueError: If required credentials are missing
    """
    if not AZURE_SEARCH_API_KEY:
        raise ValueError("AZURE_SEARCH_API_KEY environment variable is required")

    credential = AzureKeyCredential(AZURE_SEARCH_API_KEY)
    search_client = SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX_NAME,
        credential=credential,
    )

    logger.info(f"Created search client for index: {AZURE_SEARCH_INDEX_NAME}")
    return search_client


def search_products(
    search_client: SearchClient,
    query_text: str,
    query_params: Optional[Dict[str, Any]] = None,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Search products using Azure Cognitive Search.

    Args:
        search_client: Azure Search client
        query_text: Search query text
        query_params: Optional query parameters for filtering
        top_k: Number of results to return

    Returns:
        List of search results
    """
    filter_expr = build_filter(query_params)

    if filter_expr:
        logger.debug(f"Using filter: {filter_expr}")

    try:
        results = search_client.search(
            search_text=query_text,
            query_type="semantic",
            semantic_configuration_name=SEMANTIC_CONFIGURATION_NAME,
            search_fields=[
                "product_name",
                "product_description",
                "product_class",
                "material",
                "style",
                "color",
            ],
            filter=filter_expr,
            select=[
                "product_id",
                "product_name",
                "product_description",
                "product_class",
                "price",
                "average_rating",
                "rating_count",
                "material",
                "style",
                "color",
            ],
            top=top_k,
        )

        # Convert results to list of dictionaries
        result_list = []
        for result in results:
            if hasattr(result, "document"):
                result_list.append(result.document)
            else:
                result_list.append(result)

        logger.debug(f"Found {len(result_list)} results for query: {query_text}")
        return result_list

    except Exception as e:
        logger.error(f"Search failed for query '{query_text}': {e}")
        return []


def save_results(query_results: Dict[str, List[str]], output_path: str) -> None:
    """
    Save search results to output file.

    Args:
        query_results: Dictionary mapping query IDs to product ID lists

    Raises:
        Exception: If saving fails
    """
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(query_results, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved results for {len(query_results)} queries to {output_path}")

    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        raise


def load_queries_by_params_type(params_type: str) -> List[Dict[str, Any]]:
    """
    Load queries based on parameter type.

    Args:
        params_type: Type of parameters to use ("nlq", "gt", "none")

    Returns:
        List of query dictionaries
    """
    if params_type == "nlq":
        queries = load_queries_from_file()
        # Ensure all queries have query_params
        for q in queries:
            if "query_params" not in q or q["query_params"] is None:
                q["query_params"] = {}
        return queries

    elif params_type == "none":
        queries = load_queries_from_bucket()
        # Empty query_params
        for q in queries:
            q["query_params"] = {}
        return queries

    elif params_type == "gt":
        queries = load_queries_from_bucket()
        return queries

    else:
        raise ValueError(f"Unknown parameter type: {params_type}")


def process_queries(
    search_client: SearchClient, queries: List[Dict[str, Any]]
) -> Dict[str, List[str]]:
    """
    Process all queries and return results.

    Args:
        search_client: Azure Search client
        queries: List of query dictionaries

    Returns:
        Dictionary mapping query IDs to product ID lists
    """
    query_results: Dict[str, List[str]] = {item["query_id"]: [] for item in queries}

    for item in queries:
        query_id = item["query_id"]
        query_text = item["query_text"]
        query_params = item.get("query_params", {})

        logger.info(f"Processing query: {query_id}, {query_text}")

        try:
            results = search_products(search_client, query_text, query_params, top_k=10)

            # Extract product IDs from results
            product_ids = [
                result.get("product_id")
                for result in results
                if result.get("product_id")
            ]
            query_results[query_id] = product_ids

            logger.info(f"Found {len(product_ids)} products for query {query_id}")

        except Exception as e:
            logger.error(f"Failed to process query {query_id}: {e}")
            query_results[query_id] = []

    return query_results


def main():
    try:
        # Parse arguments
        args = parse_arguments()

        if args.params == "gt":
            output_path = GT_PARAMS_OUTPUT_RESULTS
        elif args.params == "nlq":
            output_path = NLQ_OUTPUT_RESULTS
        elif args.params == "none":
            output_path = FULL_TEXT_OUTPUT_RESULTS

        # Load queries based on parameter type
        queries = load_queries_by_params_type(args.params)

        if not queries:
            logger.warning("No queries found to process")
            return

        # Create search client
        search_client = create_search_client()

        # Process queries
        query_results = process_queries(search_client, queries)

        # Save results
        save_results(query_results, output_path=output_path)

        logger.info(f"Successfully processed {len(queries)} queries")

    except Exception as e:
        logger.error(f"Ranking failed: {e}")
        raise


if __name__ == "__main__":
    main()
