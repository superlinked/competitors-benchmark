#!/usr/bin/env python3
"""
Hybrid Search Product Reranking

This module reranks search results using Google Cloud's Discovery Engine
ranking API. It takes product IDs, enriches them with product data,
and applies sophisticated ranking algorithms.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from config import (GCS_BUCKET_NAME, GCS_PRODUCTS_BLOB, GCS_QUERIES_BLOB,
                    LOCATION, PROJECT_ID, RANKED_RESULTS_FILE, RANKING_MODEL,
                    RANKING_TOP_K, RESULTS_FILE)
from google.cloud import discoveryengine_v1 as discoveryengine
from google.cloud import storage

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def load_queries_from_bucket() -> List[Dict[str, Any]]:
    """
    Load queries from Google Cloud Storage bucket.

    Args:
        bucket_name: Name of the GCS bucket
        blob_path: Path to the queries file in the bucket

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


def load_results_from_file(file_path: str) -> Dict[str, List[str]]:
    """
    Load search results from JSON file.

    Args:
        file_path: Path to the results JSON file

    Returns:
        Dictionary mapping query IDs to product ID lists

    Raises:
        Exception: If loading fails
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        logger.info(f"Loaded results for {len(results)} queries from {file_path}")
        return results

    except Exception as e:
        logger.error(f"Failed to load results from file: {e}")
        raise


def load_products_from_bucket() -> Dict[str, Dict[str, Any]]:
    """
    Load product data from Google Cloud Storage bucket.

    Returns:
        Dictionary mapping product IDs to product data

    Raises:
        Exception: If loading fails
    """
    try:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(GCS_PRODUCTS_BLOB)
        text = blob.download_as_text(encoding="utf-8")

        products = {}
        for line_num, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                product = json.loads(line)
                product_id = product.get("product_id")
                if product_id:
                    products[str(product_id)] = product
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON on line {line_num}: {e}")
                continue

        logger.info(f"Loaded {len(products)} products from bucket")
        return products

    except Exception as e:
        logger.error(f"Failed to load products from bucket: {e}")
        raise


def enrich_products_with_data(
    product_ids: List[str], products_dict: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Enrich product IDs with detailed product data for ranking.

    Args:
        product_ids: List of product IDs to enrich
        products_dict: Dictionary of product data

    Returns:
        List of enriched product dictionaries
    """
    enriched_products = []

    for product_id in product_ids:
        product = products_dict.get(str(product_id))
        if product:
            # Create enriched content for ranking
            content_parts = [
                product.get("product_description", ""),
                f"Category: {product.get('product_class', '')}",
                f"Price: ${product.get('price', 'N/A')}",
                f"Rating: {product.get('average_rating', 'N/A')} ({product.get('rating_count', 'N/A')} reviews)",
                f"Colors: {', '.join(product.get('color', []))}",
                f"Materials: {', '.join(product.get('material', []))}",
                f"Styles: {', '.join(product.get('style', []))}",
                f"Country: {', '.join(product.get('countryoforigin', []))}",
            ]

            enriched_product = {
                "id": str(product_id),
                "title": product.get("product_name", ""),
                "content": "\n".join(content_parts),
            }
            enriched_products.append(enriched_product)
        else:
            logger.warning(f"Product not found: {product_id}")

    logger.debug(f"Enriched {len(enriched_products)} products")
    return enriched_products


def create_ranking_client() -> discoveryengine.RankServiceClient:
    """
    Create Discovery Engine ranking client.

    Returns:
        Configured RankServiceClient instance

    Raises:
        Exception: If client creation fails
    """
    try:
        client = discoveryengine.RankServiceClient()
        logger.info("Created Discovery Engine ranking client")
        return client

    except Exception as e:
        logger.error(f"Failed to create ranking client: {e}")
        raise


def rerank_products(
    query: str,
    enriched_products: List[Dict[str, Any]],
    project_id: str,
    location: str,
    top_k: int,
    ranking_model: str = RANKING_MODEL,
    client=None,
) -> List[Dict[str, Any]]:
    """
    Rerank products using Discovery Engine ranking API.

    Args:
        query: Search query text
        enriched_products: List of enriched product dictionaries
        project_id: GCP project ID
        location: GCP location
        top_k: Number of top results to return
        ranking_model: Ranking model to use

    Returns:
        List of ranked product dictionaries
    """
    try:
        if client is None:
            client = create_ranking_client()

        ranking_config = client.ranking_config_path(
            project=project_id,
            location=location,
            ranking_config="default_ranking_config",
        )

        # Create ranking records
        records = []
        for product in enriched_products:
            records.append(
                discoveryengine.RankingRecord(
                    id=product["id"],
                    title=product["title"],
                    content=product["content"],
                )
            )

        # Create ranking request
        request = discoveryengine.RankRequest(
            ranking_config=ranking_config,
            model=ranking_model,
            top_n=top_k,
            query=query,
            records=records,
        )

        # Perform ranking
        response = client.rank(request=request)

        # Process ranked results
        ranked_products = []
        if hasattr(response, "records") and response.records:
            for ranked_record in response.records:
                product = next(
                    (p for p in enriched_products if p["id"] == ranked_record.id), None
                )
                if product:
                    ranked_products.append(
                        {
                            "id": ranked_record.id,
                            "title": product.get("title", ""),
                            "score": getattr(ranked_record, "score", 1.0),
                            "rank": len(ranked_products) + 1,
                        }
                    )
        else:
            # Fallback: return products in original order
            logger.warning("No ranked records returned, using original order")
            ranked_products = [
                {
                    "id": p["id"],
                    "title": p.get("title", ""),
                    "score": 1.0,
                    "rank": i + 1,
                }
                for i, p in enumerate(enriched_products[:top_k])
            ]

        logger.debug(f"Ranked {len(ranked_products)} products for query")
        return ranked_products

    except Exception as e:
        logger.error(f"Ranking failed for query '{query}': {e}")
        # Return fallback results
        return [
            {"id": p["id"], "title": p.get("title", ""), "score": 1.0, "rank": i + 1}
            for i, p in enumerate(enriched_products[:top_k])
        ]


def save_ranked_results(ranked_results: Dict[str, List[str]]) -> None:
    """
    Save ranked results to output file.

    Args:
        ranked_results: Dictionary mapping query IDs to ranked product ID lists
    """
    try:
        with open(RANKED_RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(ranked_results, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Saved ranked results for {len(ranked_results)} queries to {RANKED_RESULTS_FILE}"
        )

    except Exception as e:
        logger.error(f"Failed to save ranked results: {e}")
        raise


def process_queries_for_reranking(
    queries: List[Dict[str, Any]],
    query_results: Dict[str, List[str]],
    products_dict: Dict[str, Dict[str, Any]],
) -> Dict[str, List[str]]:
    """
    Process all queries for reranking.

    Args:
        queries: List of query dictionaries
        query_results: Dictionary of search results
        products_dict: Dictionary of product data

    Returns:
        Dictionary of ranked results
    """
    all_reranking_results = {}
    available_queries = [
        q["query_id"] for q in queries if q["query_id"] in list(query_results.keys())
    ]

    logger.info(f"Processing {len(available_queries)} queries for reranking")

    # Create ranking client once for all queries
    ranking_client = create_ranking_client()

    for query_id in available_queries:
        logger.info(f"Processing query: {query_id}")

        try:
            # Find query data
            query_data = next((q for q in queries if q["query_id"] == query_id), None)
            if not query_data:
                logger.warning(f"Query data not found for {query_id}")
                continue

            query_text = query_data["query_text"]
            product_ids = query_results.get(query_id, [])

            if not product_ids:
                logger.warning(f"No products found for query {query_id}")
                all_reranking_results[query_id] = []
                continue

            # Enrich products with data
            enriched_products = enrich_products_with_data(product_ids, products_dict)

            if not enriched_products:
                logger.warning(f"No enriched products for query {query_id}")
                all_reranking_results[query_id] = []
                continue

            # Rerank products
            ranked_products = rerank_products(
                query=query_text,
                enriched_products=enriched_products,
                project_id=PROJECT_ID,
                location=LOCATION,
                top_k=RANKING_TOP_K,
                client=ranking_client,
            )

            # Extract product IDs from ranked results
            ranked_product_ids = [ranking["id"] for ranking in ranked_products]
            all_reranking_results[query_id] = ranked_product_ids

            logger.info(
                f"Ranked {len(ranked_product_ids)} products for query {query_id}"
            )

        except Exception as e:
            logger.error(f"Failed to process query {query_id}: {e}")
            all_reranking_results[query_id] = []

    return all_reranking_results


def main():
    try:
        logger.info("Starting product reranking...")

        # Load queries and results
        queries = load_queries_from_bucket()
        query_results = load_results_from_file(RESULTS_FILE)

        logger.info(
            f"Loaded {len(queries)} queries and results for {len(query_results)} queries"
        )

        # Load product data
        products_dict = load_products_from_bucket()

        # Process queries for reranking
        all_reranking_results = process_queries_for_reranking(
            queries, query_results, products_dict
        )

        # Save ranked results
        save_ranked_results(all_reranking_results)

        logger.info(f"Reranking completed for {len(all_reranking_results)} queries")

    except Exception as e:
        logger.error(f"Product reranking failed: {e}")
        raise


if __name__ == "__main__":
    main()
