#!/usr/bin/env python3
"""
Vertex AI Search Ranking

This module performs semantic search using Google Cloud's Vertex AI Search
with configurable filtering, boost specifications, and parameter-based constraints.
"""


import argparse
import json
import logging
from itertools import islice
from typing import Any, Dict, List, Optional, Tuple

from config import (ENGINE_ID, FULL_TEXT_OUTPUT_RESULTS, GCS_BUCKET_NAME,
                    GCS_PRODUCTS_BLOB, GCS_QUERIES_BLOB,
                    GT_PARAMS_OUTPUT_RESULTS, LOCATION, NLQ_OUTPUT_RESULTS,
                    NLQ_PARAMS_QUERIES, PROJECT_ID, RANKING_TOP_K)
from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine_v1 as discoveryengine
from google.cloud import storage

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
    parser = argparse.ArgumentParser(description="Vertex AI Search Ranking")
    parser.add_argument(
        "--params",
        type=str,
        default="none",
        choices=["nlq", "predefined", "none"],
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


def build_filter(query_params: Dict[str, Any]) -> Optional[str]:
    """
    Build Vertex AI Search filter expression from query parameters.

    Args:
        query_params: Dictionary containing query parameters

    Returns:
        Filter expression string or None if no filters
    """
    if not query_params:
        return None

    conditions: List[str] = []

    # Attribute filters
    for attribute_key in ["color", "material", "style"]:
        attribute_value = query_params.get(attribute_key)
        if attribute_value:
            conditions.append(f'{attribute_key}: ANY("{attribute_value}")')

    # Negation filters
    negations_map = {
        "style_negated": "style",
        "color_negated": "color",
        "material_negated": "material",
    }
    for negated_key, base_key in negations_map.items():
        negated_value = query_params.get(negated_key)
        if negated_value:
            conditions.append(f'NOT {base_key}: ANY("{negated_value}")')

    # Numeric constraints
    if (price_max := query_params.get("price_max")) is not None:
        conditions.append(f"price <= {float(price_max)}")
    if (rating_min := query_params.get("rating_min")) is not None:
        conditions.append(f"average_rating >= {float(rating_min)}")
    if (rating_count_min := query_params.get("rating_count_min")) is not None:
        conditions.append(f"rating_count >= {float(rating_count_min)}")

    filter_expression = " AND ".join(conditions) if conditions else None

    if filter_expression:
        logger.debug(f"Built filter expression: {filter_expression}")

    return filter_expression


def quantile(sorted_values: List[float], q: float) -> Optional[float]:
    """
    Calculate quantile from sorted values.

    Args:
        sorted_values: List of sorted numeric values
        q: Quantile value (0.0 to 1.0)

    Returns:
        Quantile value or None if no values
    """
    if not sorted_values:
        return None
    q = max(0.0, min(1.0, q))
    idx = int(q * (len(sorted_values) - 1))
    return sorted_values[idx]


def compute_numeric_stats(values: List[float]) -> Dict[str, Optional[float]]:
    """
    Compute statistical percentiles from numeric values.

    Args:
        values: List of numeric values

    Returns:
        Dictionary with p25, p50, p75, p90 percentiles
    """
    sv = sorted(v for v in values if isinstance(v, (int, float)))
    return {
        "p25": quantile(sv, 0.25),
        "p50": quantile(sv, 0.50),
        "p75": quantile(sv, 0.75),
        "p90": quantile(sv, 0.90),
    }


def load_product_stats_from_bucket(
    bucket_name: str, blob_path: str
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Load product statistics from GCS bucket.

    Args:
        bucket_name: Name of the GCS bucket
        blob_path: Path to the products file in the bucket

    Returns:
        Dictionary with price, rating, and rating_count statistics

    Raises:
        Exception: If loading from bucket fails
    """
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        text = blob.download_as_text(encoding="utf-8")

        prices: List[float] = []
        ratings: List[float] = []
        rating_counts: List[float] = []

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue

            # Extract price
            p = item.get("price")
            if isinstance(p, (int, float)):
                prices.append(float(p))

            # Extract rating
            r = item.get("average_rating")
            if isinstance(r, (int, float)):
                ratings.append(float(r))

            # Extract rating count
            rc = item.get("rating_count")
            if isinstance(rc, (int, float)):
                rating_counts.append(float(rc))

        stats = {
            "price": compute_numeric_stats(prices),
            "average_rating": compute_numeric_stats(ratings),
            "rating_count": compute_numeric_stats(rating_counts),
        }

        logger.info(f"Loaded stats for {len(prices)} products")
        return stats

    except Exception as e:
        logger.error(f"Failed to load product stats from bucket: {e}")
        raise


def get_boost(x: float) -> float:
    """
    Clamp boost value to valid range.

    Args:
        x: Boost value

    Returns:
        Clamped boost value between -0.9 and 0.9
    """
    return max(-0.9, min(0.9, x))


def build_boost_spec(
    query_params: Dict[str, Any],
    stats: Dict[str, Dict[str, Optional[float]]],
) -> Optional[discoveryengine.SearchRequest.BoostSpec]:
    """
    Build boost specification for personalized ranking.

    Args:
        query_params: Query parameters containing weight preferences
        stats: Product statistics for percentile calculations

    Returns:
        Boost specification or None if no boosts
    """
    cond_specs: List[discoveryengine.SearchRequest.BoostSpec.ConditionBoostSpec] = []

    price_stats = stats.get("price", {})
    rating_stats = stats.get("average_rating", {})
    rating_count_stats = stats.get("rating_count", {})

    # Price weight handling
    pw = query_params.get("price_weight")
    if isinstance(pw, (int, float)) and pw != 0:
        p25 = price_stats.get("p25")
        p50 = price_stats.get("p50")
        p75 = price_stats.get("p75")
        p90 = price_stats.get("p90")

        if pw < 0:  # Prefer lower prices
            if p25 is not None:
                cond_specs.append(
                    discoveryengine.SearchRequest.BoostSpec.ConditionBoostSpec(
                        condition=f"price <= {p25}", boost=get_boost(0.6 * abs(pw))
                    )
                )
            if p50 is not None:
                cond_specs.append(
                    discoveryengine.SearchRequest.BoostSpec.ConditionBoostSpec(
                        condition=f"price <= {p50}", boost=get_boost(0.3 * abs(pw))
                    )
                )
        else:  # Prefer higher prices
            if p75 is not None:
                cond_specs.append(
                    discoveryengine.SearchRequest.BoostSpec.ConditionBoostSpec(
                        condition=f"price >= {p75}", boost=get_boost(0.3 * pw)
                    )
                )
            if p90 is not None:
                cond_specs.append(
                    discoveryengine.SearchRequest.BoostSpec.ConditionBoostSpec(
                        condition=f"price >= {p90}", boost=get_boost(0.5 * pw)
                    )
                )

    # Rating weight handling
    rw = query_params.get("rating_weight")
    if isinstance(rw, (int, float)) and rw != 0:
        r75 = rating_stats.get("p75")
        r90 = rating_stats.get("p90")
        r25 = rating_stats.get("p25")

        if rw > 0:  # Prefer higher ratings
            if r90 is not None:
                cond_specs.append(
                    discoveryengine.SearchRequest.BoostSpec.ConditionBoostSpec(
                        condition=f"average_rating >= {r90}", boost=get_boost(0.5 * rw)
                    )
                )
            if r75 is not None:
                cond_specs.append(
                    discoveryengine.SearchRequest.BoostSpec.ConditionBoostSpec(
                        condition=f"average_rating >= {r75}", boost=get_boost(0.3 * rw)
                    )
                )
        else:  # Prefer lower ratings
            if r25 is not None:
                cond_specs.append(
                    discoveryengine.SearchRequest.BoostSpec.ConditionBoostSpec(
                        condition=f"average_rating <= {r25}", boost=get_boost(0.3 * rw)
                    )
                )

    # Rating count weight handling
    rcw = query_params.get("rating_count_weight")
    if isinstance(rcw, (int, float)) and rcw != 0:
        rc75 = rating_count_stats.get("p75")
        rc90 = rating_count_stats.get("p90")
        rc25 = rating_count_stats.get("p25")

        if rcw > 0:  # Prefer more reviews
            if rc90 is not None:
                cond_specs.append(
                    discoveryengine.SearchRequest.BoostSpec.ConditionBoostSpec(
                        condition=f"rating_count >= {int(rc90)}",
                        boost=get_boost(0.4 * rcw),
                    )
                )
            if rc75 is not None:
                cond_specs.append(
                    discoveryengine.SearchRequest.BoostSpec.ConditionBoostSpec(
                        condition=f"rating_count >= {int(rc75)}",
                        boost=get_boost(0.2 * rcw),
                    )
                )
        else:  # Prefer fewer reviews
            if rc25 is not None:
                cond_specs.append(
                    discoveryengine.SearchRequest.BoostSpec.ConditionBoostSpec(
                        condition=f"rating_count <= {int(rc25)}",
                        boost=get_boost(0.3 * rcw),
                    )
                )

    if not cond_specs:
        return None

    return discoveryengine.SearchRequest.BoostSpec(condition_boost_specs=cond_specs)


def create_search_client(location: str) -> discoveryengine.SearchServiceClient:
    """
    Create Vertex AI Search client.

    Args:
        location: GCP location for the search service

    Returns:
        Configured SearchServiceClient instance
    """
    try:
        client_options = (
            ClientOptions(api_endpoint=f"{location}-discoveryengine.googleapis.com")
            if location != "global"
            else None
        )

        client = discoveryengine.SearchServiceClient(client_options=client_options)
        logger.info(f"Created search client for location: {location}")
        return client

    except Exception as e:
        logger.error(f"Failed to create search client: {e}")
        raise


def search_products(
    client: discoveryengine.SearchServiceClient,
    project_id: str,
    location: str,
    engine_id: str,
    search_query: str,
    filter_expression: Optional[str] = None,
    boost_spec: Optional[discoveryengine.SearchRequest.BoostSpec] = None,
) -> discoveryengine.services.search_service.pagers.SearchPager:
    """
    Search products using Vertex AI Search.

    Args:
        project_id: GCP project ID
        location: GCP location
        engine_id: Search engine ID
        search_query: Search query text
        filter_expression: Optional filter expression
        boost_spec: Optional boost specification

    Returns:
        Search results pager

    Raises:
        Exception: If search fails
    """
    try:

        serving_config = (
            f"projects/{project_id}/locations/{location}/collections/default_collection/engines/"
            f"{engine_id}/servingConfigs/default_config"
        )

        content_search_spec = discoveryengine.SearchRequest.ContentSearchSpec(
            snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
                return_snippet=True
            ),
            summary_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec(
                summary_result_count=1,
                include_citations=True,
                ignore_adversarial_query=True,
                ignore_non_summary_seeking_query=True,
                model_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec.ModelSpec(
                    version="stable",
                ),
            ),
        )

        request = discoveryengine.SearchRequest(
            serving_config=serving_config,
            query=search_query,
            page_size=10,
            content_search_spec=content_search_spec,
            query_expansion_spec=discoveryengine.SearchRequest.QueryExpansionSpec(
                condition=discoveryengine.SearchRequest.QueryExpansionSpec.Condition.AUTO,
            ),
            spell_correction_spec=discoveryengine.SearchRequest.SpellCorrectionSpec(
                mode=discoveryengine.SearchRequest.SpellCorrectionSpec.Mode.AUTO
            ),
        )

        if filter_expression:
            request.filter = filter_expression
        if boost_spec:
            request.boost_spec = boost_spec

        logger.debug(f"Searching with query: {search_query}")
        if filter_expression:
            logger.debug(f"Using filter: {filter_expression}")
        if boost_spec:
            logger.debug(
                f"Using boost spec with {len(boost_spec.condition_boost_specs)} conditions"
            )
        logger.debug(f"Request: {request}")
        return client.search(request)

    except Exception as e:
        logger.error(f"Search failed for query '{search_query}': {e}")
        raise


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
        params_type: Type of parameters to use ("nlq", "predefined", "none")

    Returns:
        List of query dictionaries
    """

    if params_type == "nlq":
        queries = load_queries_from_file()
        for q in queries:
            if "query_params" not in q or q["query_params"] is None:
                q["query_params"] = {}
        return queries
    elif params_type == "predefined":
        queries = load_queries_from_bucket()
        return queries
    elif params_type == "none":
        queries = load_queries_from_bucket()
        for q in queries:
            q["query_params"] = {}
        return queries

    else:
        raise ValueError(f"Unknown parameter type: {params_type}")


def process_queries(
    client: discoveryengine.SearchServiceClient,
    queries: List[Dict[str, Any]],
    filter_flag: bool,
    stats: Dict[str, Dict[str, Optional[float]]],
) -> Dict[str, List[str]]:
    """
    Process all queries and return results.

    Args:
        queries: List of query dictionaries
        filter_flag: Whether to apply filters and boosts
        stats: Product statistics for boost calculations

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
            # Build filter and boost specifications
            filter_expression = None
            boost_spec = None

            if filter_flag:
                filter_expression = build_filter(query_params)
                boost_spec = build_boost_spec(query_params, stats)

            # Perform search
            pager = search_products(
                client=client,
                project_id=PROJECT_ID,
                location=LOCATION,
                engine_id=ENGINE_ID,
                search_query=query_text,
                filter_expression=filter_expression,
                boost_spec=boost_spec,
            )

            # Extract product IDs from results
            product_ids = []
            for result in islice(pager, RANKING_TOP_K):
                product_id = result.document.struct_data.get("product_id")
                if product_id:
                    product_ids.append(product_id)
                    logger.debug(
                        f"Result product_id: {product_id}, "
                        f"product_name: {result.document.struct_data.get('product_name')}, "
                        f"product_class: {result.document.struct_data.get('product_class')}"
                    )

            query_results[query_id] = product_ids
            logger.info(f"Found {len(product_ids)} products for query {query_id}")
        except Exception as e:
            logger.error(f"Failed to process query {query_id}: {e}")
            query_results[query_id] = []

    return query_results


def main():
    filter_flag = False
    output_path = None
    # Parse arguments
    args = parse_arguments()
    if args.params == "gt":
        filter_flag = True
        output_path = GT_PARAMS_OUTPUT_RESULTS
    elif args.params == "nlq":
        filter_flag = True
        output_path = NLQ_OUTPUT_RESULTS
    elif args.params == "none":
        output_path = FULL_TEXT_OUTPUT_RESULTS

    # Load queries based on parameter type
    queries = load_queries_by_params_type(args.params)

    if not queries:
        logger.warning("No queries found to process")
        return

    try:
        client = create_search_client(LOCATION)
    except Exception as e:
        logger.error(f"Failed to create search client: {e}")
        raise

    # Load product statistics for boost calculations
    stats = load_product_stats_from_bucket(GCS_BUCKET_NAME, GCS_PRODUCTS_BLOB)

    # Process queries
    query_results = process_queries(client, queries, filter_flag, stats)

    # Save results
    save_results(query_results, output_path=output_path)

    logger.info(f"Successfully processed {len(queries)} queries")


if __name__ == "__main__":
    main()
