import json
import logging
import os
from typing import Any, Dict, List

from config import (LOCATION, OUTPUT_DIR, PRODUCTS_FILE, PROJECT_ID,
                    QUERIES_FILE, RANKED_RESULTS_JSON, RANKING_MODEL,
                    RANKING_TOP_K, SEARCH_RESULTS_JSON, UNIQUE_CATEGORIES_JSON)
from data_util import (load_product_data, load_queries, load_search_results,
                       load_unique_categories, upload_to_gcs)
from google.cloud import discoveryengine_v1 as discoveryengine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def push_dataset_to_gcs(
    products: List[Dict[str, Any]],
    queries: List[Dict[str, Any]],
    ranked_results: Dict[str, List[str]],
    search_results: Dict[str, List[str]],
    unique_categories: Dict[str, List[str]],
):
    """Push dataset to GCS.

    Args:
        products: List of products.
        queries: List of queries.
        ranked_results: Dictionary of ranked results.
        search_results: Dictionary of search results.
    Returns:
        None
    """

    unique_categories = load_unique_categories()

    upload_to_gcs(products, PRODUCTS_FILE.split("/")[-1])
    upload_to_gcs(queries, QUERIES_FILE.split("/")[-1])
    upload_to_gcs(ranked_results, RANKED_RESULTS_JSON.split("/")[-1])
    upload_to_gcs(search_results, SEARCH_RESULTS_JSON.split("/")[-1])
    upload_to_gcs(unique_categories, UNIQUE_CATEGORIES_JSON.split("/")[-1])


def enrich_products_with_data(
    product_ids: List[str], products_dict: Dict[str, Dict]
) -> List[Dict[str, Any]]:
    """Enrich products with data.

    Args:
        product_ids: List of product IDs.
        products_dict: Dictionary of products.

    Returns:
        List[Dict[str, Any]]: List of enriched products.
    """
    enriched_products = []
    for product_id in product_ids:
        # Try both string and integer keys since product_ids might be either
        product = products_dict.get(str(product_id)) or products_dict.get(
            int(product_id)
        )
        if product:
            content_parts = [
                product.get("product_description", ""),
                f"Category: {product.get('product_class', '')}",
                f"Price: ${product.get('price', 0)}",
                f"Rating: {product.get('average_rating', 0)}",
                f"Rating Count: {product.get('rating_count', 0)}",
                f"Colors: {', '.join(product.get('color', []))}",
                f"Materials: {', '.join(product.get('material', []))}",
                f"Styles: {', '.join(product.get('style', []))}",
                f"Country: {', '.join(product.get('countryoforigin', []))}",
            ]

            enriched_product = {
                "id": str(product_id),
                "title": product.get("product_name", ""),
                "content": " | ".join(content_parts),
            }
            enriched_products.append(enriched_product)

    return enriched_products


def rerank_products(
    query: str,
    enriched_products: List[Dict[str, Any]],
    project_id: str,
    location: str,
    top_k: int,
    ranking_model: str = RANKING_MODEL,
) -> List[Dict[str, Any]]:
    """Rerank products.

    Args:
        query: Query.
        enriched_products: List of enriched products.
        project_id: Project ID.
        location: Location.
        top_k: Top K.
        ranking_model: Ranking model.

    Returns:
        List[Dict[str, Any]]: List of ranked products.
    """
    logger.info(f"Reranking products for query: {query}")
    try:
        client = discoveryengine.RankServiceClient()

        ranking_config = client.ranking_config_path(
            project=project_id,
            location=location,
            ranking_config="default_ranking_config",
        )

        records = []
        for product in enriched_products:
            records.append(
                discoveryengine.RankingRecord(
                    id=product["id"],
                    title=product["title"],
                    content=product["content"],
                )
            )
        logger.info(f"Original order: {[p['id'] for p in enriched_products[:10]]}")
        request = discoveryengine.RankRequest(
            ranking_config=ranking_config,
            model=ranking_model,
            top_n=top_k,
            query=query,
            records=records,
        )

        response = client.rank(request=request)

        ranked_products = []

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
        logger.info(f"Ranked order: {[p['id'] for p in ranked_products[:10]]}")
        if not ranked_products:
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

        return ranked_products

    except Exception as e:
        logger.error(f"Reranking failed for query '{query}': {e}")
        return [
            {"id": p["id"], "title": p.get("title", ""), "score": 1.0, "rank": i + 1}
            for i, p in enumerate(enriched_products[:top_k])
        ]


def main():
    logger.info("Starting reranking...")
    queries = load_queries()
    query_results = load_search_results()
    logger.info(f"Loaded {len(queries)} queries and query results")

    products_df = load_product_data()
    products_dict = products_df.set_index("product_id").to_dict("index")

    all_reranking_results = {}
    available_queries = [
        q["query_id"] for q in queries if q["query_id"] in list(query_results.keys())
    ]

    for query_id in available_queries:
        logger.info(f"Processing query: {query_id}")
        query_data = next((q for q in queries if q["query_id"] == query_id), None)
        if not query_data:
            continue

        query_text = query_data["query_text"]
        product_ids = query_results.get(query_id, [])
        enriched_products = enrich_products_with_data(product_ids, products_dict)
        logger.info(
            f"Reranking {len(enriched_products)} products for query: {query_id}"
        )
        ranked_products = rerank_products(
            query=query_text,
            enriched_products=enriched_products,
            project_id=PROJECT_ID,
            location=LOCATION,
            top_k=RANKING_TOP_K,
        )

        all_reranking_results[query_id] = [ranking["id"] for ranking in ranked_products]

    with open(
        os.path.join(OUTPUT_DIR, RANKED_RESULTS_JSON), "w", encoding="utf-8"
    ) as f:
        json.dump(all_reranking_results, f, indent=2, ensure_ascii=False)

    logger.info(f"Pushing dataset to GCS")
    push_dataset_to_gcs(products_df, queries, all_reranking_results, query_results)
    logger.info(
        f"Products, queries, ranked results, search results, and unique categories pushed to GCS. Dataset generated successfully."
    )


if __name__ == "__main__":
    main()
