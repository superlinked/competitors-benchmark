import json
import logging
import os
import random
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import google.generativeai as genai
import pandas as pd
from config import (DEFAULT_TOP_K, FEATURE_EXTRACTION_MAX_RETRIES,
                    FEATURE_EXTRACTION_PROMPT_TEMPLATE,
                    FEATURE_EXTRACTION_RETRY_DELAY, FEATURE_EXTRACTION_WORKERS,
                    GEMINI_MODEL, GOOGLE_API_KEY, INPUT_DIR, MAX_DESCRIPTIONS,
                    MIN_RESULTS, OUTPUT_DIR, QDRANT_HOST, QDRANT_PORT,
                    QUERIES_JSON, QUERIES_TEMP_JSON, SEARCH_RESULTS_JSON,
                    SEARCH_RESULTS_TEMP_JSON, UNIQUE_CATEGORIES_JSON)
from data_util import (load_product_data, load_queries, load_results,
                       load_synth_queries_results)
from qdrant_client import QdrantClient
from qdrant_util import create_filters
from qdrant_util import search as qdrant_search
from tqdm import tqdm

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def setup_gemini():
    """Setup Gemini model.

    Returns:
        GenerativeModel: Gemini model.
    """
    if not GOOGLE_API_KEY:
        logger.warning("No Google API key found")
        return None

    genai.configure(api_key=GOOGLE_API_KEY)
    return genai.GenerativeModel(GEMINI_MODEL)


def setup_qdrant():
    """Setup Qdrant client.

    Returns:
        QdrantClient: Qdrant client.
    """
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def load_data():
    """Load data.

    Returns:
        Tuple[List[Dict], Dict[str, List[int]], pd.DataFrame]: Queries, results, products DataFrame.
    """
    queries = load_queries(QUERIES_TEMP_JSON)
    results = load_results(SEARCH_RESULTS_TEMP_JSON)
    products_list = load_product_data()
    products_df = pd.DataFrame(products_list)
    return queries, results, products_df


def group_queries_by_class(queries: List, results) -> List[Tuple[str, List[Dict]]]:
    """Group queries by class.

    Args:
        queries: List of queries.
        results: Dictionary of results.

    Returns:
        List[Tuple[str, List[Dict]]]: List of queries by class.
    """
    queries_by_class = []
    for query in queries:
        if query.get("query_type") == "synth":
            continue
        product_class = query.get("query_params", {}).get("product_class", "")
        if product_class and query["query_id"] in results:
            queries_by_class.append((product_class, query))

    # Group by product class
    class_groups = {}
    for product_class, query in queries_by_class:
        if product_class not in class_groups:
            class_groups[product_class] = []
        class_groups[product_class].append(query)

    return [(product_class, queries) for product_class, queries in class_groups.items()]


def get_unique_descriptions(
    queries: List, results, products_df: pd.DataFrame, max_count: int = MAX_DESCRIPTIONS
) -> List[str]:
    """Get unique descriptions.

    Args:
        queries: List of queries.
        results: Dictionary of results.
        products_df: Products DataFrame.
        max_count: Maximum count of descriptions.

    Returns:
        List[str]: List of unique descriptions.
    """
    descriptions = set()
    for query in queries:
        if query["query_id"] in results:
            retrieved_ids = results[query["query_id"]]
            retrieved_products = products_df[
                products_df["product_id"].isin(retrieved_ids)
            ]
            descriptions.update(retrieved_products["product_description"].dropna())

    return sorted(descriptions, key=len, reverse=True)[:max_count]


def build_feature_extraction_prompt(
    product_class: str, descriptions: str, column_values
) -> str:
    """Build feature extraction prompt.

    Args:
        product_class: Product class.
        descriptions: Descriptions.
        column_values: Column values.

    Returns:
        str: Feature extraction prompt.
    """
    return FEATURE_EXTRACTION_PROMPT_TEMPLATE.format(
        product_class=product_class,
        product_class_lower=product_class.lower(),
        descriptions=descriptions,
        column_values=column_values,
    )


def make_llm_request_with_retry(
    model, prompt: str, max_retries: int = FEATURE_EXTRACTION_MAX_RETRIES
) -> str:
    """Make LLM request with retry.

    Args:
        model: Model.
        prompt: Prompt.
        max_retries: Maximum retries.

    Returns:
        str: Response text.
    """
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            if "504" in str(e) or "Deadline" in str(e):
                if attempt < max_retries - 1:
                    logger.warning(f"Timeout on attempt {attempt + 1}, retrying...")
                    time.sleep(FEATURE_EXTRACTION_RETRY_DELAY)
                    continue
                else:
                    raise e
            else:
                raise e


def parse_llm_response(response_text: str, product_class: str):
    """Parse LLM response.

    Args:
        response_text: Response text.
        product_class: Product class.

    Returns:
        Dict: Parsed response.
    """
    if not response_text or response_text.strip() == "":
        logger.warning(f"Empty response from Gemini for {product_class}")
        raise Exception("Empty response from Gemini")

    # Clean up markdown code blocks - handle various formats
    text = response_text.strip()

    # Remove opening code blocks (```json, ```python, ```, etc.)
    if text.startswith("```"):
        # Find the first newline after ```
        newline_pos = text.find("\n")
        if newline_pos != -1:
            text = text[newline_pos + 1 :]  # Remove ```lang and newline
        else:
            text = text[3:]  # Just remove ```

    # Remove closing code blocks
    if text.endswith("```"):
        text = text[:-3]  # Remove trailing ```

    text = text.strip()

    try:
        features = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON response for {product_class}: {e}")
        logger.warning(f"Response text: {response_text}")
        logger.warning(f"Cleaned text: {text}")
        raise Exception(f"Invalid JSON response: {e}")

    required_keys = [
        "FUNCTIONAL_FEATURES",
        "USAGE_CONTEXTS",
        "SIZE_DESCRIPTORS",
        "EMOTIONAL_DESCRIPTORS",
        "CONSTRUCTION_FEATURES",
        "BENEFIT_PHRASES",
    ]
    for key in required_keys:
        if key not in features or not isinstance(features[key], list):
            features[key] = []

    logger.info(f"Extracted features for {product_class}: {features}")
    return features


def call_llm_for_features(product_class: str, descriptions: List[str], gemini_model):
    """Call LLM for features.

    Args:
        product_class: Product class.
        descriptions: List of descriptions.
        gemini_model: Gemini model.

    Returns:
        Dict: Features.
    """
    try:
        with open(os.path.join(INPUT_DIR, UNIQUE_CATEGORIES_JSON), "r") as f:
            column_values_dict = json.load(f)
            # Flatten all values from the dictionary into a single list
            column_values = []
            for values_list in column_values_dict.values():
                column_values.extend(values_list)

        combined_descriptions = "\n\n".join(descriptions)
        prompt = build_feature_extraction_prompt(
            product_class, combined_descriptions, column_values
        )
        response_text = make_llm_request_with_retry(gemini_model, prompt)
        return parse_llm_response(response_text, product_class)

    except Exception as e:
        logger.warning(f"Failed to extract features for {product_class}: {e}")
        return {
            "FUNCTIONAL_FEATURES": [],
            "USAGE_CONTEXTS": [],
            "SIZE_DESCRIPTORS": [],
            "EMOTIONAL_DESCRIPTORS": [],
            "CONSTRUCTION_FEATURES": [],
            "BENEFIT_PHRASES": [],
        }


def process_product_class(
    product_class: str,
    class_queries: List,
    results,
    products_df: pd.DataFrame,
    gemini_model,
) -> Tuple[str, Optional[Dict]]:
    """Process product class.

    Args:
        product_class: Product class.
        class_queries: List of class queries.
        results: Dictionary of results.
        products_df: Products DataFrame.
        gemini_model: Gemini model.

    Returns:
        Tuple[str, Optional[Dict]]: Product class, features.
    """
    logger.info(f"Extracting features for {product_class}")

    descriptions = get_unique_descriptions(class_queries, results, products_df)

    if descriptions:
        result = call_llm_for_features(product_class, descriptions, gemini_model)
        return product_class, result
    return product_class, None


def extract_features(
    queries: List, results, products_df: pd.DataFrame, gemini_model
) -> Dict[str, Optional[Dict]]:
    """Extract features.

    Args:
        queries: List of queries.
        results: Dictionary of results.
        products_df: Products DataFrame.
        gemini_model: Gemini model.

    Returns:
        Dict[str, Optional[Dict]]: Features.
    """
    if not gemini_model:
        logger.warning("No Gemini model available")
        return {}

    queries_by_class = group_queries_by_class(queries, results)
    features = {}

    with ThreadPoolExecutor(max_workers=FEATURE_EXTRACTION_WORKERS) as executor:
        futures = []
        for product_class, class_queries in queries_by_class:
            future = executor.submit(
                process_product_class,
                product_class,
                class_queries,
                results,
                products_df,
                gemini_model,
            )
            futures.append(future)

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Extracting features"
        ):
            product_class, result = future.result()
            if result:
                features[product_class] = result
                logger.info(f"Completed feature extraction for {product_class}")

    return features


def enrich_query(original_query, features):
    """Enrich query.

    Args:
        original_query: Original query.
        features: Features.

    Returns:
        Dict: Enriched query.
    """
    base_text = original_query["query_text"]

    feature_types = [
        "FUNCTIONAL_FEATURES",
        "USAGE_CONTEXTS",
        "SIZE_DESCRIPTORS",
        "EMOTIONAL_DESCRIPTORS",
        "CONSTRUCTION_FEATURES",
        "BENEFIT_PHRASES",
    ]

    available_features = {}
    for feature_type in feature_types:
        if features.get(feature_type):
            available_features[feature_type] = features[feature_type]

    if not available_features:
        return original_query

    num_feature_types = min(1, len(available_features))
    selected_features = random.sample(
        list(available_features.keys()), num_feature_types
    )

    enriched_text = base_text
    for feature_type in selected_features:
        feature_list = available_features[feature_type]
        selected_feature_term = random.choice(feature_list)

        if feature_type == "USAGE_CONTEXTS":
            enriched_text = f"{enriched_text}, for {selected_feature_term}"
        else:
            enriched_text = f"{selected_feature_term} {enriched_text}"

    enriched_query = original_query.copy()
    enriched_query["query_text"] = enriched_text
    enriched_query["query_params"]["product_description"] = enriched_text

    if enriched_text == base_text:
        logger.warning(
            f"Enriched query is the same as the original query for {original_query['query_id']}"
        )
    return enriched_query


def validate_query(query, qdrant_client) -> Optional[List[str]]:
    """Validate query.

    Args:
        query: Query.
        qdrant_client: Qdrant client.

    Returns:
        Optional[List[str]]: List of product IDs.
    """
    try:
        query_text = query["query_params"]["product_description"]
        query_params = {k: v for k, v in query.get("query_params", {}).items() if v}

        filters = create_filters(query)
        results = qdrant_search(
            qdrant_client,
            query_text,
            top_k=DEFAULT_TOP_K,
            filters=filters,
            query_params=query_params,
        )

        if results and len(results) >= MIN_RESULTS:
            return [r.get("product_id") for r in results]
        return None

    except Exception as e:
        logger.warning(f"Validation error: {e}")
        return None


def create_enriched_queries(queries: List, features, qdrant_client):
    """Create enriched queries.

    Args:
        queries: List of queries.
        features: Features.
        qdrant_client: Qdrant client.

    Returns:
        Tuple[List[Dict], Dict[str, List[int]]]: Enriched queries, enriched results.
    """
    enriched_queries = []
    enriched_results = {}

    non_synthetic_queries = [q for q in queries if q.get("query_type") != "synth"]

    for query in non_synthetic_queries:
        product_class = query.get("query_params", {}).get("product_class", "")

        if product_class in features:
            enriched_query = enrich_query(query, features[product_class])
            result_ids = validate_query(enriched_query, qdrant_client)
            logger.info(f"Original query: {query['query_text']}")
            logger.info(
                f"Enriched query: {enriched_query['query_text']} with {result_ids}"
            )
            if result_ids:
                enriched_queries.append(enriched_query)
                enriched_results[enriched_query["query_id"]] = result_ids
            else:
                logger.info(
                    f"No results found for enriched query: {enriched_query['query_text']}"
                )
                enriched_queries.append(query)
                enriched_results[query["query_id"]] = []

    return enriched_queries, enriched_results


def combine_with_synth(
    enriched_queries: List, enriched_results, synth_queries: List, synth_results: Dict
):
    """Combine with synth.

    Args:
        enriched_queries: List of enriched queries.
        enriched_results: Dictionary of enriched results.
        synth_queries: List of synth queries.
        synth_results: Dictionary of synth results.
    """
    for query in synth_queries:
        enriched_queries.append(query)
        enriched_results[query["query_id"]] = synth_results[query["query_id"]]

    return enriched_queries, enriched_results


def save_results(enriched_queries: List, enriched_results):
    """Save results.

    Args:
        enriched_queries: List of enriched queries.
        enriched_results: Dictionary of enriched results.
    """
    with open(os.path.join(OUTPUT_DIR, QUERIES_JSON), "w") as f:
        json.dump(enriched_queries, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, SEARCH_RESULTS_JSON), "w") as f:
        json.dump(enriched_results, f, indent=2)

    logger.info(f"Saved {len(enriched_queries)} enriched queries and results")


def main():
    try:
        logger.info("Starting query enrichment pipeline")

        gemini_model = setup_gemini()
        qdrant_client = setup_qdrant()

        if not gemini_model:
            raise Exception("Cannot proceed without Gemini model")

        queries, results, products_df = load_data()

        logger.info("Extracting features from products")
        features = extract_features(queries, results, products_df, gemini_model)

        logger.info("Creating enriched queries")
        enriched_queries, enriched_results = create_enriched_queries(
            queries, features, qdrant_client
        )

        logger.info("Combining with synth")
        synth_queries, synth_results = load_synth_queries_results()
        enriched_queries, enriched_results = combine_with_synth(
            enriched_queries, enriched_results, synth_queries, synth_results
        )

        save_results(enriched_queries, enriched_results)

        logger.info(
            f"Enrichment completed successfully: {len(enriched_queries)-len(synth_queries)} enriched queries"
        )

    except Exception as e:
        logger.error(f"Enrichment failed: {e}")
        raise


if __name__ == "__main__":
    main()
