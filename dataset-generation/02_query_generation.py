import json
import logging
import os
import random
import re
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from config import (CLEANED_PRODUCTS_JSONL, COLLECTION_NAME,
                    DEFAULT_COMPLEX_QUERIES, DEFAULT_NEGATION_QUERIES,
                    DEFAULT_NUMBER_QUERIES, DEFAULT_SIMPLE_QUERIES,
                    DEFAULT_STAT_VALUE, DEFAULT_TOP_K, INPUT_DIR, MAX_ATTEMPTS,
                    MIN_RESULTS, N_PRODUCT_CLASSES, OUTPUT_DIR,
                    PRICE_BETA_PARAMS, QDRANT_HOST, QDRANT_PORT,
                    QDRANT_TIMEOUT, QUERIES_TEMP_JSON, QUERY_GENERATION_JSONL,
                    RATING_BETA_PARAMS, RATING_COUNT_BETA_PARAMS,
                    SEARCH_RESULTS_TEMP_JSON, SYNTHETIC_QUERIES_JSON,
                    SYNTHETIC_RESULTS_JSON)
from qdrant_client import QdrantClient
# Qdrant helpers
from qdrant_util import create_filters
from qdrant_util import search as qdrant_search

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

QDRANT_CLIENT = None


def initialize_qdrant() -> None:
    """Initialize Qdrant client connection to existing instance."""
    global QDRANT_CLIENT
    logger.info("Connecting to existing Qdrant instance...")

    QDRANT_CLIENT = QdrantClient(
        host=QDRANT_HOST, port=QDRANT_PORT, timeout=QDRANT_TIMEOUT
    )

    # Test the connection by checking if collection exists
    try:
        collection_info = QDRANT_CLIENT.get_collection(COLLECTION_NAME)
        logger.info(
            f"Connected to Qdrant. Collection '{COLLECTION_NAME}' has {collection_info.points_count} points"
        )
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant or collection not found: {e}")
        raise RuntimeError(f"Qdrant connection failed: {e}")

    logger.info("Qdrant connection established successfully")


def load_dataset() -> pd.DataFrame:
    logger.info("Loading product dataset...")
    dataset_path = os.path.join(OUTPUT_DIR, CLEANED_PRODUCTS_JSONL)
    dataset = pd.read_json(dataset_path, lines=True)

    # Restrict to top K most frequent product classes
    logger.info("Restricting to top K product classes by frequency...")
    top_k = (
        dataset["product_class"]
        .value_counts()
        .nlargest(N_PRODUCT_CLASSES)
        .index.tolist()
    )
    dataset = dataset[dataset["product_class"].isin(top_k)].reset_index(drop=True)
    logger.info(
        f"Dataset restricted to {len(dataset)} rows across {len(top_k)} classes"
    )
    logger.info(f"Product classes: {top_k}")
    return dataset


def create_statistics_dicts(dataset: pd.DataFrame) -> tuple:
    logger.info("Creating statistics dictionaries...")

    rating_stats = (
        dataset.groupby("product_class")["average_rating"]
        .agg(["min", "max"])
        .round(2)
        .apply(lambda x: [x["min"], x["max"]], axis=1)
        .to_dict()
    )
    rating_count_stats = (
        dataset.groupby("product_class")["rating_count"]
        .agg(["min", "max"])
        .round(2)
        .apply(lambda x: [x["min"], x["max"]], axis=1)
        .to_dict()
    )
    price_stats = (
        dataset.groupby("product_class")["price"]
        .agg(["min", "max"])
        .round(2)
        .apply(lambda x: [x["min"], x["max"]], axis=1)
        .to_dict()
    )
    logger.info("Statistics dictionaries created")
    return rating_stats, rating_count_stats, price_stats


def create_values_dict(dataset: pd.DataFrame, column: str) -> Dict[str, set]:
    values_dict = {}

    for product_class, group in dataset.groupby("product_class"):
        values = []
        for value_list in group[column].dropna():
            if isinstance(value_list, str):
                vals = [v.strip() for v in value_list.split(",")]
                values.extend(vals)
            elif isinstance(value_list, list):
                values.extend(value_list)

        clean_values = []
        for val in values:
            if isinstance(val, str):
                val = re.sub(r"\d+\s*%", "", val)
                val = " ".join(val.split())
                if val:
                    clean_values.append(val.lower())

        values_dict[product_class] = set(clean_values)

    return values_dict


def create_feature_dictionaries(dataset: pd.DataFrame) -> tuple:
    logger.info("\nCreating feature dictionaries...")

    material_dict = create_values_dict(dataset, "material")
    color_dict = create_values_dict(dataset, "color")
    style_dict = create_values_dict(dataset, "style")

    logger.info("Feature dictionaries created")
    return material_dict, color_dict, style_dict


def create_query_generation_dataframe(dataset: pd.DataFrame) -> pd.DataFrame:
    logger.info("\nCreating query generation dataframe...")

    product_classes = dataset["product_class"].unique().tolist()
    material_dict, color_dict, style_dict = create_feature_dictionaries(dataset)
    rating_stats, rating_count_stats, price_stats = create_statistics_dicts(dataset)

    dicts = [
        material_dict,
        color_dict,
        style_dict,
        rating_stats,
        rating_count_stats,
        price_stats,
    ]
    dict_names = [
        "materials",
        "colors",
        "styles",
        "rating_stats",
        "rating_count_stats",
        "price_stats",
    ]

    df = pd.DataFrame(index=pd.Index(product_classes))
    for d, name in zip(dicts, dict_names):
        df[name] = pd.Series(d)

    df = df.reset_index()
    df = df.rename(columns={"index": "product_class"})
    logger.info("Query generation dataframe created")
    return df


def save_query_generation_data(df: pd.DataFrame) -> None:
    logger.info("\nSaving query generation data...")
    output_path = os.path.join(OUTPUT_DIR, QUERY_GENERATION_JSONL)
    df.to_json(output_path, orient="records", lines=True)
    logger.info(f"Query generation data saved to {output_path}")


def get_random_stat(stat: str, min_val: float, max_val: float) -> float:
    min_val = float(min_val)
    max_val = float(max_val)

    if stat == "rating":
        rating = np.random.beta(*RATING_BETA_PARAMS)
        scaled_rating = min_val + (max_val - min_val) * rating
        return round(scaled_rating, 1)
    elif stat == "rating_count":
        count = np.random.beta(*RATING_COUNT_BETA_PARAMS)
        scaled_count = min_val + (max_val - min_val) * count
        return float(round(scaled_count / 10.0) * 10)
    elif stat == "price":
        price = np.random.beta(*PRICE_BETA_PARAMS)
        scaled_price = min_val + (max_val - min_val) * price
        return float(round(scaled_price / 10.0) * 10)
    else:
        return 0.0


def get_number_representation(stat: str) -> Optional[Dict[str, int]]:
    if stat == "rating":
        return random.choice([{"top rated": 1}, {"highly rated": 1}, {"excellent": 1}])
    elif stat == "rating_count":
        return random.choice(
            [
                {"super popular": 1},
                {"well known": 1},
                {"viral": 1},
            ]
        )
    elif stat == "price":
        return random.choice(
            [{"affordable": -1}, {"budget friendly": -1}, {"cheap": -1}]
        )
    return None


def get_random_category(cats: Any) -> str:
    if isinstance(cats, (list, set)):
        filtered = [x for x in cats if x and str(x).lower() != "unknown"]
        return random.choice(filtered) if filtered else ""
    return ""


def generate_query_schema() -> Dict[str, Any]:
    return {
        "query_id": "",
        "query_type": "",
        "query_text": "",
        "query_params": {
            "product_class": "",
            "product_description": "",
            "style": "",
            "color": "",
            "material": "",
            "style_negated": "",
            "color_negated": "",
            "material_negated": "",
            "rating_min": "",
            "rating_max": "",
            "rating_count_min": "",
            "rating_count_max": "",
            "price_min": "",
            "price_max": "",
            "rating_weight": "",
            "rating_count_weight": "",
            "price_weight": "",
        },
    }


def add_category_to_query(
    query: Dict[str, Any], row: pd.Series, item: str, category: Optional[str] = None
) -> str:
    if category is None:
        category = random.choice(["material", "color", "style"])

    cat_value = get_random_category(row[f"{category}s"])
    query["query_params"][category] = cat_value
    query["query_text"] = build_query_text(item, category, cat_value)
    return category


def cleanup_query_params(query: Dict[str, Any]) -> None:
    query["query_params"] = {k: v for k, v in query["query_params"].items() if v}


def set_query_description(query: Dict[str, Any]) -> None:
    query["query_params"]["product_description"] = query["query_text"]


def build_query_text(item: str, category: str, value: Any) -> str:
    try:
        item = item.lower()
        value = value.lower()
    except:
        logger.warning(f"Error: {value} is not a string")
        return ""

    templates = {
        "material": f"{value} {item}",
        "color": f"{value} {item}",
        "style": f"{item} in {value} style",
    }

    return templates.get(category, "")


def generate_simple_query(row: pd.Series, item: str) -> Dict[str, Any]:
    query = generate_query_schema()
    query["query_type"] = "simple"
    add_category_to_query(query, row, item)
    set_query_description(query)
    return query


def generate_negation_query(row: pd.Series, item: str) -> Dict[str, Any]:
    query = generate_query_schema()
    query["query_type"] = "negation"

    categories = random.sample(["material", "color", "style"], 2)
    negated_cat = random.choice(categories)
    query["query_text"] = ""
    negated_value = None

    for i, cat in enumerate(categories):
        cat_value = get_random_category(row[f"{cat}s"])
        query["query_params"][cat] = cat_value
        if cat == negated_cat:
            negated_value = cat_value
            query["query_text"] += f"not {cat_value} {negated_cat}"
        else:
            query["query_text"] += build_query_text(item, cat, cat_value)
        if i < len(categories) - 1:
            query["query_text"] += ", "

    set_query_description(query)

    if negated_value:
        query["query_params"][f"{negated_cat}_negated"] = negated_value
        del query["query_params"][negated_cat]

    return query


def generate_number_query(row: pd.Series, item: str) -> Dict[str, Any]:
    query = generate_query_schema()
    query["query_type"] = "number"

    add_category_to_query(query, row, item)

    stat = random.choice(["rating", "rating_count", "price"])
    mask = "price_max" if stat == "price" else f"{stat}_min"

    stat_stats = row.get(f"{stat}_stats")
    stat_value = (
        get_random_stat(stat, float(stat_stats[0]), float(stat_stats[1]))
        if stat_stats
        else DEFAULT_STAT_VALUE
    )

    query["query_params"][mask] = stat_value
    temp_text = query["query_text"]

    if stat == "rating":
        query["query_text"] += f", with more than {stat_value:.1f} stars rating"
    elif stat == "rating_count":
        query["query_text"] += f", with more than {int(stat_value)} ratings"
    elif stat == "price":
        query["query_text"] += f", under ${int(stat_value)}"

    query["query_params"]["product_description"] = temp_text
    return query


def generate_complex_query(row: pd.Series, item: str) -> Dict[str, Any]:
    query = generate_query_schema()
    query["query_type"] = "complex"
    query["query_text"] = ""

    stat = random.choice(["rating", "rating_count", "price"])
    stat_representation = get_number_representation(stat)

    if stat_representation:
        key, value = next(iter(stat_representation.items()))
        if stat == "rating":
            query["query_params"][f"average_rating_weight"] = value
        elif stat == "rating_count":
            query["query_params"][f"rating_count_weight"] = value
        elif stat == "price":
            query["query_params"][f"price_weight"] = value

        query["query_text"] += f"{key} "

    color_material = random.choice(["color", "material"])
    cat_value = get_random_category(row[f"{color_material}s"])
    query["query_params"][color_material] = cat_value
    query["query_text"] += build_query_text(item, color_material, cat_value)

    query["query_text"] = query["query_text"].strip()
    set_query_description(query)
    return query


def convert_to_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj


def load_synthetic_queries_results() -> Dict[str, List[str]]:
    logger.info("\nLoading synthetic results...")
    results_file = os.path.join(INPUT_DIR, SYNTHETIC_RESULTS_JSON)
    queries_file = os.path.join(INPUT_DIR, SYNTHETIC_QUERIES_JSON)
    with open(queries_file, "r") as f:
        synthetic_queries = json.load(f)
    with open(results_file, "r") as f:
        synthetic_results = json.load(f)
    queries_results = []
    for query in synthetic_queries:
        queries_results.append((query, synthetic_results[query["query_id"]]))
    logger.info(
        f"Loaded {len(synthetic_results)} synthetic results, {len(synthetic_queries)} synthetic queries"
    )
    return queries_results


def generate_single_query(
    df: pd.DataFrame, query_type: str, count: int
) -> Optional[Dict[str, Any]]:
    try:
        row = df.sample(1).iloc[0]
        item = row["product_class"]

        query_generators = {
            "simple": generate_simple_query,
            "negation": generate_negation_query,
            "number": generate_number_query,
            "complex": generate_complex_query,
        }

        if query_type not in query_generators:
            logger.warning(f"Unknown query type: {query_type}")
            return None

        query = query_generators[query_type](row, item)

        query["query_id"] = f"{query_type}-{count:03d}"

        if item in df["product_class"].values:
            query["query_params"]["product_class"] = item

        cleanup_query_params(query)
        return query

    except Exception as e:
        logger.warning(f"Failed to generate {query_type} query {count}: {e}")
        return None


def fix_query_ids(
    queries: List[Dict[str, Any]], results: Dict[str, List[int]], max_count: int = 100
) -> Tuple[List[Dict[str, Any]], Dict[str, List[int]]]:

    by_type = defaultdict(list)
    for query in queries:
        query_type = query["query_id"].split("-")[0]
        by_type[query_type].append(query)

    new_queries = []
    new_results = {}

    for query_type, query_list in by_type.items():
        query_list.sort(key=lambda x: int(x["query_id"].split("-")[1]))
        query_list = query_list[:max_count]

        for i, query in enumerate(query_list, 1):
            old_id = query["query_id"]
            new_id = f"{query_type}-{i:03d}"

            query["query_id"] = new_id
            new_queries.append(query)

            if old_id in results:
                new_results[new_id] = results[old_id]

    return new_queries, new_results


def save_queries_and_results(
    queries_with_results: List[Tuple[Dict[str, Any], List[int]]],
) -> None:
    logger.info("\nSaving queries and results...")

    queries = [query for query, _ in queries_with_results]

    results = {
        query["query_id"]: result_ids for query, result_ids in queries_with_results
    }

    # Fix query IDs to be sequential 001-N for each type
    logger.info("Fixing query IDs to be sequential...")
    fixed_queries, fixed_results = fix_query_ids(queries, results)

    serializable_queries = convert_to_serializable(fixed_queries)

    queries_path = os.path.join(OUTPUT_DIR, QUERIES_TEMP_JSON)
    with open(queries_path, "w") as f:
        json.dump(serializable_queries, f, indent=2)

    results_path = os.path.join(OUTPUT_DIR, SEARCH_RESULTS_TEMP_JSON)
    with open(results_path, "w") as f:
        json.dump(fixed_results, f, indent=2)

    logger.info(f"Queries saved to {queries_path}")
    logger.info(f"Search results saved to {results_path}")


def generate_and_validate_query(
    df: pd.DataFrame, query_type: str, count: int, max_attempts: int = MAX_ATTEMPTS
) -> Optional[Tuple[Dict[str, Any], List[int]]]:
    for attempt in range(max_attempts):
        try:
            query = generate_single_query(df, query_type, count)
            if not query:
                continue

            query_text = query["query_params"]["product_description"]
            query_params = query.get("query_params", {})
            query_params = {k: v for k, v in query_params.items() if v}
            query["query_params"] = query_params

            filters = create_filters(query)
            results = qdrant_search(
                QDRANT_CLIENT,
                query_text,
                top_k=DEFAULT_TOP_K,
                filters=filters,
                query_params=query_params,
            )
            result_ids = [r.get("product_id") for r in results]
            score = sum(float(x.get("score", 0.0)) for x in results) / float(
                len(results)
            )
            logger.info(f"Query '{query_text}' score: {score}")
            if len(result_ids) >= MIN_RESULTS:
                logger.info(f"Query '{query_text}' returned {len(result_ids)} results")
                return (query, result_ids)
            else:
                logger.info(
                    f"Query '{query_text}' returned {len(result_ids)} results - retrying"
                )

        except Exception as e:
            logger.warning(f"Error generating/testing {query_type} query {count}: {e}")
            continue

    logger.warning(
        f"Failed to generate valid {query_type} query {count} after {max_attempts} attempts"
    )
    return None


def generate_queries_with_validation(
    df: pd.DataFrame, query_type: str, target_count: int
) -> List[Tuple[Dict[str, Any], List[int]]]:
    valid_queries = []
    current_count = 1
    if target_count >= 1:
        logger.info(f"Generating {target_count} {query_type} query with validation...")
    else:
        return []

    while len(valid_queries) < target_count:
        result = generate_and_validate_query(df, query_type, current_count)

        if result:
            query, result_ids = result
            valid_queries.append((query, result_ids))
            logger.info(
                f"Generated {len(valid_queries)}/{target_count} valid {query_type} queries"
            )

            if len(valid_queries) >= target_count:
                break
        else:
            logger.warning(
                f"Failed to generate valid {query_type} query {current_count}"
            )

        current_count += 1

        if current_count > target_count * MAX_ATTEMPTS:
            logger.warning(
                f"Stopping {query_type} query generation after {current_count} attempts (target: {target_count})"
            )
            break

    success_rate = (len(valid_queries) / target_count) * 100
    logger.info(
        f"Generated {len(valid_queries)}/{target_count} valid {query_type} queries ({success_rate:.1f}% success rate)"
    )
    return valid_queries


def main() -> None:
    logger.info("Starting query generation pipeline (Qdrant)...")
    logger.info("=" * 70)

    initialize_qdrant()

    dataset = load_dataset()
    df = create_query_generation_dataframe(dataset)
    save_query_generation_data(df)

    all_queries_with_results: List[Tuple[Dict[str, Any], List[int]]] = []

    logger.info("Generating new queries with validation (Qdrant)...")

    query_configs = [
        ("simple", DEFAULT_SIMPLE_QUERIES),
        ("negation", DEFAULT_NEGATION_QUERIES),
        ("number", DEFAULT_NUMBER_QUERIES),
        ("complex", DEFAULT_COMPLEX_QUERIES),
    ]

    for query_type, target_count in query_configs:
        valid_queries = generate_queries_with_validation(df, query_type, target_count)
        all_queries_with_results.extend(valid_queries)

    logger.info(f"Generated {len(all_queries_with_results)} new valid queries")

    synthetic_queries_results = load_synthetic_queries_results()

    all_queries_with_results.extend(synthetic_queries_results)

    logger.info("Fixing query ids...")
    save_queries_and_results(all_queries_with_results)

    logger.info("\nQuery Generation Summary:")
    logger.info("-" * 50)
    logger.info(f"Product classes processed: {len(df)}")
    logger.info(f"Total valid queries: {len(all_queries_with_results)}")

    expected_counts = {
        "simple": DEFAULT_SIMPLE_QUERIES,
        "negation": DEFAULT_NEGATION_QUERIES,
        "number": DEFAULT_NUMBER_QUERIES,
        "complex": DEFAULT_COMPLEX_QUERIES,
    }

    query_types: Dict[str, int] = {}
    for query, _ in all_queries_with_results:
        query_type = query.get("query_type", "unknown")
        query_types[query_type] = query_types.get(query_type, 0) + 1

    logger.info("\nQuery Type Breakdown:")
    for query_type, count in query_types.items():
        expected = expected_counts.get(query_type, 0)
        if expected > 0:
            logger.info(
                f"  {query_type.capitalize()}: {count}/{expected} ({count/expected*100:.1f}%)"
            )
        else:
            logger.info(f"  {query_type.capitalize()}: {count}")

    missing_queries = []
    for query_type, expected in expected_counts.items():
        actual = query_types.get(query_type, 0)
        if actual < expected:
            missing = expected - actual
            missing_queries.append(f"{missing} {query_type}")

    if missing_queries:
        logger.warning(f"Missing queries: {', '.join(missing_queries)}")
    else:
        logger.info("All target query counts achieved!")

    logger.info("\n" + "=" * 70)
    logger.info("Query generation pipeline completed")


if __name__ == "__main__":
    main()
