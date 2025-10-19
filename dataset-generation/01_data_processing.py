import json
import logging
import os
import random
import warnings
from typing import Any, Dict

import pandas as pd
from config import (BATCH_SIZE, CLEANED_PRODUCTS_JSONL, EMBEDDINGS_FILE,
                    INPUT_DIR, NUM_WORKERS, OUTPUT_DIR,
                    PRODUCT_CLASS_MAPPING_JSONL, RANDOM_SEED, RF_N_ESTIMATORS,
                    RF_RANDOM_STATE, SAMPLE_SIZE, UNIQUE_CATEGORIES_JSON)
from data_util import (load_raw_products, load_synthetic_products,
                       save_product_data)
from emb_util import create_embeddings
from model_manager import model_manager
from price_mapping import add_prices_to_dataframe
from qdrant_util import load_embeddings_from_file, setup_qdrant, start_qdrant
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

warnings.filterwarnings("ignore")


def load_normalization_map_from_jsonl(file_path: str) -> Dict[str, Any]:
    """Load normalization map from JSONL file.

    Args:
        file_path: Path to the JSONL file.

    Returns:
        Dictionary mapping raw values to normalized values.
    """
    normalization_map = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entry = json.loads(line.strip())
                raw_value = entry["raw"]
                normalized_value = entry["normalized"]
                normalization_map[raw_value] = normalized_value
    return normalization_map


def get_normalization_maps():
    """Get all normalization mapping dictionaries from JSONL files."""
    base_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "input"
    )

    return {
        "material": load_normalization_map_from_jsonl(
            os.path.join(base_path, "material_normalization.jsonl")
        ),
        "country": load_normalization_map_from_jsonl(
            os.path.join(base_path, "country_normalization.jsonl")
        ),
        "color": load_normalization_map_from_jsonl(
            os.path.join(base_path, "color_normalization.jsonl")
        ),
        "style": load_normalization_map_from_jsonl(
            os.path.join(base_path, "style_normalization.jsonl")
        ),
    }


def get_valid_values():
    """Get sets of valid normalized values for each category from JSONL files."""
    normalization_maps = get_normalization_maps()

    # Extract unique normalized values, handling both strings and lists
    valid_values = {}
    for category, mapping in normalization_maps.items():
        values = set()
        for normalized_value in mapping.values():
            if isinstance(normalized_value, list):
                values.update(normalized_value)
            else:
                values.add(normalized_value)
        valid_values[category] = values

    return valid_values


def fill_product_description(products: pd.DataFrame) -> pd.DataFrame:
    """Fill product description with product name, product class, color, style, material, and country of origin.

    Args:
        products: DataFrame to fill.

    Returns:
        DataFrame with filled product description.
    """
    logger.debug("Cleaning data...")

    cleaned_products = products.copy()

    cleaned_products["color_str"] = cleaned_products["color"].apply(
        lambda x: ", ".join(x) if isinstance(x, list) else str(x)
    )
    cleaned_products["style_str"] = cleaned_products["style"].apply(
        lambda x: ", ".join(x) if isinstance(x, list) else str(x)
    )
    cleaned_products["material_str"] = cleaned_products["material"].apply(
        lambda x: ", ".join(x) if isinstance(x, list) else str(x)
    )
    cleaned_products["countryoforigin_str"] = cleaned_products["countryoforigin"].apply(
        lambda x: ", ".join(x) if isinstance(x, list) else str(x)
    )

    cleaned_products["product_description"] = cleaned_products[
        "product_description"
    ].fillna(
        (
            cleaned_products["product_name"]
            + ", "
            + cleaned_products["product_class"]
            + ", "
            + cleaned_products["color_str"]
            + ", "
            + cleaned_products["style_str"]
            + ", "
            + cleaned_products["material_str"]
            + ", "
            + cleaned_products["countryoforigin_str"]
        )
    )

    cleaned_products = cleaned_products.drop(
        columns=["color_str", "style_str", "material_str", "countryoforigin_str"]
    )

    return cleaned_products


def process_features(features_str: Any) -> dict:
    """Process features string into a dictionary.

    Args:
        features_str: String of features.

    Returns:
        Dictionary of features.
    """
    if pd.isna(features_str):
        return {}

    features_list = str(features_str).split("|")
    features_dict = {}

    for item in features_list:
        if ":" in item:
            key, value = item.split(":", 1)
            features_dict[key.strip()] = value.strip()
        elif " : " in item:
            key, value = item.split(" : ", 1)
            features_dict[key.strip()] = value.strip()

    material_keys = [k for k in features_dict.keys() if "material" in k.lower()]
    style_keys = [k for k in features_dict.keys() if "style" in k.lower()]
    color_keys = [k for k in features_dict.keys() if "color" in k.lower()]
    country_keys = [k for k in features_dict.keys() if "country" in k.lower()]

    result = {}
    for key_list, prefix in [
        (material_keys, "material"),
        (style_keys, "style"),
        (color_keys, "color"),
        (country_keys, "countryoforigin"),
    ]:
        if key_list:
            result[prefix] = [features_dict[k] for k in key_list]

    return result


def filter_products_with_features(cleaned_products: pd.DataFrame) -> pd.DataFrame:
    """Filter products to only include those with all required features.

    Args:
        cleaned_products: Raw product DataFrame.

    Returns:
        pd.DataFrame: Filtered products with complete features.
    """
    logger.info("Filtering products with complete features...")

    processed_rows = []
    for idx, row in cleaned_products.iterrows():
        features = process_features(row["product_features"])
        if all(
            key in features for key in ["material", "style", "color", "countryoforigin"]
        ):
            processed_rows.append(
                {
                    "product_id": row["product_id"],
                    "product_name": row["product_name"],
                    "product_description": row["product_description"],
                    "product_class": row["product_class"],
                    "material": features["material"],
                    "style": sorted(features["style"]),
                    "color": sorted(features["color"]),
                    "rating_count": row["rating_count"],
                    "average_rating": row["average_rating"],
                    "countryoforigin": features["countryoforigin"],
                }
            )

    filtered_products = pd.DataFrame(processed_rows)
    logger.info(f"Original dataset size: {len(cleaned_products)}")
    logger.info(f"Filtered dataset size: {len(filtered_products)}")

    return filtered_products


def normalize_and_filter_values(
    values: Any, normalization_map: dict, valid_values: set
) -> list:
    """Normalize and filter values using a mapping dictionary.

    Args:
        values: Values to normalize.
        normalization_map: Mapping dictionary for normalization.
        valid_values: Set of valid values to filter by.

    Returns:
        list: Normalized and filtered values.
    """
    if isinstance(values, str):
        values = [values]

    normalized = []
    for value in values:
        raw = value.lower().strip()
        norm = normalization_map.get(raw, raw)

        if isinstance(norm, list):
            for n in norm:
                if n in valid_values and n not in normalized:
                    normalized.append(n)
        else:
            if norm in valid_values and norm not in normalized:
                normalized.append(norm)

    return normalized


def normalize_features(filtered_products: pd.DataFrame) -> pd.DataFrame:
    """Normalize all features using the mapping dictionaries.

    Args:
        filtered_products: DataFrame to normalize.
    """
    logger.debug("Normalizing features...")

    normalization_maps = get_normalization_maps()
    valid_values = get_valid_values()
    normalized_products = filtered_products.copy()

    normalized_products["material"] = normalized_products["material"].apply(
        lambda materials: normalize_and_filter_values(
            materials, normalization_maps["material"], valid_values["material"]
        )
    )

    normalized_products["countryoforigin"] = normalized_products[
        "countryoforigin"
    ].apply(
        lambda countries: normalize_and_filter_values(
            countries, normalization_maps["country"], valid_values["country"]
        )
    )

    normalized_products["color"] = normalized_products["color"].apply(
        lambda colors: normalize_and_filter_values(
            colors, normalization_maps["color"], valid_values["color"]
        )
    )

    normalized_products["style"] = normalized_products["style"].apply(
        lambda styles: normalize_and_filter_values(
            styles, normalization_maps["style"], valid_values["style"]
        )
    )

    logger.info("Feature normalization completed")

    return normalized_products


def predict_missing_product_classes(normalized_products: pd.DataFrame) -> pd.DataFrame:
    """Use machine learning to predict missing product classes.

    Args:
        normalized_products: DataFrame with potentially missing product classes.
    Returns:
        pd.DataFrame: DataFrame with predicted product classes.
    """
    logger.debug("Predicting missing product classes...")

    to_be_filled_products = normalized_products.copy()

    to_be_filled_products["text"] = (
        to_be_filled_products["product_name"].fillna("")
        + " "
        + to_be_filled_products["product_description"].fillna("")
    )

    train_df = to_be_filled_products[
        to_be_filled_products["product_class"].notna()
    ].copy()
    test_df = to_be_filled_products[
        to_be_filled_products["product_class"].isna()
    ].copy()

    if len(test_df) == 0:
        logger.info("No missing product classes to predict")

        filled_products = to_be_filled_products.drop(columns=["text"])
        return filled_products

    le = LabelEncoder()
    y_train = le.fit_transform(train_df["product_class"])

    # Use model manager for consistent model loading
    model = model_manager.get_qwen_model()  # Using QWEN model for consistency

    X_train = model.encode(
        train_df["text"].tolist(), show_progress_bar=False, batch_size=128
    )
    X_test = model.encode(
        test_df["text"].tolist(), show_progress_bar=False, batch_size=128
    )

    clf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS, random_state=RF_RANDOM_STATE
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    test_df["product_class"] = le.inverse_transform(y_pred)

    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    filled_products = combined_df.sort_index()
    filled_products = filled_products.drop(columns=["text"])

    logger.info(f"Predicted {len(test_df)} missing product classes")
    return filled_products


def normalize_product_classes(filled_products: pd.DataFrame) -> pd.DataFrame:
    """Normalize product classes using the mapping dictionaries.

    Args:
        filled_products: DataFrame to normalize.

    Returns:
        DataFrame with normalized product classes.
    """
    logger.info("Normalizing product classes...")
    product_class_mapping = pd.read_json(
        os.path.join(INPUT_DIR, PRODUCT_CLASS_MAPPING_JSONL), lines=True
    )

    class_mapping_dict = dict(
        zip(
            product_class_mapping["product_class"],
            product_class_mapping["product_class_fixed"],
        )
    )

    normalized_products = filled_products.copy()
    normalized_products["product_class"] = (
        normalized_products["product_class"]
        .map(class_mapping_dict)
        .fillna(normalized_products["product_class"])
    )

    logger.info("Product classes normalized")
    return normalized_products


def final_cleaning(filled_products: pd.DataFrame) -> pd.DataFrame:
    """Final cleaning steps.

    Args:
        filled_products: DataFrame to clean.

    Returns:
        DataFrame with final cleaning.
    """
    logger.debug("Performing final cleaning...")

    full_final_products = filled_products.copy()

    avg_rating_mean = full_final_products["average_rating"].mean()
    full_final_products["average_rating"] = full_final_products[
        "average_rating"
    ].fillna(avg_rating_mean)
    rating_count_median = full_final_products["rating_count"].median()
    full_final_products["rating_count"] = full_final_products["rating_count"].fillna(
        rating_count_median
    )

    logger.info("Final cleaning completed")

    return full_final_products


def fix_numeric_values(final_full_products: pd.DataFrame) -> pd.DataFrame:
    """Fix numeric values in the DataFrame using realistic distributions.

    Args:
        final_full_products: DataFrame to fix.
    """
    logger.debug("Fixing numeric values...")

    final_full_products["average_rating"] = final_full_products["average_rating"].apply(
        lambda x: (
            round(random.triangular(low=3.0, high=5.0, mode=4.2), 1) if x < 3.0 else x
        )
    )

    final_full_products["rating_count"] = final_full_products["rating_count"].apply(
        lambda x: (
            int(random.triangular(low=100, high=10000, mode=1000))
            if x < 100
            else int(x)
        )
    )

    logger.info("Numeric values fixed with realistic distributions")
    return final_full_products


def filter_complete_products(final_full_products: pd.DataFrame) -> pd.DataFrame:
    """Filter to only include products with all required features.

    Args:
        final_full_products: DataFrame to filter.

    Returns:
        pd.DataFrame: Filtered DataFrame with complete products.
    """
    logger.debug("Filtering to complete products...")

    final_filtered_products = final_full_products.copy()

    mask = (
        (final_filtered_products["material"].apply(lambda x: len(x) > 0))
        & (final_filtered_products["countryoforigin"].apply(lambda x: len(x) > 0))
        & (final_filtered_products["color"].apply(lambda x: len(x) > 0))
        & (final_filtered_products["style"].apply(lambda x: len(x) > 0))
    )

    final = final_filtered_products[mask]

    logger.info(f"Final dataset size: {len(final)}")
    return final


def get_unique_categories(products: pd.DataFrame) -> Dict[str, set]:
    """Get column values from the input directory.

    Args:
        products: DataFrame to get column values from.

    Returns:
        Dict[str, set]: Dictionary of column values.
    """
    column_values = {}
    for col in ["color", "style", "material", "product_class"]:
        values = products[col].explode().dropna().unique()
        column_values[col] = sorted({str(v).strip() for v in values if str(v).strip()})
    with open(os.path.join(OUTPUT_DIR, UNIQUE_CATEGORIES_JSON), "w") as f:
        json.dump(column_values, f, indent=4)
    return column_values


def main() -> None:
    """Main data processing pipeline."""
    logger.info("Starting data processing pipeline...")
    logger.info("=" * 50)

    logger.info("Loading data...")
    products = load_raw_products()

    logger.info("Filtering products with features...")
    filtered_products = filter_products_with_features(products)

    logger.info("Normalizing features...")
    normalized_products = normalize_features(filtered_products)

    logger.info("Filling product description...")
    cleaned_products = fill_product_description(normalized_products)

    logger.info("Predicting missing product classes...")
    filled_products = predict_missing_product_classes(cleaned_products)

    logger.info("Final cleaning...")
    final_full_products = final_cleaning(filled_products)

    logger.info("Filtering to complete products...")
    final_filtered_products = filter_complete_products(final_full_products)

    logger.info("Fixing numeric values...")
    final_filtered_products = fix_numeric_values(final_filtered_products)

    final_filtered_products = final_filtered_products.dropna(
        subset=["product_description"]
    )

    logger.info("Getting column values...")
    get_unique_categories(final_filtered_products)

    logger.info("Creating sample for output...")
    final_sample = final_filtered_products.sample(
        min(SAMPLE_SIZE, len(final_filtered_products)), random_state=RANDOM_SEED
    ).sort_values(by="product_id")

    logger.info("Loading synthetic products...")
    synthetic_products = load_synthetic_products()

    logger.info("Combining with final sample...")
    final_dataset = pd.concat([final_sample, synthetic_products], ignore_index=True)

    logger.info("Resetting product_id for combined dataset...")
    final_dataset.loc[:, "product_id"] = range(1, len(final_dataset) + 1)
    final_dataset["product_id"] = final_dataset["product_id"].apply(
        lambda x: f"product_{int(x)}"
    )

    logger.info(
        f"Combined {len(synthetic_products)} synthetic products with {len(final_sample)} original products"
    )

    logger.info("Normalizing product classes...")
    final_dataset = normalize_product_classes(final_dataset)

    logger.info("Adding prices to final dataset...")
    final_dataset = add_prices_to_dataframe(final_dataset)

    logger.info("Saving final data...")
    save_product_data(final_dataset)

    logger.info("\nFinal Data Analysis:")
    logger.info("\n" + "=" * 50)
    logger.info(f"Total products in final dataset: {len(final_dataset)}")
    logger.info(f"Columns: {list(final_dataset.columns)}")

    null_check = final_dataset.isnull().sum()
    if null_check.sum() == 0:
        logger.info("No null values in final dataset")
    else:
        logger.warning("Remaining null values:")
        logger.warning(null_check[null_check > 0])

    logger.info("\n" + "=" * 50)
    logger.info("Data processing pipeline completed successfully!")
    logger.info("\n" + "=" * 50)

    try:
        logger.info("Creating embeddings for Qdrant...")
        create_embeddings(
            products_file=os.path.join(OUTPUT_DIR, CLEANED_PRODUCTS_JSONL),
            output_file=os.path.join(OUTPUT_DIR, EMBEDDINGS_FILE),
            num_workers=NUM_WORKERS,
            batch_size=BATCH_SIZE,
        )
        logger.info("Embeddings created")
    except Exception as e:
        logger.warning(f"Failed to create embeddings: {e}")

    try:
        logger.info("Starting Qdrant and uploading embeddings...")
        if not start_qdrant():
            logger.warning("Qdrant did not start; skipping Qdrant setup")
        else:
            embeddings = load_embeddings_from_file()
            setup_qdrant(embeddings)
            logger.info("Qdrant is ready with uploaded embeddings")
    except Exception as e:
        logger.warning(f"Failed to initialize Qdrant: {e}")


if __name__ == "__main__":
    main()
