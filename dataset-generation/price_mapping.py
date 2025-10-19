#!/usr/bin/env python3
"""
Price Mapping and Generation

This module handles price mapping from product classes and generates realistic
prices based on materials and styles using feature weights from JSONL files.
"""
import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from config import RANDOM_SEED

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "input")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "output")

DEFAULT_PRICE_MAPPING_FILE = os.path.join(INPUT_DIR, "price_mapping.jsonl")
DEFAULT_FEATURE_WEIGHTS_FILE = os.path.join(INPUT_DIR, "feature_weights.jsonl")

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def load_feature_weights(
    weights_file: str = DEFAULT_FEATURE_WEIGHTS_FILE,
) -> Dict[str, Dict[str, float]]:
    """Load feature weights from JSONL file.

    Args:
        weights_file: Path to the feature weights JSONL file.

    Returns:
        Dictionary with material, style, country weights, and rating config.
    """

    material_weights = {}
    style_weights = {}
    country_weights = {}
    rating_config = {}

    with open(weights_file, "r") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                feature_type = data["feature_type"]

                if feature_type == "material":
                    weight = data["weight"]
                    items = data["items"]
                    item_weights = {item.lower(): weight for item in items}
                    material_weights.update(item_weights)

                elif feature_type == "style":
                    weight = data["weight"]
                    items = data["items"]
                    item_weights = {item.lower(): weight for item in items}
                    style_weights.update(item_weights)

                elif feature_type == "country":
                    weight = data["weight"]
                    items = data["items"]
                    item_weights = {item.lower(): weight for item in items}
                    country_weights.update(item_weights)

                elif feature_type == "rating_config":
                    rating_config.update(data)
                    if "rating_thresholds" in rating_config:
                        thresholds = rating_config["rating_thresholds"]
                        rating_config["rating_thresholds"] = {
                            k: tuple(v) if isinstance(v, list) else v
                            for k, v in thresholds.items()
                        }

    return {
        "material": material_weights,
        "style": style_weights,
        "country": country_weights,
        "rating_config": rating_config,
    }


def load_price_mappings(jsonl_file: str = DEFAULT_PRICE_MAPPING_FILE) -> Dict:
    """Load price mappings from JSONL file.

    Args:
        jsonl_file: Path to the price mapping JSONL file.

    Returns:
        Dictionary mapping product classes to price ranges.
    """
    if not os.path.exists(jsonl_file):
        raise FileNotFoundError(f"Price mapping file not found: {jsonl_file}")

    mappings = {}
    with open(jsonl_file, "r") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                mappings[data["product_class"]] = {
                    "min": data["price_min"],
                    "max": data["price_max"],
                }
    return mappings


def get_rating_multiplier(
    rating: Optional[float],
    rating_config: Optional[Dict] = None,
) -> float:
    """Calculate price multiplier based on product rating and review count.

    Args:
        rating: Product rating (1.0-5.0).
        rating_count: Number of ratings.
        rating_config: Rating configuration dictionary.

    Returns:
        Price multiplier based on rating.
    """
    if rating is None or pd.isna(rating):
        return 1.0

    if rating_config is None:
        rating_config = load_feature_weights()["rating_config"]

    # Clamp rating to valid range
    min_rating = rating_config["min_rating"]
    max_rating = rating_config["max_rating"]
    rating = max(min_rating, min(rating, max_rating))

    # Calculate rating-based multiplier
    rating_mult = 1.0
    thresholds = rating_config.get("rating_thresholds", {})

    for threshold_name, (min_rating, max_rating, multiplier) in thresholds.items():
        if min_rating <= rating <= max_rating:
            rating_mult = multiplier
            break

    # Combine rating and review effects
    combined_mult = rating_mult

    # Apply bounds to prevent extreme values
    return max(0.8, min(combined_mult, 1.4))


def get_rating_confidence(
    rating: Optional[float],
    rating_count: Optional[int] = None,
) -> float:
    """Calculate confidence in the rating based on number of reviews.

    Args:
        rating: Product rating.
        rating_count: Number of ratings.

    Returns:
        Confidence score (0.0-1.0) indicating how reliable the rating is.
    """
    if rating is None or pd.isna(rating):
        return 0.0

    if rating_count is None or pd.isna(rating_count):
        return 0.5  # Default confidence for unknown rating count

    # More reviews = higher confidence
    rating_count = max(1, rating_count)
    confidence = min(rating_count / 100, 1.0)  # Saturation at 100 reviews

    return confidence


def get_material_multiplier(
    materials: List[str], feature_weights: Optional[Dict[str, Dict[str, float]]] = None
) -> float:
    """Calculate price multiplier based on materials.

    Args:
        materials: List of materials.
        feature_weights: Feature weights dictionary (loaded automatically if None).

    Returns:
        Price multiplier for materials.
    """
    if not materials:
        return 1.0

    if feature_weights is None:
        feature_weights = load_feature_weights()

    material_weights = feature_weights["material"]

    # Calculate the average multiplier for the materials
    material_sum = 0.0
    valid_materials = 0
    for material in materials:
        material_lower = material.lower()
        if material_lower in material_weights:
            material_sum += material_weights[material_lower]
            valid_materials += 1

    return material_sum / max(valid_materials, 1) + 0.1


def get_style_multiplier(
    styles: List[str], feature_weights: Optional[Dict[str, Dict[str, float]]] = None
) -> float:
    """Calculate price multiplier based on styles.

    Args:
        styles: List of styles.
        feature_weights: Feature weights dictionary (loaded automatically if None).

    Returns:
        Price multiplier for styles.
    """
    if not styles:
        return 1.0

    if feature_weights is None:
        feature_weights = load_feature_weights()

    style_weights = feature_weights["style"]

    # Calculate the average multiplier for the styles
    style_sum = 0.0
    valid_styles = 0
    for style in styles:
        style_lower = style.lower()
        if style_lower in style_weights:
            style_sum += style_weights[style_lower]
            valid_styles += 1

    return style_sum / max(valid_styles, 1) + 0.1


def get_country_multiplier(
    countries: List[str], feature_weights: Optional[Dict[str, Dict[str, float]]] = None
) -> float:
    """Calculate price multiplier based on country of origin.

    Args:
        countries: List of countries.
        feature_weights: Feature weights dictionary (loaded automatically if None).

    Returns:
        Price multiplier for countries.
    """
    if not countries:
        return 1.0

    if feature_weights is None:
        feature_weights = load_feature_weights()

    country_weights = feature_weights["country"]

    # Calculate the average multiplier for the countries
    country_sum = 0.0
    valid_countries = 0
    for country in countries:
        country_lower = country.lower()
        if country_lower in country_weights:
            country_sum += country_weights[country_lower]
            valid_countries += 1

    return country_sum / max(valid_countries, 1) + 0.1


def round_price(price: float) -> float:
    """Round price to realistic price points.

    Args:
        price: Raw price to round.

    Returns:
        Rounded price.
    """
    if price < 100:
        return round(price / 5) * 5
    elif price < 1000:
        return round(price / 25) * 25
    else:
        return round(price / 100) * 100


def generate_price(min_price: float, max_price: float) -> float:
    np.random.seed(RANDOM_SEED)
    """Generate base price using normal distribution.

    Args:
        min_price: Minimum price.
        max_price: Maximum price.

    Returns:
        Generated price within the range.
    """
    mu = (min_price + max_price) / 2
    sigma = (max_price - min_price) / 6
    generated_price = np.random.normal(mu, sigma)
    return generated_price


def generate_realistic_price(
    product_class: str,
    materials: Optional[List[str]] = None,
    styles: Optional[List[str]] = None,
    countries: Optional[List[str]] = None,
    rating: Optional[float] = None,
    rating_count: Optional[int] = None,
    price_mappings: Optional[Dict] = None,
    feature_weights: Optional[Dict[str, Dict[str, float]]] = None,
    use_rating: bool = True,
) -> Optional[float]:
    """Generate realistic price based on class, materials, styles, country, and rating.

    Args:
        product_class: Product category.
        materials: List of materials.
        styles: List of styles.
        countries: List of countries of origin.
        rating: Product rating (1.0-5.0).
        rating_count: Number of ratings.
        price_mappings: Price mappings dictionary (loaded automatically if None).
        feature_weights: Feature weights dictionary (loaded automatically if None).
        use_rating: Whether to include rating in price calculation.

    Returns:
        Generated price or None if product_class not found.
    """
    if price_mappings is None:
        price_mappings = load_price_mappings()

    if product_class not in price_mappings:
        print(f"Product class {product_class} not found in price mappings")
        return None

    materials = materials or []
    styles = styles or []
    countries = countries or []

    # Get base price range
    base_min = price_mappings[product_class]["min"]
    base_max = price_mappings[product_class]["max"]

    # Calculate multipliers
    material_mult = get_material_multiplier(materials, feature_weights)
    style_mult = get_style_multiplier(styles, feature_weights)
    country_mult = get_country_multiplier(countries, feature_weights)

    # Calculate rating multiplier if enabled
    rating_mult = 1.0
    if use_rating and feature_weights:
        rating_config = feature_weights.get("rating_config")
        rating_mult = get_rating_multiplier(rating, rating_config)

    # Combined multiplier with reasonable bounds
    combined_mult = min(material_mult * style_mult * country_mult * rating_mult, 3.5)
    combined_mult = max(combined_mult, 0.3)

    # Apply multipliers to the range
    adjusted_min = base_min * combined_mult
    adjusted_max = base_max * combined_mult

    # Generate price using normal distribution
    price = generate_price(adjusted_min, adjusted_max)

    # Round to realistic price points
    return abs(round_price(price))


def add_prices_to_dataframe(
    df: pd.DataFrame,
    price_mappings: Optional[Dict] = None,
    feature_weights: Optional[Dict[str, Dict[str, float]]] = None,
    price_column: str = "price",
    use_rating: bool = True,
    rating_column: str = "average_rating",
    rating_count_column: str = "rating_count",
) -> pd.DataFrame:
    """Add price columns to DataFrame with optional rating-based pricing.

    Args:
        df: DataFrame with product data.
        price_mappings: Price mappings dictionary (loaded automatically if None).
        feature_weights: Feature weights dictionary (loaded automatically if None).
        price_column: Name for the final price column.
        use_rating: Whether to include rating in price calculation.
        rating_column: Name of the rating column.
        rating_count_column: Name of the rating count column.

    Returns:
        DataFrame with added price columns.
    """
    logger.info("Adding prices to DataFrame...")

    if price_mappings is None:
        price_mappings = load_price_mappings()

    if feature_weights is None:
        feature_weights = load_feature_weights()

    # Log rating configuration if using ratings
    if use_rating and "rating_config" in feature_weights:
        rating_config = feature_weights["rating_config"]
        logger.info(
            f"Using rating configuration: weight={rating_config.get('rating_weight', 0.15)}"
        )

    # Create a copy to avoid modifying original
    result_df = df.copy()

    # Initialize new columns
    result_df[price_column] = None

    # Track statistics
    processed_count = 0
    missing_class_count = 0
    rating_stats = {"with_rating": 0, "without_rating": 0}

    for idx, row in result_df.iterrows():
        product_class = row["product_class"]

        # Check if product_class exists in mappings
        if product_class not in price_mappings:
            logger.warning(f"Product class {product_class} not found in price mappings")
            missing_class_count += 1
            continue

        # Extract rating data
        rating = row.get(rating_column)
        rating_count = row.get(rating_count_column)

        # Track rating usage
        if rating is not None and not pd.isna(rating):
            rating_stats["with_rating"] += 1
        else:
            rating_stats["without_rating"] += 1

        # Use the existing generate_realistic_price function
        price = generate_realistic_price(
            product_class=product_class,
            materials=row.get("material", []),
            styles=row.get("style", []),
            countries=row.get("country", []),
            rating=rating,
            rating_count=rating_count,
            price_mappings=price_mappings,
            feature_weights=feature_weights,
            use_rating=use_rating,
        )

        result_df.at[idx, price_column] = price
        processed_count += 1

    logger.info(f"Processed {processed_count} products with prices")
    if missing_class_count > 0:
        logger.warning(
            f"Skipped {missing_class_count} products with missing/invalid product classes"
        )

    if use_rating:
        logger.info(
            f"Rating stats: {rating_stats['with_rating']} with rating, {rating_stats['without_rating']} without rating"
        )

    return result_df
