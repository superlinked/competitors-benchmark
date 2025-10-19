#!/usr/bin/env python3
"""
Shared Configuration Base

This module provides common configuration settings and utilities
used across all search implementations (Azure, Hybrid-search, Vertex AI).
"""

import json
import logging
import os
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()

import pandas as pd

# ============================================================================
# PROJECT CONFIGURATION
# ============================================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "input")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "output")

# ============================================================================
# GOOGLE CLOUD STORAGE CONFIGURATION
# ============================================================================

GCS_BUCKET_NAME = "superlinked-benchmarks-internal"
GCS_BUCKET_PATH = "wayfair_wands_datasets/"
GCS_BUCKET_VERSION = "20250917"  # Centralized version control
GCS_BUCKET_FULL_PATH = f"{GCS_BUCKET_PATH}{GCS_BUCKET_VERSION}/"

GCS_QUERIES_BLOB = f"{GCS_BUCKET_FULL_PATH.rstrip('/')}/queries.json"
GCS_PRODUCTS_BLOB = f"{GCS_BUCKET_FULL_PATH.rstrip('/')}/products.jsonl"
GCS_RANKED_RESULTS_BLOB = f"{GCS_BUCKET_FULL_PATH.rstrip('/')}/ranked_results.json"

# ============================================================================
# COMMON FILE PATHS
# ============================================================================

# Input files
QUERIES_FILE = os.path.join(OUTPUT_DIR, "queries.json")
PRODUCTS_FILE = os.path.join(OUTPUT_DIR, "products.jsonl")

# Output files
RESULTS_FILE = "output/results.json"
NLQ_PARAMS_QUERIES = "output/nlq_params_queries.json"

# ============================================================================
# COMMON MODELS
# ============================================================================

GEMINI_MODEL_NAME = "gemini-2.0-flash-001"

# ============================================================================
# COMMON RANKING CONFIGURATION
# ============================================================================

PROJECT_ID = "data-359211"
LOCATION = "global"
RANKING_MODEL = "semantic-ranker-default@latest"
RANKING_TOP_K = 10

# ============================================================================
# PROMPT GENERATION UTILITIES
# ============================================================================


def load_products_for_prompt() -> Dict[str, List[str]]:
    """
    Load products data for prompt generation.

    Returns:
        Dictionary with color, material, style, and product_class lists

    Raises:
        Exception: If loading products fails
    """
    try:
        # Check if the products file exists
        if not os.path.exists(PRODUCTS_FILE):
            logging.warning(f"Products file not found: {PRODUCTS_FILE}")
            return {"colors": [], "materials": [], "styles": [], "product_classes": []}

        products_data = []
        with open(PRODUCTS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    products_data.append(json.loads(line))

        products = pd.DataFrame(products_data)

        return {
            "colors": list(
                set([color for colors in products["color"] for color in colors])
            ),
            "materials": list(
                set(
                    [
                        material
                        for materials in products["material"]
                        for material in materials
                    ]
                )
            ),
            "styles": list(
                set([style for styles in products["style"] for style in styles])
            ),
            "product_classes": products["product_class"].unique().tolist(),
        }
    except Exception as e:
        logging.warning(f"Failed to load products for prompt: {e}")
        return {"colors": [], "materials": [], "styles": [], "product_classes": []}


def create_extraction_prompt() -> str:
    """
    Create the entity extraction prompt template.

    Returns:
        Formatted prompt string
    """
    product_data = load_products_for_prompt()

    return f"""
You are a document entity extraction specialist. Given a document (query). 
Your task is to extract the text value of the following entities: 
- color (string)
- material (string) 
- style (string)
- color_negated (string)
- product_class (string)
- material_negated (string)
- style_negated (string)
- price_max (float)
- rating_min (float) 
- rating_count_min (float)
- product_class (string)
- product_description (string)

Put the product description in the product_description field. Everything except numeric constraints should be in the product_description field.

Here is a list of possible colors:
{product_data['colors']}
Here is a list of possible materials:
{product_data['materials']}
Here is a list of possible styles:
{product_data['styles']}
Here is a list of possible product classes:
{product_data['product_classes']}
Some entities may be mentioned, some not.

Examples:

query: "white not rustic style wool ottomans for living rooms, more than 1000 ratings"
output:
{{
    "color": "white",
    "material": "wool",
    "style_negated": "rustic",
    "rating_count_min": 1000,
    "product_class": "ottomans",
    "product_description": "white wool ottomans for living rooms"
}}

Please extract entities from this query: {{query_text}}

Return the result as a JSON object with the following structure:
{{
    "color": "",
    "material": "",
    "style": "",
    "color_negated": "",
    "material_negated": "",
    "style_negated": "",
    "price_max": 0.0,
    "rating_min": 0.0,
    "rating_count_min": 0.0,
    "product_class": "",
    "product_description": ""
}}
Return only filled features
"""


# Generate the prompt lazily when needed
def get_extraction_prompt() -> str:
    """
    Get the entity extraction prompt template.

    Returns:
        Formatted prompt string
    """
    return create_extraction_prompt()


# For backward compatibility, keep PROMPT as a function that can be called
# This avoids calling create_extraction_prompt() at module import time
PROMPT = get_extraction_prompt


# ============================================================================
# API KEYS
# ============================================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
