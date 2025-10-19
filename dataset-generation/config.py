#!/usr/bin/env python3
"""
Dataset Generation Configuration

This module provides dataset generation-specific configuration settings
that extend the shared base configuration.
"""

import os
import sys

# Add parent directory to path to import shared config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared_config import *

# ============================================================================
# QUERY GENERATION CONFIGURATION
# ============================================================================

MAX_ATTEMPTS = 10
MIN_RESULTS = 15
MIN_SCORE = 1.1
N_PRODUCT_CLASSES = 50
DEFAULT_TOP_K = 20
# Default query counts by type
DEFAULT_SIMPLE_QUERIES = 5
DEFAULT_NEGATION_QUERIES = 5
DEFAULT_NUMBER_QUERIES = 5
DEFAULT_COMPLEX_QUERIES = 5

# Query generation timeouts and retries
QDRANT_TIMEOUT = 1.0
QUERY_VALIDATION_TIMEOUT = 30.0

# Statistical distribution parameters for query generation
RATING_BETA_PARAMS = (1, 3)
RATING_COUNT_BETA_PARAMS = (1, 3)
PRICE_BETA_PARAMS = (3, 1)
DEFAULT_STAT_VALUE = 2

# ============================================================================
# DATA PROCESSING CONFIGURATION
# ============================================================================

# Sample size for final dataset
SAMPLE_SIZE = 10000

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# DATASET GENERATION-SPECIFIC FILE NAMES
# ============================================================================

# Get the directory of this config file (dataset-generation directory)
DATASET_GEN_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(DATASET_GEN_DIR)
DATA_INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "input")
DATA_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "output")

# Input files
RAW_PRODUCT_CSV = os.path.join(DATA_INPUT_DIR, "product.csv")
SYNTHETIC_PRODUCTS_JSON = os.path.join(DATA_INPUT_DIR, "synthetic_products.json")
SYNTHETIC_QUERIES_JSON = os.path.join(DATA_INPUT_DIR, "synthetic_queries.json")
SYNTHETIC_RESULTS_JSON = os.path.join(DATA_INPUT_DIR, "synthetic_results.json")
FEATURE_WEIGHTS_JSONL = os.path.join(DATA_INPUT_DIR, "feature_weights.jsonl")
PRICE_MAPPING_JSONL = os.path.join(DATA_INPUT_DIR, "price_mapping.jsonl")
UNIQUE_CATEGORIES_JSON = os.path.join(DATA_OUTPUT_DIR, "unique_categories.json")
PRODUCT_CLASS_DESCRIPTIONS_JSON = os.path.join(
    DATA_INPUT_DIR, "product_class_descriptions.json"
)
PRODUCT_CLASS_MAPPING_JSONL = os.path.join(
    DATA_INPUT_DIR, "product_class_mapping.jsonl"
)

# Output files
FEATURES_JSON = os.path.join(DATA_OUTPUT_DIR, "features.json")
CLEANED_PRODUCTS_CSV = os.path.join(DATA_OUTPUT_DIR, "products.csv")
CLEANED_PRODUCTS_JSONL = os.path.join(DATA_OUTPUT_DIR, "products.jsonl")
QUERIES_JSON = os.path.join(DATA_OUTPUT_DIR, "queries.json")
QUERIES_TEMP_JSON = os.path.join(DATA_OUTPUT_DIR, "queries_temp.json")
QUERY_GENERATION_JSONL = os.path.join(DATA_OUTPUT_DIR, "query_generation.jsonl")
SEARCH_RESULTS_JSON = os.path.join(DATA_OUTPUT_DIR, "search_results.json")
SEARCH_RESULTS_TEMP_JSON = os.path.join(DATA_OUTPUT_DIR, "search_results_temp.json")
RANKED_RESULTS_JSON = os.path.join(DATA_OUTPUT_DIR, "ranked_results.json")
RANKED_RESULTS_TEMP_JSON = os.path.join(DATA_OUTPUT_DIR, "ranked_results_temp.json")
SEARCH_RESULTS_WITH_ATTRIBUTES_JSON = os.path.join(
    DATA_OUTPUT_DIR, "search_results_with_attributes.json"
)
RANKED_RESULTS_WITH_ATTRIBUTES_JSON = os.path.join(
    DATA_OUTPUT_DIR, "ranked_results_with_attributes.json"
)
EMPTY_QUERY_IDS_JSON = os.path.join(DATA_OUTPUT_DIR, "empty_query_ids.json")
NLQ_PARAMS_QUERIES = os.path.join(DATA_OUTPUT_DIR, "nlq_params_queries.json")

# ============================================================================
# EMBEDDING CONFIGURATION
# ============================================================================

# Text processing
MAX_TEXT_LENGTH = 256
DEVICE = "mps"
DEVICE_SPLADE = "cpu"
# Sentence transformer model for classification
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
SPLADE_MODEL = "naver/splade-v3"
QWEN_EMB_MODEL = "Qwen/Qwen3-Embedding-0.6B"
NUM_WORKERS = 10
BATCH_SIZE = 200
VECTOR_SIZE = 1024  # Dimension size for dense embeddings
MODEL_CACHE_DIR = os.path.join(
    DATASET_GEN_DIR, "model_cache"
)  # Cache directory for models

# Vertex AI / Gemini embeddings
GEMINI_MODEL = "gemini-2.5-flash"

# Embeddings/artifacts output
EMBEDDINGS_FILE = os.path.join(DATA_OUTPUT_DIR, "product_embs.json")

# Qdrant configuration
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "gt_pipeline"
QDRANT_TOP_K = 20

# ============================================================================
# MACHINE LEARNING CONFIGURATION FOR PRODUCT CLASS PREDICTION
# ============================================================================

# Random Forest parameters for product class prediction
RF_N_ESTIMATORS = 100
RF_RANDOM_STATE = 42

# ============================================================================
# LLM RETRY CONFIGURATION FOR JSON PARSING IN FEATURE EXTRACTION
# ============================================================================

# Retry configuration for LLM JSON parsing failures
FEATURE_EXTRACTION_MAX_RETRIES = 3
FEATURE_EXTRACTION_RETRY_DELAY = 1.0  # Initial delay in seconds
FEATURE_EXTRACTION_RETRY_BACKOFF_MULTIPLIER = 2.0  # Exponential backoff multiplier
FEATURE_EXTRACTION_WORKERS = 5
MAX_DESCRIPTIONS = 10

FEATURE_EXTRACTION_PROMPT_TEMPLATE = """You are a search optimization expert. Extract 3-8 high-quality search features from these {product_class} product descriptions that customers commonly search for.

Product Descriptions:
{descriptions}

TASK: Identify features that customers would type into a search box when looking for {product_class_lower}. Focus on terms that help distinguish between different products and improve search relevance. 

FUNCTIONAL_FEATURES: Practical capabilities customers search for
- Examples: "adjustable", "removable", "foldable", "stackable", "reclining"
- Focus: What the product can DO, not what it's made of

USAGE_CONTEXTS: Where/how the product is used  
- Examples: "living room", "office", "outdoor", "bedroom", "kitchen"
- Format: Will be added as "for [context]" to queries

SIZE_DESCRIPTORS: Size-related terms customers search for
- Examples: "compact", "oversized", "space-saving", "large", "small"
- Format: Will be added to the beginning of queries

EMOTIONAL_DESCRIPTORS: Aesthetic qualities customers search for
- Examples: "elegant", "cozy", "luxurious"
- Format: Will be added to the beginning of queries

CONSTRUCTION_FEATURES: Quality/build characteristics
- Examples: "handcrafted", "premium", "durable", "sturdy", "solid"
- Format: Will be added to the beginning of queries

BENEFIT_PHRASES: Customer benefits and conveniences
- Examples: "easy to clean", "long lasting", "maintenance free", "comfortable"
- Format: Will be added to the beginning of queries

QUALITY CRITERIA:
- Must appear in multiple product descriptions (popularity)
- Must help customers distinguish between products (relevance)  
- Must be terms customers would actually search for (searchability)
- Must be single words or short phrases (2-4 words max)
- Must NOT be in the excluded list below

EXCLUDED TERMS (do not extract these): {column_values}

Before you respond, think about the quality criteria and the excluded terms. like if feature + product_class makes sense.
OUTPUT FORMAT: Return ONLY a valid JSON dictionary. Do NOT wrap in markdown code blocks (```). Do NOT include any explanations, comments, or extra text. Your response will be parsed using json.loads() so it must be valid JSON.
{{
    "FUNCTIONAL_FEATURES": ["term1", "term2"],
    "USAGE_CONTEXTS": ["context1", "context2"], 
    "SIZE_DESCRIPTORS": ["size1", "size2"],
    "EMOTIONAL_DESCRIPTORS": ["emotion1", "emotion2"],
    "CONSTRUCTION_FEATURES": ["construction1", "construction2"],
    "BENEFIT_PHRASES": ["benefit1", "benefit2"]
}}"""
