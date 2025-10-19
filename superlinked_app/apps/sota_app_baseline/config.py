import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

from shared_config import (GCS_BUCKET_NAME, GCS_BUCKET_PATH,
                           GCS_BUCKET_VERSION, GCS_PRODUCTS_BLOB,
                           GCS_QUERIES_BLOB, GCS_RANKED_RESULTS_BLOB)
from superlinked_app.util.enum import QueryMode

# Get the directory of this config file
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(CONFIG_DIR)))
)

load_dotenv(os.path.join(CONFIG_DIR, ".env"))

open_ai_api_key: str = os.getenv("OPEN_AI_API_KEY")
redis_password: str = os.getenv("REDIS_PASSWORD")


class Configuration(BaseSettings):
    # algo params
    reingest: bool = True
    redo_nlq: bool = False
    query_mode: QueryMode = QueryMode.USE_FULL_QUERY_TEXT
    text_embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    dataset_id: str = GCS_BUCKET_VERSION
    benchmark_run_id: str = f"{dataset_id}_sota_app_{query_mode.value}"
    # colname constants
    float_cols: list[str] = ["rating_count", "average_rating", "price"]
    query_type_colname: str = "query_type"
    query_params_colname: str = "query_params"
    query_text_colname: str = "query_text"
    # guides what columns to join for full-text search
    join_cols: list[str] | None = [
        "product_description",
        "product_class",
        "material",
        "style",
        "color",
        "rating_count",
        "average_rating",
        "price",
    ]
    # paths
    bucket_name: str = GCS_BUCKET_NAME
    product_dataset_path: str = GCS_PRODUCTS_BLOB
    query_dataset_path: str = GCS_QUERIES_BLOB
    ground_truth_input_path: str = GCS_RANKED_RESULTS_BLOB
    unique_categories_path: str = (
        f"{GCS_BUCKET_PATH}{dataset_id}/unique_categories.json"
    )
    default_output_path: str = (
        f"{GCS_BUCKET_PATH}outputs/{dataset_id}/{benchmark_run_id}"
    )
    query_latency_output_path: str = f"{default_output_path}/query_latencies.json"
    error_output_path: str = f"{default_output_path}/errors/"
    search_param_output_path: str = f"{default_output_path}/search_params.pkl"
    eval_result_output_path: str = f"{default_output_path}/evaluation_results.pkl"
    eval_ground_truth_output_path: str = (
        f"{default_output_path}/evaluation_ground_truth.pkl"
    )
    final_result_path: str = f"{default_output_path}/result_rankings.txt"
    # redis
    redis_url: str = "redis-10619.c41690.eu-central2-3.gcp.cloud.rlrcp.com"
    redis_port: int = 10619
    redis_username: str = "default"


config = Configuration()
