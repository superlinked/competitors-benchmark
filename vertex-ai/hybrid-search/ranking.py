#!/usr/bin/env python3
"""
Hybrid Search with Qdrant

This module implements hybrid search using Qdrant vector database with both
sparse (SPLADE) and dense (Gemini) embeddings for enhanced search relevance.
"""

import json
import logging
import os
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from config import (BATCH_SIZE, COLLECTION_NAME, EMBEDDINGS_FILE,
                    NLQ_PARAMS_QUERIES, QDRANT_HOST, QDRANT_PORT, RESULTS_FILE,
                    TOP_K, VECTOR_SIZE)
from create_embs import (get_gemini_embeddings_batch,
                         get_splade_embeddings_batch, init_models)
from qdrant_client import QdrantClient, models
from qdrant_client.models import (Distance, PointStruct, SparseVectorParams,
                                  VectorParams)

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# Initialize models globally
splade_model, gemini_model = init_models()


def start_qdrant() -> bool:
    """
    Start Qdrant Docker container.

    Returns:
        True if successful, False otherwise
    """
    try:
        container_name = "qdrant-hybrid-search"

        # Check if container is already running
        result = subprocess.run(
            [
                "docker",
                "ps",
                "--filter",
                f"name={container_name}",
                "--format",
                "{{.Names}}",
            ],
            capture_output=True,
            text=True,
        )

        if container_name in result.stdout.split():
            logger.info("Qdrant already running")
            return True

        logger.info("Starting Qdrant...")
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                container_name,
                "-p",
                "6334:6333",
                "qdrant/qdrant",
            ],
            check=True,
        )

        # Wait for Qdrant to start
        time.sleep(3)
        logger.info("Qdrant started successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to start Qdrant: {e}")
        return False


def stop_qdrant() -> None:
    """
    Stop and remove Qdrant Docker container.
    """
    try:
        container_name = "qdrant-hybrid-search"
        subprocess.run(["docker", "stop", container_name], check=True)
        subprocess.run(["docker", "rm", container_name], check=True)
        logger.info("Qdrant stopped and removed")
    except Exception as e:
        logger.warning(f"Failed to stop Qdrant: {e}")


def load_embeddings_from_file() -> List[Dict[str, Any]]:
    """
    Load embeddings from JSONL file.

    Returns:
        List of embedding dictionaries

    Raises:
        Exception: If loading fails
    """
    try:
        embeddings = []
        with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    embedding = json.loads(line)
                    embeddings.append(embedding)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON on line {line_num}: {e}")
                    continue

        logger.info(f"Loaded {len(embeddings)} embeddings from {EMBEDDINGS_FILE}")
        return embeddings

    except Exception as e:
        logger.error(f"Failed to load embeddings: {e}")
        raise


def load_queries_from_file() -> List[Dict[str, Any]]:
    """
    Load queries from JSON file.

    Returns:
        List of query dictionaries

    Raises:
        Exception: If loading fails
    """
    try:
        with open(NLQ_PARAMS_QUERIES, "r", encoding="utf-8") as f:
            queries = json.load(f)

        logger.info(f"Loaded {len(queries)} queries from {NLQ_PARAMS_QUERIES}")
        return queries

    except Exception as e:
        logger.error(f"Failed to load queries: {e}")
        raise


def load_data() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load both embeddings and queries.

    Returns:
        Tuple of (embeddings, queries)
    """
    try:
        embeddings = load_embeddings_from_file()
        queries = load_queries_from_file()

        logger.info(f"Loaded {len(embeddings)} embeddings and {len(queries)} queries")
        return embeddings, queries

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise


def create_qdrant_client() -> QdrantClient:
    """
    Create Qdrant client connection.

    Returns:
        Configured QdrantClient instance

    Raises:
        Exception: If connection fails
    """
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60.0)
        logger.info(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
        return client

    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        raise


def setup_qdrant_collection(client: QdrantClient) -> None:
    """
    Setup Qdrant collection with proper vector configuration.

    Args:
        client: Qdrant client instance
    """
    try:
        # Delete existing collection if it exists
        try:
            client.delete_collection(COLLECTION_NAME)
            logger.info(f"Deleted existing collection: {COLLECTION_NAME}")
        except Exception:
            pass  # Collection might not exist

        # Create new collection
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            },
            sparse_vectors_config={"sparse": SparseVectorParams()},
        )

        logger.info(f"Created collection: {COLLECTION_NAME}")

    except Exception as e:
        logger.error(f"Failed to setup Qdrant collection: {e}")
        raise


def create_point_from_embedding(embedding: Dict[str, Any], index: int) -> PointStruct:
    """
    Create Qdrant point from embedding data.

    Args:
        embedding: Embedding dictionary
        index: Point index

    Returns:
        PointStruct for Qdrant
    """
    try:
        if "metadata" in embedding:
            # New format with metadata
            point = PointStruct(
                id=index,
                vector={
                    "dense": embedding["embedding"],
                    "sparse": {
                        "indices": embedding["sparse_embedding"]["indices"],
                        "values": embedding["sparse_embedding"]["values"],
                    },
                },
                payload={
                    "product_id": embedding["id"],
                    "product_name": embedding["metadata"].get("product_name", ""),
                    "product_description": embedding["metadata"].get(
                        "product_description", ""
                    ),
                    "product_class": embedding["metadata"].get("product_class", ""),
                    "color": embedding["metadata"].get("color", []),
                    "material": embedding["metadata"].get("material", []),
                    "style": embedding["metadata"].get("style", []),
                    "price": embedding["metadata"].get("price", 0.0),
                    "average_rating": embedding["metadata"].get("average_rating", 0.0),
                    "rating_count": embedding["metadata"].get("rating_count", 0.0),
                },
            )
        else:
            # Legacy format
            point = PointStruct(
                id=index,
                vector={
                    "dense": embedding["embedding"],
                    "sparse": {
                        "indices": embedding["sparse_embedding"]["dimensions"],
                        "values": embedding["sparse_embedding"]["values"],
                    },
                },
                payload={
                    "product_id": embedding["id"],
                    "color": embedding.get("color", []),
                    "material": embedding.get("material", []),
                    "style": embedding.get("style", []),
                    "price": embedding.get("price", 0.0),
                    "average_rating": embedding.get("average_rating", 0.0),
                    "rating_count": embedding.get("rating_count", 0.0),
                },
            )

        return point

    except Exception as e:
        logger.error(f"Failed to create point from embedding {index}: {e}")
        raise


def upload_embeddings_to_qdrant(
    client: QdrantClient, embeddings: List[Dict[str, Any]]
) -> None:
    """
    Upload embeddings to Qdrant in batches.

    Args:
        client: Qdrant client instance
        embeddings: List of embedding dictionaries
    """
    try:
        points = []
        for i, emb in enumerate(embeddings):
            point = create_point_from_embedding(emb, i)
            points.append(point)

        # Upload in batches
        total_batches = (len(points) + BATCH_SIZE - 1) // BATCH_SIZE
        logger.info(f"Uploading {len(points)} points in {total_batches} batches")

        for i in range(0, len(points), BATCH_SIZE):
            batch = points[i : i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1

            try:
                client.upsert(collection_name=COLLECTION_NAME, points=batch)
                logger.info(f"Uploaded batch {batch_num}/{total_batches}")
            except Exception as e:
                logger.error(f"Failed to upload batch {batch_num}: {e}")
                # Try with smaller batch size
                smaller_batch_size = 50
                for j in range(0, len(batch), smaller_batch_size):
                    smaller_batch = batch[j : j + smaller_batch_size]
                    try:
                        client.upsert(
                            collection_name=COLLECTION_NAME, points=smaller_batch
                        )
                        logger.info(
                            f"Uploaded smaller batch {j//smaller_batch_size + 1}"
                        )
                    except Exception as e2:
                        logger.error(f"Failed to upload smaller batch: {e2}")
                        raise

        logger.info(f"Successfully uploaded {len(points)} products to Qdrant")

    except Exception as e:
        logger.error(f"Failed to upload embeddings to Qdrant: {e}")
        raise


def setup_qdrant(embeddings: List[Dict[str, Any]]) -> QdrantClient:
    """
    Complete Qdrant setup with collection and data upload.

    Args:
        embeddings: List of embedding dictionaries

    Returns:
        Configured QdrantClient instance
    """
    try:
        client = create_qdrant_client()
        setup_qdrant_collection(client)
        upload_embeddings_to_qdrant(client, embeddings)

        logger.info("Qdrant setup completed successfully")
        return client

    except Exception as e:
        logger.error(f"Failed to setup Qdrant: {e}")
        raise


def create_query_text(
    query_text: str, query_params: Optional[Dict[str, Any]] = None
) -> str:
    """
    Create enhanced query text with parameters.

    Args:
        query_text: Original query text
        query_params: Optional query parameters

    Returns:
        Enhanced query text
    """
    if not query_params:
        return query_text

    parts = [query_text]

    if "product_class" in query_params and query_params["product_class"]:
        parts.append(f"product_class: {query_params['product_class'].upper()}")

    return " | ".join(filter(None, parts))


def extract_sparse_vector(sparse_vec: Any) -> Tuple[List[int], List[float]]:
    """
    Extract sparse vector indices and values from various formats.

    Args:
        sparse_vec: Sparse vector in various formats

    Returns:
        Tuple of (indices, values)
    """
    indices = []
    values = []

    try:
        if (
            isinstance(sparse_vec, dict)
            and "indices" in sparse_vec
            and "values" in sparse_vec
        ):
            indices = list(sparse_vec["indices"])
            values = list(sparse_vec["values"])
        elif hasattr(sparse_vec, "to_dense"):
            dense = sparse_vec.to_dense().cpu().numpy()
            nonzero_mask = dense != 0
            indices = np.where(nonzero_mask)[0].tolist()
            values = dense[nonzero_mask].tolist()
        elif hasattr(sparse_vec, "indices") and hasattr(sparse_vec, "values"):
            idx = sparse_vec.indices()
            val = sparse_vec.values()
            if hasattr(idx, "cpu"):
                idx = idx.cpu().numpy()
                val = val.cpu().numpy()
            if getattr(idx, "ndim", 1) > 1:
                idx = idx[0]
            indices = idx.tolist()
            values = val.tolist()
        else:
            dense = (
                sparse_vec
                if isinstance(sparse_vec, np.ndarray)
                else np.array(sparse_vec)
            )
            nonzero_mask = dense != 0
            indices = np.where(nonzero_mask)[0].tolist()
            values = dense[nonzero_mask].tolist()

    except Exception as e:
        logger.warning(f"Failed to extract SPLADE sparse vector: {e}")
        indices = []
        values = []

    return indices, values


def get_query_vector(
    query_text: str, query_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate query vector with both dense and sparse embeddings.

    Args:
        query_text: Query text
        query_params: Optional query parameters

    Returns:
        Dictionary with dense and sparse vectors
    """
    try:
        processed_query_text = create_query_text(query_text, query_params)

        # Get dense embedding
        try:
            gemini_dense = get_gemini_embeddings_batch(
                [processed_query_text], gemini_model, VECTOR_SIZE
            )[0]
        except Exception as e:
            logger.warning(f"Failed to get Gemini embedding: {e}")
            gemini_dense = [0.0] * VECTOR_SIZE

        # Get sparse embedding
        splade_vec = get_splade_embeddings_batch([processed_query_text], splade_model)[
            0
        ]
        indices, values = extract_sparse_vector(splade_vec)

        return {"dense": gemini_dense, "sparse": {"indices": indices, "values": values}}

    except Exception as e:
        logger.error(f"Failed to generate query vector: {e}")
        return {"dense": [0.0] * VECTOR_SIZE, "sparse": {"indices": [], "values": []}}


def create_filters(query_data: Dict[str, Any]) -> Optional[models.Filter]:
    """
    Create Qdrant filters from query parameters.

    Args:
        query_data: Query data with parameters

    Returns:
        Qdrant filter or None
    """
    try:
        conditions_must = []
        conditions_must_not = []
        conditions_should = []
        query_params = query_data.get("query_params", {})

        def to_field_condition_array(key: str, value: Any) -> models.FieldCondition:
            """Convert parameter to field condition."""
            if isinstance(value, list):
                return models.FieldCondition(key=key, match=models.MatchAny(any=value))
            if key in ["material", "color", "style"]:
                return models.FieldCondition(
                    key=key, match=models.MatchAny(any=[value])
                )
            return models.FieldCondition(key=key, match=models.MatchValue(value=value))

        # Positive filters
        if "color" in query_params and query_params["color"]:
            conditions_must.append(
                to_field_condition_array("color", query_params["color"])
            )
        if "material" in query_params and query_params["material"]:
            conditions_must.append(
                to_field_condition_array("material", query_params["material"])
            )
        if "style" in query_params and query_params["style"]:
            conditions_must.append(
                to_field_condition_array("style", query_params["style"])
            )

        # Numeric filters
        if "price_max" in query_params and query_params["price_max"]:
            conditions_must.append(
                models.FieldCondition(
                    key="price", range=models.Range(lte=query_params["price_max"])
                )
            )
        if "rating_min" in query_params and query_params["rating_min"]:
            conditions_must.append(
                models.FieldCondition(
                    key="average_rating",
                    range=models.Range(gte=query_params["rating_min"]),
                )
            )
        if "rating_count_min" in query_params and query_params["rating_count_min"]:
            conditions_must.append(
                models.FieldCondition(
                    key="rating_count",
                    range=models.Range(gte=query_params["rating_count_min"]),
                )
            )

        # Negation filters
        if "color_negated" in query_params and query_params["color_negated"]:
            conditions_must_not.append(
                to_field_condition_array("color", query_params["color_negated"])
            )
        if "material_negated" in query_params and query_params["material_negated"]:
            conditions_must_not.append(
                to_field_condition_array("material", query_params["material_negated"])
            )
        if "style_negated" in query_params and query_params["style_negated"]:
            conditions_must_not.append(
                to_field_condition_array("style", query_params["style_negated"])
            )

        if conditions_must or conditions_must_not:
            return models.Filter(
                must=conditions_must or None,
                must_not=conditions_must_not or None,
                should=conditions_should or None,
            )

        return None

    except Exception as e:
        logger.error(f"Failed to create filters: {e}")
        return None


def search_products(
    client: QdrantClient,
    query_text: str,
    top_k: int = TOP_K,
    filters: Optional[models.Filter] = None,
    query_params: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Search products using hybrid search.

    Args:
        client: Qdrant client
        query_text: Search query text
        top_k: Number of results to return
        filters: Optional Qdrant filters
        query_params: Optional query parameters

    Returns:
        List of search results
    """
    try:
        query_vector = get_query_vector(query_text, query_params)
        if not query_vector:
            return []

        base_multiplier = 6
        prefetch_n_sparse = top_k * base_multiplier
        prefetch_n_dense = top_k * base_multiplier

        prefetch = [
            models.Prefetch(
                query=models.SparseVector(**query_vector.get("sparse")),
                using="sparse",
                limit=prefetch_n_sparse,
                filter=filters,
            ),
            models.Prefetch(
                query=query_vector.get("dense"),
                using="dense",
                limit=prefetch_n_dense,
                filter=filters,
            ),
        ]
        query = models.FusionQuery(fusion=models.Fusion.DBSF)

        results = client.query_points(
            COLLECTION_NAME,
            prefetch=prefetch,
            query=query,
            with_payload=True,
            limit=top_k * 6,
        )

        formatted_results = []
        for point in results.points:
            formatted_results.append(
                {
                    "product_id": point.payload["product_id"],
                    "score": point.score,
                    "product_name": point.payload.get("product_name", ""),
                    "product_description": point.payload.get("product_description", ""),
                    "color": point.payload.get("color", []),
                    "product_class": point.payload.get("product_class", ""),
                    "material": point.payload.get("material", []),
                    "style": point.payload.get("style", []),
                    "price": point.payload.get("price", 0.0),
                    "average_rating": point.payload.get("average_rating", 0.0),
                    "rating_count": point.payload.get("rating_count", 0.0),
                }
            )

        return formatted_results[:top_k]

    except Exception as e:
        logger.error(f"Search failed for query '{query_text}': {e}")
        return []


def save_search_results(search_results: Dict[str, List[str]]) -> None:
    """
    Save search results to output file.

    Args:
        search_results: Dictionary mapping query IDs to product ID lists
    """
    try:
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(search_results, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Saved results for {len(search_results)} queries to {RESULTS_FILE}"
        )

    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        raise


def main():
    try:
        logger.info("Starting Qdrant hybrid search...")

        # Start Qdrant
        if not start_qdrant():
            logger.error("Failed to start Qdrant")
            return

        # Load data
        embeddings, queries = load_data()

        # Setup Qdrant
        client = setup_qdrant(embeddings)

        # Process queries
        search_results = {}

        for i, query_data in enumerate(queries):
            query_text = query_data["query_params"].get("product_description", "")
            query_id = query_data["query_id"]
            query_params = query_data.get("query_params", {})

            logger.info(f"Processing query {i+1}/{len(queries)}: {query_id}")
            logger.info(f"Query text: '{query_text}'")

            try:
                filters = create_filters(query_data)
                results = search_products(
                    client, query_text, filters=filters, query_params=query_params
                )

                product_ids = [result["product_id"] for result in results]
                search_results[query_id] = product_ids

                logger.info(f"Found {len(product_ids)} products for query {query_id}")

            except Exception as e:
                logger.error(f"Search failed for query {query_id}: {e}")
                search_results[query_id] = []

        # Save results
        save_search_results(search_results)

        logger.info(f"Search completed for {len(queries)} queries")

        # Display results summary
        logger.info("Results summary:")
        for query_id, product_ids in search_results.items():
            logger.info(f"{query_id}: {len(product_ids)} products")

    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        raise
    finally:
        # Clean up Qdrant
        try:
            stop_qdrant()
        except Exception as e:
            logger.warning(f"Failed to stop Qdrant: {e}")


if __name__ == "__main__":
    main()
