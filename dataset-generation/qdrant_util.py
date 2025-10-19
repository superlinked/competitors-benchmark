import json
import logging
import os
import subprocess
import time

import numpy as np
from config import (BATCH_SIZE, COLLECTION_NAME, EMBEDDINGS_FILE, OUTPUT_DIR,
                    QDRANT_HOST, QDRANT_PORT, QDRANT_TOP_K, VECTOR_SIZE)
from model_manager import model_manager
from qdrant_client import QdrantClient, models
from qdrant_client.models import (Distance, PointStruct, SparseVectorParams,
                                  VectorParams)

logger = logging.getLogger(__name__)


def start_qdrant():
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=qdrant", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
        )
        if "qdrant" in result.stdout:
            logger.info("Qdrant already running")
            return True

        logger.info("Starting Qdrant...")
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                "qdrant",
                "-p",
                "6333:6333",
                "qdrant/qdrant",
            ],
            check=True,
        )
        time.sleep(3)
        return True
    except Exception as e:
        logger.error(f"Failed to start Qdrant: {e}")
        return False


def load_embeddings_from_file():
    embeddings_file_path = os.path.join(OUTPUT_DIR, EMBEDDINGS_FILE)
    embeddings = []
    with open(embeddings_file_path, "r") as f:
        for line in f:
            embeddings.append(json.loads(line.strip()))
    logger.info(f"Loaded {len(embeddings)} embeddings")
    return embeddings


def setup_qdrant(embeddings):
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60.0)

    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        },
        sparse_vectors_config={"sparse": SparseVectorParams()},
    )

    splade_model, qwen_model = model_manager.get_models()
    points = []
    for i, emb in enumerate(embeddings):
        point = PointStruct(
            id=i,
            vector={
                "dense": emb["dense_embedding"],
                "sparse": {
                    "indices": emb["sparse_embedding"]["indices"],
                    "values": emb["sparse_embedding"]["values"],
                },
            },
            payload={
                "product_id": emb["id"],
                "product_name": emb["metadata"].get("product_name", ""),
                "product_description": emb["metadata"].get("product_description", ""),
                "product_class": emb["metadata"].get("product_class", ""),
                "color": emb["metadata"].get("color", []),
                "material": emb["metadata"].get("material", []),
                "style": emb["metadata"].get("style", []),
                "price": emb["metadata"].get("price", 0.0),
                "average_rating": emb["metadata"].get("average_rating", 0.0),
                "rating_count": emb["metadata"].get("rating_count", 0.0),
            },
        )
        points.append(point)

    batch_size = BATCH_SIZE
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        try:
            client.upsert(collection_name=COLLECTION_NAME, points=batch)
            logger.info(
                f"Uploaded batch {i//batch_size + 1}/{(len(points) + batch_size - 1)//batch_size}"
            )
        except Exception as e:
            logger.error(f"Failed to upload batch {i//batch_size + 1}: {e}")
            smaller_batch_size = 50
            for j in range(0, len(batch), smaller_batch_size):
                smaller_batch = batch[j : j + smaller_batch_size]
                try:
                    client.upsert(collection_name=COLLECTION_NAME, points=smaller_batch)
                except Exception as e2:
                    logger.error(f"Failed to upload smaller batch: {e2}")
                    raise

    logger.info(f"Added {len(points)} products to Qdrant")
    return client


def create_query_text(query_text: str, query_params: dict = None) -> str:
    if not query_params:
        return query_text

    parts = [query_text]

    parts.append(f"product_class: {query_params['product_class'].upper()}")

    return " | ".join(filter(None, parts))


def get_query_vector(query_text: str, query_params: dict = None):
    processed_query_text = create_query_text(query_text, query_params)

    splade_model, qwen_model = model_manager.get_models()

    qwen_dense = qwen_model.encode([processed_query_text])[0].tolist()

    splade_emb = splade_model.encode([processed_query_text])[0]
    if hasattr(splade_emb, "to_dense"):
        dense_tensor = splade_emb.to_dense()
        sparse_array = dense_tensor.cpu().numpy()
    else:
        sparse_array = np.array(splade_emb)

    nonzero_mask = sparse_array != 0
    indices = np.where(nonzero_mask)[0].tolist()
    values = sparse_array[nonzero_mask].tolist()

    return {"dense": qwen_dense, "sparse": {"indices": indices, "values": values}}


def search(client, query_text, top_k=QDRANT_TOP_K, filters=None, query_params=None):
    query_vector = get_query_vector(query_text, query_params)
    if query_vector is None:
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


def create_filters(query_data):
    conditions_must = []
    conditions_must_not = []
    conditions_should = []
    query_params = query_data.get("query_params", {})

    def to_field_condition_array(key, value):
        if isinstance(value, list):
            return models.FieldCondition(key=key, match=models.MatchAny(any=value))
        if key in ["material", "color", "style"]:
            return models.FieldCondition(key=key, match=models.MatchAny(any=[value]))
        return models.FieldCondition(key=key, match=models.MatchValue(value=value))

    if "color" in query_params:
        conditions_must.append(to_field_condition_array("color", query_params["color"]))
    if "material" in query_params:
        conditions_must.append(
            to_field_condition_array("material", query_params["material"])
        )
    if "style" in query_params:
        conditions_must.append(to_field_condition_array("style", query_params["style"]))

    if "price_max" in query_params:
        conditions_must.append(
            models.FieldCondition(
                key="price", range=models.Range(lte=query_params["price_max"])
            )
        )
    if "rating_min" in query_params:
        conditions_must.append(
            models.FieldCondition(
                key="average_rating", range=models.Range(gte=query_params["rating_min"])
            )
        )
    if "rating_count_min" in query_params:
        conditions_must.append(
            models.FieldCondition(
                key="rating_count",
                range=models.Range(gte=query_params["rating_count_min"]),
            )
        )

    if "color_negated" in query_params:
        conditions_must_not.append(
            to_field_condition_array("color", query_params["color_negated"])
        )
    if "material_negated" in query_params:
        conditions_must_not.append(
            to_field_condition_array("material", query_params["material_negated"])
        )
    if "style_negated" in query_params:
        conditions_must_not.append(
            to_field_condition_array("style", query_params["style_negated"])
        )

    if "product_class" in query_params:
        conditions_should.append(
            models.FieldCondition(
                key="product_class",
                match=models.MatchText(text=query_params["product_class"]),
            )
        )

    # if "product_class" in query_params:
    #     conditions_must.append(models.FieldCondition(key="product_class", match=models.MatchValue(value=query_params["product_class"])))

    if conditions_must or conditions_must_not:
        return models.Filter(
            must=conditions_must or None,
            must_not=conditions_must_not or None,
            should=conditions_should or None,
        )

    return None
