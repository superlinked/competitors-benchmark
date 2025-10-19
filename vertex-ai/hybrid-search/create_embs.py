#!/usr/bin/env python3
"""
Create SPLADE sparse embeddings and Gemini dense embeddings for products
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import numpy as np
import torch
from config import (BATCH_SIZE, DEVICE_SPLADE, EMBEDDINGS_FILE,
                    GCS_BUCKET_NAME, GCS_PRODUCTS_BLOB, GEMINI_MODEL,
                    SPLADE_MODEL, VECTOR_SIZE)
from google import genai
from google.cloud import storage
from google.genai.types import EmbedContentConfig
from sentence_transformers import SparseEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


def load_products_from_bucket() -> Dict[str, Dict]:
    """Load products from bucket.

    Returns:
        Dict[str, Dict]: Products.
    """
    try:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(GCS_PRODUCTS_BLOB)

        text = blob.download_as_text(encoding="utf-8")

        products = {}
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                product = json.loads(line)
                product_id = product.get("product_id")
                if product_id:
                    products[str(product_id)] = product
            except Exception:
                continue

        return products
    except Exception as e:
        print(f"Error loading products from bucket: {e}")
        return {}


def init_models():
    """Initialize models.

    Returns:
        Tuple[SparseEncoder, genai.Client]: Models.
    """
    client = genai.Client()
    return SparseEncoder(SPLADE_MODEL, device=DEVICE_SPLADE), client


def create_product_text(product: Dict[str, Any]) -> str:
    """Create product text.

    Args:
        product: Product.

    Returns:
        str: Product text.
    """
    name = product.get("product_name", "")
    description = product.get("product_description", "")
    product_class = product.get("product_class", "")

    semantic_parts = [
        f"product_class: {product_class.upper()}",
        f"product_name: {name}",
        f"product_description: {description}",
    ]
    return " | ".join(filter(None, semantic_parts))


def get_gemini_embeddings_batch(
    texts: List[str], gemini_model: genai.Client, out_dim: int = VECTOR_SIZE
) -> List[List[float]]:
    """Get Gemini embeddings for a batch of texts.

    Args:
        texts: List of texts.
        gemini_model: Gemini model.
        out_dim: Output dimension.

    Returns:
        List[List[float]]: List of embeddings.
    """
    logger.info(f"Getting Gemini embeddings for batch of {len(texts)} texts")
    response = gemini_model.models.embed_content(
        model=GEMINI_MODEL,
        contents=texts,
        config=EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT", output_dimensionality=out_dim
        ),
    )
    return [emb.values for emb in response.embeddings]


def get_splade_embeddings_batch(
    texts: List[str],
    splade_model: SparseEncoder = SPLADE_MODEL,
) -> List[List[float]]:
    """Get SPLADE embeddings for a batch of texts.

    Args:
        texts: List of texts.
        splade_model: SPLADE model.

    Returns:
        List[List[float]]: List of embeddings.
    """
    logger.info(f"Getting SPLADE embeddings for batch of {len(texts)} texts")
    return splade_model.encode(texts)


def process_embeddings_parallel(
    product_texts: List[str],
    splade_model: SparseEncoder,
    gemini_model_client: genai.Client,
) -> tuple:
    """Process dense and sparse embeddings in parallel.

    Args:
        product_texts: List of product texts.
        splade_model: SPLADE model.
        gemini_model_client: Gemini model client.
    """
    dense_embeddings = []
    sparse_embeddings = []

    def process_dense_batch(batch_texts):
        """Process dense embeddings for a batch of texts.

        Args:
            batch_texts: List of texts.

        Returns:
            List[List[float]]: List of embeddings.
        """
        try:
            return get_gemini_embeddings_batch(
                batch_texts, gemini_model_client, VECTOR_SIZE
            )
        except Exception as e:
            logger.warning(f"Error creating dense embeddings for batch: {e}")
            return [[0.0] * VECTOR_SIZE] * len(batch_texts)

    def process_sparse_batch(batch_texts):
        """Process sparse embeddings for a batch of texts.

        Args:
            batch_texts: List of texts.

        Returns:
            List[List[float]]: List of embeddings.
        """
        try:
            return get_splade_embeddings_batch(batch_texts, splade_model)
        except Exception as e:
            logger.warning(f"Error creating sparse embeddings for batch: {e}")
            return [np.zeros(30522, dtype=np.float32)] * len(batch_texts)

    gemini_batches = [
        product_texts[i : i + BATCH_SIZE]
        for i in range(0, len(product_texts), BATCH_SIZE)
    ]
    splade_batches = [
        product_texts[i : i + BATCH_SIZE]
        for i in range(0, len(product_texts), BATCH_SIZE)
    ]

    with ThreadPoolExecutor(max_workers=2) as executor:
        dense_futures = {
            executor.submit(process_dense_batch, batch): i
            for i, batch in enumerate(gemini_batches)
        }
        sparse_futures = {
            executor.submit(process_sparse_batch, batch): i
            for i, batch in enumerate(splade_batches)
        }

        dense_results = [None] * len(gemini_batches)
        sparse_results = [None] * len(splade_batches)

        for future in as_completed(dense_futures):
            batch_idx = dense_futures[future]
            dense_results[batch_idx] = future.result()

        for future in as_completed(sparse_futures):
            batch_idx = sparse_futures[future]
            sparse_results[batch_idx] = future.result()

    for batch_result in dense_results:
        dense_embeddings.extend(batch_result)

    for batch_result in sparse_results:
        sparse_embeddings.extend(batch_result)

    return dense_embeddings, sparse_embeddings


def process_product_batch(
    batch_data: tuple, splade_model: SparseEncoder, gemini_model_client: genai.Client
) -> List[Dict[str, Any]]:
    """Process product batch.

    Args:
        batch_data: Batch data tuple.
        splade_model: SPLADE model.
        gemini_model_client: Gemini model client.

    Returns:
        List[Dict[str, Any]]: List of embeddings.
    """
    products, batch_idx = batch_data

    embeddings = []

    product_texts = []
    product_metadata = []

    for product in products:
        try:
            product_text = create_product_text(product)
            product_texts.append(product_text)
            product_metadata.append(product)
        except Exception as e:
            logger.error(
                f"Error creating text for product {product.get('product_id', 'unknown')}: {e}"
            )
            product_texts.append("")
            product_metadata.append(product)

    try:
        dense_embeddings, sparse_embeddings = process_embeddings_parallel(
            product_texts, splade_model, gemini_model_client
        )

        if hasattr(torch, "mps") and torch.mps.is_available():
            torch.mps.empty_cache()
        elif hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

        for i, (product, dense_emb, sparse_emb) in enumerate(
            zip(product_metadata, dense_embeddings, sparse_embeddings)
        ):
            try:
                if hasattr(sparse_emb, "to_dense"):
                    dense_tensor = sparse_emb.to_dense()
                    sparse_array = dense_tensor.cpu().numpy()
                elif isinstance(sparse_emb, np.ndarray):
                    sparse_array = sparse_emb
                else:
                    sparse_array = np.array(sparse_emb)

                nonzero_mask = sparse_array != 0
                indices = np.where(nonzero_mask)[0].tolist()
                values = sparse_array[nonzero_mask].tolist()

            except Exception as e:
                logger.warning(
                    f"Error processing sparse embedding for product {i}: {e}"
                )
                indices = []
                values = []

            embedding_data = {
                "id": product.get(
                    "product_id", f"product_{batch_idx * len(products) + i}"
                ),
                "embedding": dense_emb,
                "sparse_embedding": {"indices": indices, "values": values},
                "metadata": {
                    "product_name": product.get("product_name", ""),
                    "product_description": product.get("product_description", ""),
                    "product_class": product.get("product_class", ""),
                    "price": float(product.get("price", 0)),
                    "average_rating": float(product.get("average_rating", 0)),
                    "rating_count": int(product.get("rating_count", 0)),
                    "material": product.get("material", []),
                    "style": product.get("style", []),
                    "color": product.get("color", []),
                },
            }

            embeddings.append(embedding_data)

    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        for i, product in enumerate(product_metadata):
            embeddings.append(
                {
                    "id": product.get(
                        "product_id", f"product_{batch_idx * len(products) + i}"
                    ),
                    "embedding": [0.0] * VECTOR_SIZE,
                    "sparse_embedding": {"indices": [], "values": []},
                    "metadata": {
                        "product_name": product.get("product_name", ""),
                        "product_description": product.get("product_description", ""),
                        "product_class": product.get("product_class", ""),
                        "price": 0.0,
                        "average_rating": 0.0,
                        "rating_count": 0,
                        "material": [],
                        "style": [],
                        "color": [],
                    },
                }
            )

    return embeddings


def create_embeddings(output_file: str, num_workers: int, batch_size: int):
    """Create embeddings.

    Args:
        output_file: Output file.
        num_workers: Number of workers.
        batch_size: Batch size.
    """
    logger.info(
        f"Starting embedding creation with {num_workers} workers, batch size {batch_size}"
    )

    products = load_products_from_bucket()
    if not products:
        logger.error("No products loaded. Exiting.")
        return

    product_list = list(products.values())

    for i, product in enumerate(product_list):
        product["index"] = i

    splade_model, gemini_model_client = init_models()

    product_batches = [
        (product_list[i : i + batch_size], i // batch_size)
        for i in range(0, len(product_list), batch_size)
    ]

    logger.info(f"Created {len(product_batches)} batches")

    all_embeddings = []

    for batch_idx, batch in enumerate(product_batches):
        try:
            batch_embeddings = process_product_batch(
                batch, splade_model, gemini_model_client
            )
            all_embeddings.extend(batch_embeddings)

            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(product_batches):
                logger.info(
                    f"Completed {batch_idx + 1}/{len(product_batches)} batches, {len(all_embeddings)} embeddings processed"
                )

        except Exception as e:
            logger.error(f"Error processing batch {batch_idx}: {e}")

    def sort_key(emb):
        emb_id = emb["id"]
        try:
            if emb_id.startswith("product_"):
                return int(emb_id.replace("product_", ""))
            return int(emb_id)
        except (ValueError, TypeError):
            return emb_id

    all_embeddings.sort(key=sort_key)

    logger.info(f"Saving {len(all_embeddings)} embeddings to {output_file}")

    with open(output_file, "w") as f:
        for emb in all_embeddings:
            f.write(json.dumps(emb) + "\n")

    logger.info(f"Embeddings saved successfully to {output_file}")


def sort_key(emb):
    """Sort key.

    Args:
        emb: Embedding.

    Returns:
        int: Sort key.
    """
    emb_id = emb["id"]
    try:
        if emb_id.startswith("product_"):
            return int(emb_id.replace("product_", ""))
        return int(emb_id)
    except (ValueError, TypeError):
        return emb_id


def main():
    products = load_products_from_bucket()
    logger.info(f"Loaded {len(products)} products")
    product_list = list(products.values())
    logger.info(f"Converted {len(product_list)} products to list")
    for i, product in enumerate(product_list):
        product["index"] = i
    logger.info(f"Assigned indices to {len(product_list)} products")
    splade_model, gemini_model_client = init_models()
    logger.info(f"Initialized models")
    product_batches = [
        (product_list[i : i + BATCH_SIZE], i // BATCH_SIZE)
        for i in range(0, len(product_list), BATCH_SIZE)
    ]
    logger.info(f"Created {len(product_batches)} batches")
    all_embeddings = []
    for batch_idx, batch in enumerate(product_batches):
        try:
            batch_embeddings = process_product_batch(
                batch, splade_model, gemini_model_client
            )
            all_embeddings.extend(batch_embeddings)
            logger.info(
                f"Completed {batch_idx + 1}/{len(product_batches)} batches, {len(all_embeddings)} embeddings processed"
            )
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(product_batches):
                logger.info(
                    f"Completed {batch_idx + 1}/{len(product_batches)} batches, {len(all_embeddings)} embeddings processed"
                )

        except Exception as e:
            logger.error(f"Error processing batch {batch_idx}: {e}")

    all_embeddings.sort(key=sort_key)

    with open(EMBEDDINGS_FILE, "w") as f:
        for emb in all_embeddings:
            f.write(json.dumps(emb) + "\n")
    logger.info(f"Embeddings saved successfully to {EMBEDDINGS_FILE}")


if __name__ == "__main__":
    main()
