import json
import logging
from typing import Any, Dict, List

import numpy as np
import torch
from config import BATCH_SIZE, VECTOR_SIZE
from data_util import load_product_data, save_embeddings
from model_manager import model_manager

logger = logging.getLogger()


def initialize_models():
    """Initialize models using the model manager"""
    model_manager.initialize_models()


def create_product_text(product: Dict[str, Any]) -> str:
    """Create product text.

    Args:
        product: Product dictionary.

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


def process_product_batch(batch_data: tuple) -> List[Dict[str, Any]]:
    """Process product batch.

    Args:
        batch_data: Batch data tuple.

    Returns:
        List[Dict[str, Any]]: List of embeddings.
    """
    products, batch_idx = batch_data

    # Get models from manager
    splade_model = model_manager.get_splade_model()
    qwen_model = model_manager.get_qwen_model()

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
        # Create dense embeddings
        dense_embeddings = []
        for i in range(0, len(product_texts), BATCH_SIZE):
            batch_texts = product_texts[i : i + BATCH_SIZE]
            batch_embeddings = qwen_model.encode(batch_texts)
            dense_embeddings.extend(batch_embeddings)

            # Memory cleanup
            del batch_embeddings
            if hasattr(torch, "mps") and torch.mps.is_available():
                torch.mps.empty_cache()
            elif hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Create sparse embeddings
        sparse_embeddings = []
        for i in range(0, len(product_texts), BATCH_SIZE):
            batch_texts = product_texts[i : i + BATCH_SIZE]
            batch_sparse = splade_model.encode(batch_texts)
            sparse_embeddings.extend(batch_sparse)

            del batch_sparse
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
                "dense_embedding": dense_emb,
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
                    "dense_embedding": np.zeros(VECTOR_SIZE, dtype=np.float32),
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


def create_embeddings(
    products_file: str, output_file: str, num_workers: int, batch_size: int
):
    """Create embeddings.

    Args:
        products_file: Products file.
        output_file: Output file.
        num_workers: Number of workers.
        batch_size: Batch size.
    """
    logger.info(
        f"Starting embedding creation with {num_workers} workers, batch size {batch_size}"
    )

    # Ensure models are initialized
    if not model_manager.is_initialized():
        model_manager.initialize_models()

    products = load_product_data(products_file)
    if products is None or products.empty:
        logger.error("No products loaded. Exiting.")
        return

    product_batches = [
        (products[i : i + batch_size].to_dict("records"), i // batch_size)
        for i in range(0, len(products), batch_size)
    ]

    logger.info(f"Created {len(product_batches)} batches")

    all_embeddings = []

    for batch_idx, batch in enumerate(product_batches):
        try:
            batch_embeddings = process_product_batch(batch)
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
    save_embeddings(all_embeddings, output_file)

    logger.info(f"Embeddings saved successfully to {output_file}")
