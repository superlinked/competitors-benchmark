import logging
import os
from typing import Optional, Tuple

from config import DEVICE, DEVICE_SPLADE, QWEN_EMB_MODEL, SPLADE_MODEL
from sentence_transformers import SentenceTransformer, SparseEncoder

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


class ModelManager:
    """Centralized model manager with lazy loading and caching"""

    def __init__(self):
        self._splade_model: Optional[SparseEncoder] = None
        self._qwen_model: Optional[SentenceTransformer] = None
        self._initialized = False

    def initialize_models(self) -> None:
        """Initialize all models lazily"""
        if self._initialized:
            return

        logger.info("Initializing models...")
        try:
            self._splade_model = SparseEncoder(SPLADE_MODEL, device=DEVICE_SPLADE)
            self._qwen_model = SentenceTransformer(QWEN_EMB_MODEL, device=DEVICE)
            self._initialized = True
            logger.info("Models initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise

    def get_splade_model(self) -> SparseEncoder:
        """Get SPLADE model, loading if necessary"""
        if not self._initialized:
            self.initialize_models()
        return self._splade_model

    def get_qwen_model(self) -> SentenceTransformer:
        """Get QWEN model, loading if necessary"""
        if not self._initialized:
            self.initialize_models()
        return self._qwen_model

    def get_models(self) -> Tuple[SparseEncoder, SentenceTransformer]:
        """Get both models, loading if necessary"""
        if not self._initialized:
            self.initialize_models()
        return self._splade_model, self._qwen_model

    def is_initialized(self) -> bool:
        """Check if models are initialized"""
        return self._initialized

    def clear_models(self) -> None:
        """Clear models from memory"""
        self._splade_model = None
        self._qwen_model = None
        self._initialized = False
        logger.info("Models cleared from memory")


# Global instance
model_manager = ModelManager()
