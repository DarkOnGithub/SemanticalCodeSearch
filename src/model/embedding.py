import logging
import numpy as np
import torch
import gc
from sentence_transformers import SentenceTransformer
from transformers import BitsAndBytesConfig
from typing import List, Union
from src.IR.models import CodeSnippet

logger = logging.getLogger(__name__)

class JinaEmbeddingModel:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(JinaEmbeddingModel, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = "jinaai/jina-code-embeddings-1.5b", use_4bit: bool = True):
        """
        Initializes the Jina Code Embeddings model using sentence-transformers.
        Supports 4-bit quantization for GPU indexing.
        """
        if self._initialized:
            return
            
        self.model_name = model_name
        self.use_4bit = use_4bit
        self.model = None

    def load(self):
        """Public method to force load the model."""
        self._load_model()

    def _load_model(self):
        if self.model is not None:
            return

        if torch.cuda.is_available():
            device = "cuda"
            if self.use_4bit:
                logger.info(f"Loading {self.model_name} in 4-bit quantization for GPU")
                model_kwargs = {
                    "quantization_config": BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    ),
                    "dtype": torch.float16,
                    "attn_implementation": "eager",
                }
            else:
                logger.info(f"Loading {self.model_name} in float16 for GPU")
                model_kwargs = {"dtype": torch.float16, "attn_implementation": "eager"}
        else:
            device = "cpu"
            logger.info(f"Loading {self.model_name} on CPU (4-bit quantization via bitsandbytes is GPU-only, loading in float32)")
            model_kwargs = {}

        try:
            self.model = SentenceTransformer(
                self.model_name, 
                device=device,
                trust_remote_code=True,
                model_kwargs=model_kwargs
            )
            self._initialized = True
            logger.info(f"Jina Embedding model loaded successfully on {device}.")
        except Exception as e:
            logger.error(f"Failed to load Jina Embedding model: {e}")
            raise

    def embed_text(self, text: Union[str, List[str]], batch_size: int = 1) -> np.ndarray:
        """
        Generates embeddings for the given text or list of texts.
        """
        self._load_model()
        try:
            embeddings = self.model.encode(
                text, 
                batch_size=batch_size, 
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embeddings
        except Exception as e:
            logger.error(f"Error during embedding generation: {e}")
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                gc.collect()
            return np.array([])

    def embed_snippets(self, snippets: List[CodeSnippet], batch_size: int = 1, use_summary: bool = False) -> List[np.ndarray]:
        """
        Batch embeds a list of CodeSnippet objects.
        If use_summary is True, combines the summary and the code content.
        Otherwise uses only the code content.
        """
        if not snippets:
            return []
            
        texts = []
        for s in snippets:
            texts.append(s.to_embeddable_text(use_summary=use_summary))
            
        embeddings = self.embed_text(texts, batch_size=batch_size)
        
        if len(embeddings) == 0:
            logger.warning(f"Embedding generation failed for {len(snippets)} snippets. Returning zero vectors.")
            dim = self.model.get_sentence_embedding_dimension() if self.model else 1536
            return [np.zeros(dim) for _ in range(len(snippets))]
            
        return [embeddings[i] for i in range(len(snippets))]

    def clear_cache(self):
        """Manually clear CUDA cache and collect garbage."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def get_embedding_model():
    return JinaEmbeddingModel()
