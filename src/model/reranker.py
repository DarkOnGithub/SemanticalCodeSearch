import logging
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class JinaReranker:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(JinaReranker, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_id: str = "jinaai/jina-reranker-v2-base-multilingual"):
        if self._initialized:
            return
        
        self.model_id = model_id
        self.tokenizer = None
        self.model = None

    def load(self):
        """Public method to force load the model."""
        self._load_model()

    def _load_model(self):
        if self.model is not None:
            return

        # int8 quantization configuration
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

        logger.info(f"Loading {self.model_id} in int8 quantization...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="eager",
            )
            self.model.eval()
            self._initialized = True
            logger.info("Jina Reranker loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Jina Reranker: {e}")
            raise

    def rerank(self, query: str, documents: List[str], top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Reranks a list of documents based on a query.
        Returns a list of dictionaries with index and score, sorted by score descending.
        """
        if not documents:
            return []
            
        self._load_model()

        try:
            # Prepare pairs for the reranker
            pairs = [[query, doc] for doc in documents]
            
            with torch.no_grad():
                # Jina reranker v2 base multilingual provides compute_score
                # which handles tokenization and scoring internally if trust_remote_code=True
                if hasattr(self.model, 'compute_score'):
                    # Some versions return a tuple, some return a list. Handling both.
                    res = self.model.compute_score(pairs, max_length=1024)
                    if isinstance(res, (list, torch.Tensor)):
                        scores = res
                    elif isinstance(res, tuple) and len(res) > 0:
                        scores = res[0]
                    else:
                        scores = res
                else:
                    # Fallback to manual scoring if compute_score is not available
                    inputs = self.tokenizer(
                        pairs, 
                        padding=True, 
                        truncation=True, 
                        return_tensors="pt", 
                        max_length=1024
                    ).to(self.model.device)
                    
                    outputs = self.model(**inputs, return_dict=True)
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits.view(-1, ).float()
                    else:
                        # Sometimes it might return a tuple if return_dict=False or other reasons
                        logits = outputs[0].view(-1, ).float()
                        
                    scores = torch.sigmoid(logits).cpu().numpy().tolist()

            # Create list of (index, score) pairs
            results = [
                {"index": i, "score": float(score)}
                for i, score in enumerate(scores)
            ]
            
            # Sort by score descending
            results.sort(key=lambda x: x["score"], reverse=True)
            
            return results[:top_n]
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            # Fallback: return first top_n results with 0 score if reranking fails
            return [{"index": i, "score": 0.0} for i in range(min(len(documents), top_n))]

def get_reranker():
    return JinaReranker()

