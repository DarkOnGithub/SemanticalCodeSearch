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

        logger.info(f"Loading {self.model_id} in float16...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            self.model.eval()
            self._initialized = True
            logger.info("Jina Reranker loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Jina Reranker: {e}")
            raise

    def rerank(self, query: str, documents: List[str], top_n: int = 5, batch_size: int = 4) -> List[Dict[str, Any]]:
        """
        Reranks a list of documents based on a query using small batches to avoid OOM.
        Returns a list of dictionaries with index and score, sorted by score descending.
        """
        if not documents:
            return []
            
        self._load_model()

        try:
            clean_docs = [str(doc) if doc is not None else "" for doc in documents]
            all_scores = []
            
            # Process in small batches to avoid OOM
            for i in range(0, len(clean_docs), batch_size):
                batch_docs = clean_docs[i:i + batch_size]
                pairs = [[query, doc] for doc in batch_docs]
                
                batch_scores = None
                with torch.no_grad():
                    # Try the model's built-in compute_score first
                    if hasattr(self.model, 'compute_score'):
                        try:
                            res = self.model.compute_score(pairs, max_length=1024)
                            if isinstance(res, (list, torch.Tensor)):
                                batch_scores = res
                            elif isinstance(res, tuple) and len(res) > 0:
                                batch_scores = res[0]
                            else:
                                batch_scores = res
                        except Exception as ce:
                            logger.warning(f"Batch compute_score failed: {ce}")

                    # Fallback to manual scoring for this batch
                    if batch_scores is None:
                        inputs = self.tokenizer(
                            pairs, 
                            padding=True, 
                            truncation=True, 
                            return_tensors="pt", 
                            max_length=1024
                        ).to(self.model.device)
                        
                        inputs.pop("token_type_ids", None)
                        
                        outputs = self.model(**inputs, return_dict=True)
                        if hasattr(outputs, 'logits'):
                            logits = outputs.logits.view(-1, ).float()
                        else:
                            logits = outputs[0].view(-1, ).float()
                            
                        batch_scores = torch.sigmoid(logits).cpu().numpy().tolist()
                
                if isinstance(batch_scores, torch.Tensor):
                    batch_scores = batch_scores.cpu().numpy().tolist()
                elif not isinstance(batch_scores, list):
                    batch_scores = [float(batch_scores)]
                
                all_scores.extend(batch_scores)
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            results = [
                {"index": i, "score": float(score)}
                for i, score in enumerate(all_scores)
            ]
            
            results.sort(key=lambda x: x["score"], reverse=True)
            
            return results[:top_n]
        except Exception as e:
            logger.error(f"Error during reranking: {e}", exc_info=True)
            return [{"index": i, "score": 0.0001 / (i + 1)} for i in range(min(len(documents), top_n))]

def get_reranker():
    return JinaReranker()

