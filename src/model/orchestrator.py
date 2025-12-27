import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)

class GemmaLLM:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(GemmaLLM, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_id: str = "google/gemma-2-2b-it"):
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
            
        # 4-bit quantization configuration
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        logger.info(f"Loading {self.model_id} in 4-bit quantization...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            self._initialized = True
            logger.info("Gemma model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Gemma model: {e}")
            raise

    def complete(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.1) -> str:
        self._load_model()
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract only the generated part
            input_length = inputs.input_ids.shape[1]
            generated_tokens = outputs[0][input_length:]
            
            return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        except Exception as e:
            logger.error(f"Error during Gemma completion: {e}")
            return ""

HYDE_DECISION_PROMPT = """You are a technical assistant. Your task is to decide if a search query about a codebase would benefit from generating a hypothetical code snippet (HyDE).

HyDE is useful for:
- "How to" questions (e.g., "how to implement a search")
- Questions about specific logic or algorithms (e.g., "where is the depth-first search implemented")
- Questions about function or class usage.

HyDE is NOT useful for:
- General metadata questions (e.g., "what database is used", "how many files are there")
- Questions about project structure in general.
- Questions that don't involve searching for specific code patterns.

Respond with ONLY 'YES' if HyDE is beneficial, or 'NO' if it is not.

Query: {query}

Benefit from HyDE?"""

HYDE_GENERATION_PROMPT = """Generate ONLY a concise, hypothetical code snippet that directly addresses the user's query. 
NO markdown code blocks, NO explanations, ONLY the raw code.

Query: {query}
Code:"""

class Orchestrator:
    def __init__(self):
        self.llm = GemmaLLM()

    def load(self):
        """Forces loading of the underlying LLM."""
        self.llm.load()

    def process_query(self, query: str) -> str:
        """
        Processes the query using Gemma 2 2b to decide if HyDE is needed and generates it if so.
        Returns the augmented query.
        """
        logger.info(f"Orchestrating query: {query}")
        
        decision_prompt = HYDE_DECISION_PROMPT.format(query=query)
        decision = self.llm.complete(decision_prompt, max_new_tokens=5).upper() # Reduced tokens for decision
        
        logger.info(f"HyDE decision: {decision}")
        
        if "YES" in decision:
            logger.info("Generating hypothetical code (HyDE)...")
            gen_prompt = HYDE_GENERATION_PROMPT.format(query=query)
            # Reduced tokens for generation and added more direct instructions
            fake_code = self.llm.complete(gen_prompt, max_new_tokens=128) 
            
            augmented_query = f"{query}\n\nHypothetical code implementation:\n{fake_code}"
            logger.debug(f"Augmented query: {augmented_query}")
            return augmented_query
        
        logger.info("Skipping HyDE generation.")
        return query

def get_orchestrator():
    return Orchestrator()
