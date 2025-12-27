import logging
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core.base.llms.types import ChatMessage
from typing import List, Optional
from src.IR.models import CodeSnippet

logger = logging.getLogger(__name__)

SUMMARY_PROMPT = """You are an expert software architect. Your task is to provide a concise, high-level technical summary of the code provided below.
Focus on:
1. The main responsibility/purpose of the code.
2. Key inputs and outputs (if applicable).
3. Any significant side effects or dependencies.

Avoid line-by-line explanations. Keep the summary under 3 sentences.

Code:
{code}

Technical Summary:"""

class QwenLLM:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(QwenLLM, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self, 
        model_url: Optional[str] = "https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF/resolve/main/qwen2.5-coder-3b-instruct-q8_0.gguf",
        model_path: Optional[str] = None,
        temperature: float = 0.1,
        max_new_tokens: int = 512,
        context_window: int = 32768,
        generate_kwargs: Optional[dict] = None,
        model_kwargs: Optional[dict] = None,
        verbose: bool = False
    ):
        """
        Initializes the Qwen 2.5 Coder 3B Instruct model using llama-cpp.
        """
        if self._initialized:
            return
            
        self.llm = LlamaCPP(
            model_url=model_url,
            model_path=model_path,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            context_window=context_window,
            generate_kwargs=generate_kwargs or {"stop": ["<|endoftext|>", "<|im_end|>", "<|im_start|>"]},
            model_kwargs=model_kwargs or {"n_gpu_layers": -1}, # Use GPU if available
            verbose=verbose,
        )
        self._initialized = True

    def complete(self, prompt: str) -> str:
        return str(self.llm.complete(prompt))

    def chat(self, messages: List[ChatMessage]) -> str:
        return str(self.llm.chat(messages))

    def summarize_snippet(self, snippet: CodeSnippet):
        """Generates a summary for a snippet and updates it in place."""
        if not snippet.content:
            return
            
        try:
            logger.info(f"Generating summary for snippet: {snippet.name}")
            prompt = SUMMARY_PROMPT.format(code=snippet.content)
            snippet.summary = self.complete(prompt)
        except Exception as e:
            logger.error(f"Error summarizing snippet {snippet.id}: {e}")

def get_llm():
    return QwenLLM()

