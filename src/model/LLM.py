import logging
import os
import json
from openai import OpenAI  # Changed from google.genai
from typing import List, Dict
from src.IR.models import CodeSnippet

logger = logging.getLogger(__name__)

# DeepSeek specific configuration

THINKING = False # DeepSeek V3 is standard; set to True to use 'deepseek-reasoner' (R1)
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

SUMMARY_PROMPT = """You are an expert software architect. Your task is to provide a concise, high-level technical summary of the code provided below.
Focus on:
1. The main responsibility/purpose of the code.
2. Key inputs and outputs (if applicable).
3. Any significant side effects or dependencies.

Avoid line-by-line explanations. Keep the summary under 3 sentences.

Code:
{code}

Technical Summary:"""

BATCH_SUMMARY_PROMPT = """You are an expert software architect. Your task is to provide concise, technical summaries for the multiple code components provided below.

For each component, provide a summary (under 3 sentences) focusing on:
1. Responsibility/purpose.
2. Key inputs/outputs.
3. Side effects/dependencies.

You MUST return your response as a valid JSON object where the keys are the IDs provided and the values are the technical summaries.
Ensure all newlines in summaries are escaped as '\\n'.

Components to summarize:
{components_json}

JSON Output:"""

class DeepSeekLLM:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(DeepSeekLLM, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self, 
        model_name: str = "deepseek-chat", # Points to DeepSeek-V3
        temperature: float = 0.1,
        max_output_tokens: int = 4096,
    ):
        """
        Initializes the DeepSeek client using the OpenAI SDK.
        Expects DEEPSEEK_API_KEY to be set in the environment.
        """
        if self._initialized:
            return
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            logger.warning("DEEPSEEK_API_KEY not found in environment. DeepSeekLLM may fail.")
            
        # Initialize OpenAI client pointing to DeepSeek's URL
        self.client = OpenAI(
            api_key=api_key, 
            base_url=DEEPSEEK_BASE_URL
        )
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self._initialized = True

    def complete(self, prompt: str, json_mode: bool = False) -> str:
        try:
            # Handle Model Selection based on "THINKING" flag
            # DeepSeek-V3 (chat) is fast/cheap. DeepSeek-R1 (reasoner) is for thinking.
            current_model = "deepseek-reasoner" if THINKING else self.model_name
            
            # Prepare arguments
            system_content = "You are a helpful assistant."
            if json_mode:
                system_content += " Always output valid JSON."
                
            kwargs = {
                "model": current_model,
                "messages": [
                    {"role": "system", "content": system_content}, 
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": self.max_output_tokens,
            }

            # Add JSON mode enforcement if requested
            # Note: DeepSeek requires the word "json" in the prompt AND this parameter
            if json_mode and not THINKING:
                kwargs["response_format"] = {"type": "json_object"}
                kwargs["temperature"] = 0.0 # Force deterministic output for JSON
            elif not THINKING:
                kwargs["temperature"] = self.temperature

            response = self.client.chat.completions.create(**kwargs)
            
            text = response.choices[0].message.content
            
            # Basic cleanup if markdown fences leak through
            if json_mode:
                text = text.strip()
                if text.startswith("```"):
                    lines = text.splitlines()
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    text = "\n".join(lines).strip()
            return text
        except Exception as e:
            logger.error(f"Error during DeepSeek completion: {e}")
            return ""

    def summarize_batch(self, snippets: List[CodeSnippet], child_summaries_map: Dict[str, List[str]]):
        """Summarizes a batch of snippets in a single LLM call."""
        if not snippets:
            return
            
        components = []
        for s in snippets:
            c_summaries = child_summaries_map.get(s.id, [])
            context = f"Code snippet:\n{s.content}\n" if s.content else ""
            if c_summaries:
                context += "\nSub-components of this item:\n" + "\n".join(f"- {cs}" for cs in c_summaries)
            
            components.append({
                "id": s.id,
                "name": s.name,
                "type": s.type.value,
                "context": context
            })

        try:
            logger.info(f"Batch summarizing {len(snippets)} snippets with DeepSeek...")
            prompt = BATCH_SUMMARY_PROMPT.format(components_json=json.dumps(components, indent=2))
            
            response_text = self.complete(prompt, json_mode=True)
            if not response_text:
                return

            try:
                results = json.loads(response_text)
            except json.JSONDecodeError as je:
                logger.warning(f"Failed to parse JSON directly. Attempting extraction. Error: {je}")
                # Fallback extraction logic
                import re
                match = re.search(r"(\{.*\})", response_text, re.DOTALL)
                if match:
                    try:
                        results = json.loads(match.group(1))
                    except Exception:
                        logger.error("Regex extraction failed.")
                        return
                else:
                    return

            for s in snippets:
                if s.id in results:
                    s.summary = results[s.id]
        except Exception as e:
            logger.error(f"Error in batch summarization: {e}")

    def summarize_snippet(self, snippet: CodeSnippet, child_summaries: List[str] = None):
        """Generates a summary for a single snippet."""
        self.summarize_batch([snippet], {snippet.id: child_summaries or []})

def get_llm():
    return DeepSeekLLM()
