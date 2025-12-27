import logging
import os
import json
import re
from google import genai
from google.genai import types
from typing import List, Dict, Generator
from src.IR.models import CodeSnippet

logger = logging.getLogger(__name__)

# Gemini specific configuration
# The summarizer llm is gemini 2.5 flash lite and the answerer is gemini flash 3 low thinking
SUMMARIZER_MODEL_NAME = "gemini-2.5-flash-lite"
ANSWERER_MODEL_NAME = "gemini-3-flash-preview"

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

class GeminiLLM:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(GeminiLLM, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self, 
        summarizer_model: str = SUMMARIZER_MODEL_NAME,
        answerer_model: str = ANSWERER_MODEL_NAME,
        temperature: float = 0.1,
        max_output_tokens: int = 4096,
    ):
        """
        Initializes the Gemini client using the Google GenAI SDK.
        Expects GEMINI_API_KEY to be set in the environment.
        """
        if self._initialized:
            return
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not found in environment. GeminiLLM may fail.")
            
        self.client = genai.Client(api_key=api_key)
        
        self.summarizer_model = summarizer_model
        self.answerer_model = answerer_model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self._initialized = True

    def complete(self, prompt: str, json_mode: bool = False) -> str:
        """Generates a completion using the answerer model."""
        try:
            thinking_config = None
            if "gemini-3" in self.answerer_model:
                try:
                    thinking_config = types.ThinkingConfig(thinking_level="low")
                except (AttributeError, TypeError):
                    thinking_config = types.ThinkingConfig(include_thoughts=True)
            
            config = types.GenerateContentConfig(
                max_output_tokens=self.max_output_tokens,
                temperature=0.0 if json_mode else self.temperature,
                thinking_config=thinking_config,
                response_mime_type="application/json" if json_mode else "text/plain"
            )

            response = self.client.models.generate_content(
                model=self.answerer_model,
                contents=prompt,
                config=config
            )
            
            text = response.text
            
            if json_mode:
                text = self._clean_json_response(text)
            return text
        except Exception as e:
            logger.error(f"Error during Gemini completion: {e}")
            return ""

    def stream_complete(self, prompt: str) -> Generator[str, None, None]:
        """Generates a stream of tokens using the answerer model."""
        try:
            thinking_config = None
            if "gemini-3" in self.answerer_model:
                try:
                    thinking_config = types.ThinkingConfig(thinking_level="low")
                except (AttributeError, TypeError):
                    thinking_config = types.ThinkingConfig(include_thoughts=True)

            config = types.GenerateContentConfig(
                max_output_tokens=self.max_output_tokens,
                temperature=self.temperature,
                thinking_config=thinking_config
            )

            response = self.client.models.generate_content_stream(
                model=self.answerer_model,
                contents=prompt,
                config=config
            )
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.error(f"Error during Gemini streaming: {e}")
            yield f"Error: {str(e)}"

    def summarize_batch(self, snippets: List[CodeSnippet], child_summaries_map: Dict[str, List[str]]):
        """Summarizes a batch of snippets using the summarizer model (Gemini 2.5 Flash Lite)."""
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
            logger.info(f"Batch summarizing {len(snippets)} snippets with {self.summarizer_model}...")
            prompt = BATCH_SUMMARY_PROMPT.format(components_json=json.dumps(components, indent=2))
            
            config = types.GenerateContentConfig(
                max_output_tokens=self.max_output_tokens,
                temperature=0.0,
                response_mime_type="application/json"
            )
            
            response = self.client.models.generate_content(
                model=self.summarizer_model,
                contents=prompt,
                config=config
            )
            
            response_text = self._clean_json_response(response.text)
            if not response_text:
                return

            try:
                results = json.loads(response_text)
            except json.JSONDecodeError as je:
                logger.warning(f"Failed to parse JSON directly. Attempting extraction. Error: {je}")
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
                    val = results[s.id]
                    if isinstance(val, dict):
                        # If the LLM returned a JSON object instead of a string,
                        # try to get a 'summary' or 'content' key, otherwise dump to string.
                        s.summary = val.get("summary") or val.get("content") or json.dumps(val)
                    else:
                        s.summary = str(val) if val is not None else None
        except Exception as e:
            logger.error(f"Error in batch summarization: {e}")

    def summarize_snippet(self, snippet: CodeSnippet, child_summaries: List[str] = None):
        """Generates a summary for a single snippet."""
        self.summarize_batch([snippet], {snippet.id: child_summaries or []})

    def _clean_json_response(self, text: str) -> str:
        """Basic cleanup for JSON responses."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text

def get_llm():
    return GeminiLLM()
