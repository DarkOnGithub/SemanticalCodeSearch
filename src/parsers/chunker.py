from typing import List
from llama_index.core.node_parser import CodeSplitter

class CodeChunker:
    def __init__(self, language: str, chunk_lines: int = 100, chunk_lines_overlap: int = 20, max_chars: int = 8000):
        self.splitter = CodeSplitter(
            language=language,
            chunk_lines=chunk_lines,
            chunk_lines_overlap=chunk_lines_overlap,
            max_chars=max_chars,
        )

    def chunk(self, code: str) -> List[str]:
        """Splits code into smaller chunks using LlamaIndex CodeSplitter."""
        return self.splitter.split_text(code)

