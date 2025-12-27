import logging
from typing import List
from llama_index.core.schema import TextNode
from llama_index.packs.code_hierarchy import CodeHierarchyNodeParser

logger = logging.getLogger(__name__)

class CodeChunker:
    def __init__(self, language: str, chunk_min_characters: int = 1000, chunk_max_characters: int = 8000):
        target_language = "cpp" if language == "c" else language
        
        self.parser = CodeHierarchyNodeParser(
            language=target_language,
            chunk_min_characters=chunk_min_characters,
        )

    def chunk_to_nodes(self, code: str) -> List[TextNode]:
        """Splits code into a hierarchy of nodes."""
        return self.parser.get_nodes_from_documents([TextNode(text=code)])

