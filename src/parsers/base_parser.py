from abc import ABC, abstractmethod
import logging
from typing import List, Optional, Dict, Any
from src.IR.models import CodeSnippet

logger = logging.getLogger(__name__)

class BaseParser(ABC):
    def __init__(self, chunk_size: int = 8000):
        self._tree_cache: Dict[str, Any] = {}
        self._code_cache: Dict[str, str] = {}
        self._snippet_cache: Dict[str, List[CodeSnippet]] = {}
        self._content_hash_cache: Dict[str, str] = {}
        self._metadata_cache: Dict[str, Dict[str, Any]] = {}
        self.chunk_size = chunk_size

    @property
    @abstractmethod
    def language_id(self) -> str:
        """Returns the tree-sitter language identifier"""
        pass

    @abstractmethod
    def parse_file(self, code: str, file_path: Optional[str] = None) -> List[CodeSnippet]:
        """Should return a list of Normalized IR objects (CodeSnippet)"""
        pass

    def get_cached_snippets(self, file_path: str, code: str) -> Optional[List[CodeSnippet]]:
        """Returns cached snippets if the content hasn't changed"""
        import hashlib
        content_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
        
        if file_path in self._content_hash_cache:
            if self._content_hash_cache[file_path] == content_hash:
                return self._snippet_cache.get(file_path)
        
        self._content_hash_cache[file_path] = content_hash
        return None

    def cache_snippets(self, file_path: str, snippets: List[CodeSnippet]):
        """Caches snippets for a file"""
        self._snippet_cache[file_path] = snippets

    def apply_edit(self, file_path: str, start_byte: int, old_end_byte: int, new_end_byte: int,
                   start_point: tuple, old_end_point: tuple, new_end_point: tuple):
        """Applies an edit to the cached tree for incremental parsing"""
        if file_path in self._tree_cache:
            self._tree_cache[file_path].edit(
                start_byte=start_byte,
                old_end_byte=old_end_byte,
                new_end_byte=new_end_byte,
                start_point=start_point,
                old_end_point=old_end_point,
                new_end_point=new_end_point
            )

    @abstractmethod
    def get_query(self) -> str:
        """Return the Tree-sitter query string for this language"""
        pass
