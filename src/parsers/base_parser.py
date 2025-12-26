from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from src.IR.models import CodeSnippet

class BaseParser(ABC):
    def __init__(self):
        self._tree_cache: Dict[str, Any] = {}

    @property
    @abstractmethod
    def language_id(self) -> str:
        """Returns the tree-sitter language identifier"""
        pass

    @abstractmethod
    def parse_file(self, code: str, file_path: Optional[str] = None) -> List[CodeSnippet]:
        """Should return a list of Normalized IR objects (CodeSnippet)"""
        pass

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
