from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from src.IR.models import CodeSnippet, Relationship
from tree_sitter import Node, Tree

class BaseRelationshipExtractor(ABC):
    def __init__(self, code: str, tree: Tree, file_path: str, snippets: List[CodeSnippet], symbol_table: Dict[str, str] = None):
        self.code = code
        self.tree = tree
        self.file_path = file_path
        self.snippets = snippets
        self.symbol_table = symbol_table or {}
        
        self.ts_id_to_snippet_id: Dict[int, str] = {}
        self.range_to_snippet_id: Dict[tuple, str] = {}
        
        for s in snippets:
            ts_id = s.metadata.get("ts_node_id")
            if ts_id is not None:
                self.ts_id_to_snippet_id[ts_id] = s.id
            
            self.range_to_snippet_id[(s.start_byte, s.end_byte)] = s.id

    @abstractmethod
    def extract(self) -> List[Relationship]:
        pass

    def resolve_symbol(self, name: str) -> str:
        """Resolves a symbol name to a snippet ID if possible, otherwise returns name"""
        return self.symbol_table.get(name, name)

    def find_containing_snippet_id(self, node: Node) -> Optional[str]:
        """Finds the ID of the snippet that contains this node"""
        curr = node
        while curr:
            if curr.id in self.ts_id_to_snippet_id:
                return self.ts_id_to_snippet_id[curr.id]
            
            range_key = (curr.start_byte, curr.end_byte)
            if range_key in self.range_to_snippet_id:
                return self.range_to_snippet_id[range_key]
                
            curr = curr.parent
        return None

