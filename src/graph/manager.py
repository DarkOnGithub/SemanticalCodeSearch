from typing import List, Dict
from src.IR.models import CodeSnippet, Relationship, SnippetType
from src.parsers.factory import ParserFactory
from src.graph.python_extractor import PythonRelationshipExtractor
from src.graph.c_extractor import CRelationshipExtractor
import hashlib
import os

class GraphManager:
    def __init__(self, parser_factory: ParserFactory):
        self.parser_factory = parser_factory
        self.extractors = {
            "python": PythonRelationshipExtractor,
            "c": CRelationshipExtractor,
        }

    def build_graph(self, snippets: List[CodeSnippet]) -> List[Relationship]:
        """
        Second pass: Extract relationships from snippets and tree-sitter trees.
        """
        all_relationships = []
        
        # Build global symbol table for name resolution
        symbol_table: Dict[str, str] = {}
        for s in snippets:
            if s.type in [SnippetType.FUNCTION, SnippetType.CLASS, SnippetType.STRUCT, SnippetType.ENUM]:
                if s.name not in symbol_table:
                    symbol_table[s.name] = s.id

        # Group snippets by file path
        snippets_by_file: Dict[str, List[CodeSnippet]] = {}
        for s in snippets:
            if s.file_path:
                if s.file_path not in snippets_by_file:
                    snippets_by_file[s.file_path] = []
                snippets_by_file[s.file_path].append(s)

        for file_path, file_snippets in snippets_by_file.items():
            parser = self.parser_factory.get_parser_for_file(file_path)
            if not parser:
                continue

            # Get the tree and code from parser's cache (Optimization: No disk I/O in Pass 2)
            tree = parser._tree_cache.get(file_path)
            code = parser._code_cache.get(file_path)
            
            if not tree or not code:
                # Fallback to disk only if cache is missing (unlikely in 2-pass flow)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        code = f.read()
                    tree = parser.parser.parse(bytes(code, "utf8"))
                except Exception:
                    continue

            # Choose extractor based on language
            extractor_cls = self.extractors.get(parser.language_id)
            if extractor_cls:
                extractor = extractor_cls(code, tree, file_path, file_snippets, symbol_table)
                relationships = extractor.extract()
                all_relationships.extend(relationships)
        
        return all_relationships

    def create_file_snippets(self, snippets: List[CodeSnippet]) -> List[CodeSnippet]:
        """Creates snippets for the files themselves to serve as nodes in the graph"""
        file_snippets = []
        files_seen = set()
        
        for s in snippets:
            if s.file_path and s.file_path not in files_seen:
                try:
                    with open(s.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    file_id = hashlib.sha256(content.encode("utf-8")).hexdigest()
                    file_snippets.append(CodeSnippet(
                        id=file_id,
                        name=os.path.basename(s.file_path),
                        type=SnippetType.FILE,
                        content=content,
                        file_path=s.file_path,
                        start_line=0,
                        end_line=content.count("\n"),
                        start_byte=0,
                        end_byte=len(content)
                    ))
                    files_seen.add(s.file_path)
                except Exception:
                    continue
        
        return file_snippets

