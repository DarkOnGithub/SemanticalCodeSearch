import hashlib
import os
import logging
from typing import List, Dict
from src.IR.models import CodeSnippet, Relationship, SnippetType
from src.parsers.factory import ParserFactory
from src.graph.python_extractor import PythonRelationshipExtractor
from src.graph.c_extractor import CRelationshipExtractor

logger = logging.getLogger(__name__)

class GraphManager:
    def __init__(self, parser_factory: ParserFactory):
        self.parser_factory = parser_factory
        self.extractors = {
            "python": PythonRelationshipExtractor,
            "c": CRelationshipExtractor,
        }

    def build_graph(self, snippets: List[CodeSnippet], changed_files: set = None) -> List[Relationship]:
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
            # Skip relationship extraction if the file hasn't changed
            if changed_files is not None and file_path not in changed_files:
                continue
                
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

    def create_file_snippets(self, snippets: List[CodeSnippet], changed_files: set = None) -> List[CodeSnippet]:
        """Creates snippets for the files themselves, constructing a structural skeleton."""
        file_snippets = []
        
        # Group snippets by file path
        snippets_by_file: Dict[str, List[CodeSnippet]] = {}
        for s in snippets:
            if s.file_path:
                if s.file_path not in snippets_by_file:
                    snippets_by_file[s.file_path] = []
                snippets_by_file[s.file_path].append(s)
        
        for file_path, file_elements in snippets_by_file.items():
            # If the file hasn't changed, we can find the existing file snippet in the list
            # because it would have been loaded from DB in Pass 1.
            existing_file_snippet = next((s for s in file_elements if s.type == SnippetType.FILE), None)
            
            if changed_files is not None and file_path not in changed_files and existing_file_snippet:
                # File hasn't changed and we already have its snippet from DB
                continue

            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Use a stable ID for the file snippet based on path
                file_id = hashlib.sha256(f"file:{file_path}".encode("utf-8")).hexdigest()
                
                # Sort only top-level skeletons to avoid overlapping in the file-level view
                top_level_skeletons = sorted(
                    [s for s in file_elements if s.is_skeleton and s.parent_id is None],
                    key=lambda x: x.start_byte if x.start_byte is not None else 0
                )
                
                # Build the skeleton content by replacing bodies with "..."
                skeleton_parts = []
                last_idx = 0
                for s in top_level_skeletons:
                    if s.start_byte is None or s.end_byte is None:
                        continue
                    
                    # Add everything (imports, comments, etc.) between the last snippet and this one
                    skeleton_parts.append(content[last_idx:s.start_byte])
                    
                    # Add the snippet's skeleton (e.g., "class MyClass:" or "def func(a):")
                    skeleton_parts.append(s.content.strip())
                    
                    # Add a placeholder for the hidden body
                    skeleton_parts.append("\n    ... # implementation hidden ...\n")
                    last_idx = s.end_byte
                
                # Add the remainder of the file
                skeleton_parts.append(content[last_idx:])
                
                file_skeleton = "".join(skeleton_parts)
                
                file_snippets.append(CodeSnippet(
                    id=file_id,
                    name=os.path.basename(file_path),
                    type=SnippetType.FILE,
                    content=file_skeleton,
                    file_path=file_path,
                    start_line=0,
                    end_line=content.count("\n"),
                    start_byte=0,
                    end_byte=len(content),
                    is_skeleton=True
                ))
            except Exception as e:
                logger.error(f"Error creating file snippet for {file_path}: {e}")
                continue
        
        return file_snippets

