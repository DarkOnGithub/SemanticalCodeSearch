import tree_sitter_python as tspython
from tree_sitter import Language, Parser
from typing import List, Optional
from src.parsers.base_parser import BaseParser
from src.IR.models import CodeSnippet, SnippetType
import hashlib
from tree_sitter import Query, QueryCursor

from src.parsers.chunker import CodeChunker

class PythonParser(BaseParser):
    def __init__(self, chunk_size: int = 1500):
        super().__init__(chunk_size=chunk_size)
        self.language = Language(tspython.language())
        self.parser = Parser(self.language)
        self.chunker = CodeChunker(language="python", chunk_max_characters=chunk_size)

    @property
    def language_id(self) -> str:
        return "python"

    def get_query(self) -> str:
        return """
        (class_definition
          name: (identifier) @class.name
          body: (block) @class.body) @class.def

        (function_definition
          name: (identifier) @function.name
          parameters: (parameters) @function.params
          body: (block) @function.body) @function.def
        """

    def parse_file(self, code: str, file_path: Optional[str] = None) -> List[CodeSnippet]:
        if file_path:
            cached = self.get_cached_snippets(file_path, code)
            if cached is not None:
                return cached

        old_tree = self._tree_cache.get(file_path) if file_path else None
        
        if old_tree:
            tree = self.parser.parse(bytes(code, "utf8"), old_tree)
        else:
            tree = self.parser.parse(bytes(code, "utf8"))
        
        if file_path:
            self._tree_cache[file_path] = tree
            self._code_cache[file_path] = code

        query = Query(self.language, self.get_query())
        cursor = QueryCursor(query)
        captures_dict = cursor.captures(tree.root_node)
        
        # Flatten and sort captures for consistent processing
        all_captures = []
        for tag, nodes in captures_dict.items():
            for node in nodes:
                all_captures.append((node, tag))
        all_captures.sort(key=lambda x: x[0].start_byte)

        snippets = []
        for node, tag in all_captures:
            if tag in ["class.def", "function.def"]:
                extracted = self._extract_snippets(node, tag, code, file_path)
                snippets.extend(extracted)
        
        if file_path:
            self.cache_snippets(file_path, snippets)
            
        return snippets

    def _extract_snippets(self, node, tag, code, file_path) -> List[CodeSnippet]:
        snippet_content = code[node.start_byte:node.end_byte]
        
        if len(snippet_content) <= self.chunk_size:
            return [self._create_snippet(node, tag, code, file_path, snippet_content)]
        
        nodes = self.chunker.chunk_to_nodes(snippet_content)
        
        snippets = []
        for i, text_node in enumerate(nodes):
            display_name = None
            if "inclusive_scopes" in text_node.metadata and text_node.metadata["inclusive_scopes"]:
                scopes = text_node.metadata["inclusive_scopes"]
                display_name = scopes[-1]["name"]
            
            snippet = self._create_snippet(node, tag, code, file_path, text_node.get_content(), chunk_index=i)
            
            if display_name:
                snippet.name = f"{display_name}_chunk_{i}"
            
            snippet.metadata["llama_index_node_id"] = text_node.node_id
            if text_node.parent_node:
                snippet.metadata["llama_index_parent_id"] = text_node.parent_node.node_id
            
            snippets.append(snippet)
        
        return snippets

    def _create_snippet(self, node, tag, code, file_path, snippet_content, chunk_index: Optional[int] = None) -> CodeSnippet:
        snippet_id = hashlib.sha256(snippet_content.encode("utf-8")).hexdigest()

        cached_meta = self._metadata_cache.get(snippet_id)
        if cached_meta:
            return CodeSnippet(
                id=snippet_id,
                name=cached_meta["name"],
                type=cached_meta["type"],
                content=snippet_content,
                parent_id=cached_meta["parent_id"],
                docstring=cached_meta["docstring"],
                signature=cached_meta["signature"],
                file_path=file_path,
                start_line=node.start_point[0],
                end_line=node.end_point[0],
                start_byte=node.start_byte,
                end_byte=node.end_byte,
                metadata={"chunk_index": chunk_index, "ts_node_id": node.id} if chunk_index is not None else {"ts_node_id": node.id}
            )

        snippet_type = SnippetType.CLASS if "class" in tag else SnippetType.FUNCTION
        
        parent_id = None
        curr = node.parent
        while curr:
            if curr.type == "class_definition":
                if snippet_type == SnippetType.FUNCTION:
                    snippet_type = SnippetType.METHOD
                
                parent_content = code[curr.start_byte:curr.end_byte]
                parent_id = hashlib.sha256(parent_content.encode("utf-8")).hexdigest()
                break
            curr = curr.parent

        name_node = node.child_by_field_name("name")
        name = code[name_node.start_byte:name_node.end_byte] if name_node else "anonymous"
        if chunk_index is not None:
            name = f"{name}_chunk_{chunk_index}"
        
        params = ""
        if snippet_type in [SnippetType.FUNCTION, SnippetType.METHOD]:
            params_node = node.child_by_field_name("parameters")
            params = code[params_node.start_byte:params_node.end_byte] if params_node else "()"

        docstring = ""
        body_node = node.child_by_field_name("body")
        if body_node and body_node.type == "block" and body_node.children:
            first_stmt = body_node.children[0]
            if first_stmt.type == "expression_statement":
                expr = first_stmt.children[0]
                if expr.type == "string":
                    docstring = code[expr.start_byte:expr.end_byte].strip('\"\'')

        comments = []
        prev = node.prev_sibling
        while prev and prev.type == "comment":
            comments.append(code[prev.start_byte:prev.end_byte].strip("# ").strip())
            prev = prev.prev_sibling
        
        if comments:
            leading_comment = "\n".join(reversed(comments))
            if docstring:
                docstring = leading_comment + "\n" + docstring
            else:
                docstring = leading_comment

        signature = f"{name}{params}" if snippet_type in [SnippetType.FUNCTION, SnippetType.METHOD] else name

        self._metadata_cache[snippet_id] = {
            "name": name,
            "type": snippet_type,
            "parent_id": parent_id,
            "docstring": docstring if docstring else None,
            "signature": signature
        }

        metadata = {"ts_node_id": node.id}
        if chunk_index is not None:
            metadata["chunk_index"] = chunk_index

        return CodeSnippet(
            id=snippet_id,
            name=name,
            type=snippet_type,
            content=snippet_content,
            parent_id=parent_id,
            docstring=docstring if docstring else None,
            signature=signature,
            file_path=file_path,
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            metadata=metadata
        )
