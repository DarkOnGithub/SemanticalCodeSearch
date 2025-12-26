import tree_sitter_python as tspython
from tree_sitter import Language, Parser
from typing import List, Optional
from src.parsers.base_parser import BaseParser
from src.IR.models import CodeSnippet, SnippetType
import hashlib
from tree_sitter import Query, QueryCursor

class PythonParser(BaseParser):
    def __init__(self):
        super().__init__()
        self.language = Language(tspython.language())
        self.parser = Parser(self.language)

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
        old_tree = self._tree_cache.get(file_path) if file_path else None
        
        if old_tree:
            tree = self.parser.parse(bytes(code, "utf8"), old_tree)
        else:
            tree = self.parser.parse(bytes(code, "utf8"))
        
        if file_path:
            self._tree_cache[file_path] = tree

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
                snippet = self._extract_snippet(node, tag, code, file_path)
                if snippet:
                    snippets.append(snippet)
        
        return snippets

    def _extract_snippet(self, node, tag, code, file_path) -> CodeSnippet:
        snippet_type = SnippetType.CLASS if "class" in tag else SnippetType.FUNCTION
        
        # Check if this is a method (function inside a class)
        parent_id = None
        curr = node.parent
        while curr:
            if curr.type == "class_definition":
                if snippet_type == SnippetType.FUNCTION:
                    snippet_type = SnippetType.METHOD
                
                # Calculate parent class ID (hash of its content)
                parent_content = code[curr.start_byte:curr.end_byte]
                parent_id = hashlib.sha256(parent_content.encode("utf-8")).hexdigest()
                break
            curr = curr.parent

        name_node = node.child_by_field_name("name")
        name = code[name_node.start_byte:name_node.end_byte] if name_node else "anonymous"
        
        params = ""
        if snippet_type == SnippetType.FUNCTION:
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

        snippet_content = code[node.start_byte:node.end_byte]
        snippet_id = hashlib.sha256(snippet_content.encode("utf-8")).hexdigest()

        return CodeSnippet(
            id=snippet_id,
            name=name,
            type=snippet_type,
            content=snippet_content,
            parent_id=parent_id,
            docstring=docstring if docstring else None,
            signature=f"{name}{params}" if snippet_type in [SnippetType.FUNCTION, SnippetType.METHOD] else name,
            file_path=file_path,
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            start_byte=node.start_byte,
            end_byte=node.end_byte
        )
