import tree_sitter_c as tsc
from tree_sitter import Language, Parser, Query, QueryCursor
from typing import List, Optional
from src.parsers.base_parser import BaseParser
from src.IR.models import CodeSnippet, SnippetType
import hashlib

class CParser(BaseParser):
    def __init__(self):
        super().__init__()
        self.language = Language(tsc.language())
        self.parser = Parser(self.language)

    @property
    def language_id(self) -> str:
        return "c"

    def get_query(self) -> str:
        return """
        (function_definition) @function.def

        (struct_specifier
          body: (field_declaration_list)) @struct.def
        
        (type_definition
          type: (struct_specifier)) @struct.def

        (enum_specifier
          body: (enumerator_list)) @enum.def
        
        (type_definition
          type: (enum_specifier)) @enum.def
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
        
        all_captures = []
        for tag, nodes in captures_dict.items():
            for node in nodes:
                all_captures.append((node, tag))
        
        # Sort by start byte (ascending) and then by end byte (descending) 
        # to process larger nodes (like type_definition) before their children (like struct_specifier)
        all_captures.sort(key=lambda x: (x[0].start_byte, -x[0].end_byte))

        snippets = []
        processed_ranges = []

        for node, tag in all_captures:
            is_nested = False
            for start, end in processed_ranges:
                if node.start_byte >= start and node.end_byte <= end:
                    is_nested = True
                    break
            
            if is_nested:
                continue

            snippet = self._extract_snippet(node, tag, code, file_path)
            if snippet:
                snippets.append(snippet)
                processed_ranges.append((node.start_byte, node.end_byte))
        
        return snippets

    def _extract_snippet(self, node, tag, code, file_path) -> CodeSnippet:
        if tag == "function.def":
            snippet_type = SnippetType.FUNCTION
        else:
            snippet_type = SnippetType.STRUCT 

        name = "anonymous"
        
        if snippet_type == SnippetType.FUNCTION:
            decl = node.child_by_field_name("declarator")
            if decl:
                def find_identifier(n):
                    if n.type == "identifier":
                        return n
                    for child in n.children:
                        res = find_identifier(child)
                        if res:
                            return res
                    return None
                
                name_node = find_identifier(decl)
                if name_node:
                    name = code[name_node.start_byte:name_node.end_byte]
        else:
            # Struct / Enum
            if node.type == "type_definition":
                name_node = node.child_by_field_name("declarator")
                if name_node:
                    name = code[name_node.start_byte:name_node.end_byte]
            else:
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = code[name_node.start_byte:name_node.end_byte]

        # Signature
        if snippet_type == SnippetType.FUNCTION:
            decl_node = node.child_by_field_name("declarator")
            if decl_node:
                signature = code[decl_node.start_byte:decl_node.end_byte]
            else:
                signature = name
        else:
            signature = name

        # Docstring / Comments
        comments = []
        prev = node.prev_sibling
        while prev:
            if prev.type == "comment":
                comments.append(code[prev.start_byte:prev.end_byte].strip("/ ").strip("*").strip())
                prev = prev.prev_sibling
            elif prev.type in ["\n", " "]: 
                prev = prev.prev_sibling
            else:
                break
        
        docstring = "\n".join(reversed(comments)) if comments else None

        snippet_content = code[node.start_byte:node.end_byte]
        snippet_id = hashlib.sha256(snippet_content.encode("utf-8")).hexdigest()

        return CodeSnippet(
            id=snippet_id,
            name=name,
            type=snippet_type,
            content=snippet_content,
            parent_id=None,  
            docstring=docstring,
            signature=signature,
            file_path=file_path,
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            start_byte=node.start_byte,
            end_byte=node.end_byte
        )
