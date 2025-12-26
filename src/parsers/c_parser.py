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
        (function_definition
          declarator: (function_declarator
            declarator: (identifier) @function.name)
        ) @function.def

        (struct_specifier
          name: (type_identifier) @struct.name
          body: (field_declaration_list)
        ) @struct.def
        
        (enum_specifier
          name: (type_identifier) @enum.name
          body: (enumerator_list)
        ) @enum.def
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
        all_captures.sort(key=lambda x: x[0].start_byte)

        snippets = []
        for node, tag in all_captures:
            if tag in ["function.def", "struct.def", "enum.def"]:
                snippet = self._extract_snippet(node, tag, code, file_path)
                if snippet:
                    snippets.append(snippet)
        
        return snippets

    def _extract_snippet(self, node, tag, code, file_path) -> CodeSnippet:
        if tag == "function.def":
            snippet_type = SnippetType.FUNCTION
        elif tag == "struct.def":
            snippet_type = SnippetType.STRUCT
        else:
            snippet_type = SnippetType.STRUCT # Using STRUCT for enums for now or could add ENUM to SnippetType

        # Find name using Query capture or manual child search since C grammar structure varies
        name = "anonymous"
        
        # For function_definition, we look for function_declarator -> identifier
        # For struct_specifier, we look for type_identifier
        
        if snippet_type == SnippetType.FUNCTION:
            # Navigate to identifier: function_definition -> declarator (function_declarator) -> declarator (identifier)
            decl = node.child_by_field_name("declarator")
            if decl:
                if decl.type == "pointer_declarator":
                    decl = decl.child_by_field_name("declarator")
                
                if decl and decl.type == "function_declarator":
                    name_node = decl.child_by_field_name("declarator")
                    if name_node:
                        name = code[name_node.start_byte:name_node.end_byte]
        else:
            name_node = node.child_by_field_name("name")
            if name_node:
                name = code[name_node.start_byte:name_node.end_byte]

        # Signature
        if snippet_type == SnippetType.FUNCTION:
            # Find the whole declarator for signature
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
        # In C, comments might be separated by newlines which are sometimes extra nodes
        while prev:
            if prev.type == "comment":
                comments.append(code[prev.start_byte:prev.end_byte].strip("/ ").strip("*").strip())
                prev = prev.prev_sibling
            elif prev.type in ["\n", " "]: # Skip whitespace
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
            docstring=docstring,
            signature=signature,
            file_path=file_path,
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            start_byte=node.start_byte,
            end_byte=node.end_byte
        )

