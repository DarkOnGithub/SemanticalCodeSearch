import logging
from typing import List
from tree_sitter import Query, QueryCursor
from src.graph.base_extractor import BaseRelationshipExtractor
from src.IR.models import Relationship, RelationType, CodeSnippet
import hashlib

logger = logging.getLogger(__name__)

class CRelationshipExtractor(BaseRelationshipExtractor):
    def get_query(self) -> str:
        return """
        (function_definition) @function.def
        
        (struct_specifier
          name: (type_identifier)? @struct.name
          body: (field_declaration_list)?) @struct.def
          
        (type_definition
          type: (struct_specifier)
          declarator: (type_identifier) @struct.name) @struct.typedef

        (preproc_include
          path: [
            (string_literal)
            (system_lib_string)
          ] @import.path) @import
          
        (call_expression
          function: (identifier) @call.name) @call
          
        (assignment_expression
          left: [(identifier) (field_expression)] @assignment.target) @assignment
        """

    def extract(self) -> List[Relationship]:
        query = Query(self.tree.language, self.get_query())
        cursor = QueryCursor(query)
        captures_dict = cursor.captures(self.tree.root_node)
        
        captures = []
        for tag, nodes in captures_dict.items():
            for node in nodes:
                captures.append((node, tag))
        
        relationships = []
        file_content_hash = hashlib.sha256(self.code.encode("utf-8")).hexdigest()
        
        for node, tag in captures:
            # 1. DEFINES
            if tag in ["function.def", "struct.def", "struct.typedef"]:
                snippet_id = self.range_to_snippet_id.get((node.start_byte, node.end_byte))
                if snippet_id:
                    if node.parent == self.tree.root_node:
                        relationships.append(Relationship(
                            source_id=file_content_hash,
                            target_id=snippet_id,
                            type=RelationType.DEFINES
                        ))
                    
                    parent_id = self.find_containing_snippet_id(node.parent)
                    if parent_id and parent_id != snippet_id:
                        relationships.append(Relationship(
                            source_id=parent_id,
                            target_id=snippet_id,
                            type=RelationType.DEFINES
                        ))

            # 2. CALLS
            elif tag == "call":
                container_id = self.find_containing_snippet_id(node)
                if container_id:
                    func_node = node.child_by_field_name("function")
                    if func_node:
                        call_name = self.code[func_node.start_byte:func_node.end_byte]
                        target_id = self.resolve_symbol(call_name)
                        relationships.append(Relationship(
                            source_id=container_id,
                            target_id=target_id,
                            type=RelationType.CALLS
                        ))

            # 3. IMPORTS (Includes)
            elif tag == "import.path":
                include_path = self.code[node.start_byte:node.end_byte].strip('"<>')
                target_id = self.resolve_symbol(include_path)
                relationships.append(Relationship(
                    source_id=file_content_hash,
                    target_id=target_id,
                    type=RelationType.IMPORTS
                ))

            # 8. MODIFIES
            elif tag == "assignment.target":
                container_id = self.find_containing_snippet_id(node)
                if container_id:
                    target_name = self.code[node.start_byte:node.end_byte]
                    # Track field modifications or likely global modifications
                    if "->" in target_name or "." in target_name or target_name.isupper():
                        relationships.append(Relationship(
                            source_id=container_id,
                            target_id=target_name,
                            type=RelationType.MODIFIES
                        ))

        return relationships

