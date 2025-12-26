from typing import List
from tree_sitter import Query, QueryCursor
from src.graph.base_extractor import BaseRelationshipExtractor
from src.IR.models import Relationship, RelationType
import hashlib

class PythonRelationshipExtractor(BaseRelationshipExtractor):
    def get_query(self) -> str:
        return """
        (class_definition
          name: (identifier) @class.name
          superclasses: (argument_list)? @class.bases) @class.def

        (function_definition
          name: (identifier) @function.name
          return_type: (type)? @function.return_type) @function.def
        
        (decorated_definition
          (decorator) @decorator
          definition: [
            (class_definition)
            (function_definition)
          ] @decorated.def) @decorated_definition
        
        (import_statement) @import
        (import_from_statement) @import
        
        (call
          function: [(identifier) @call.name (attribute) @call.attr]) @call
        
        (assignment
          left: [(identifier) (attribute)] @assignment.target) @assignment
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
            # 1. DEFINES (File -> Class/Function)
            if tag in ["class.def", "function.def"]:
                snippet_id = self.range_to_snippet_id.get((node.start_byte, node.end_byte))
                if snippet_id:
                    # If it's a top-level definition, file defines it
                    if node.parent == self.tree.root_node:
                        relationships.append(Relationship(
                            source_id=file_content_hash,
                            target_id=snippet_id,
                            type=RelationType.DEFINES
                        ))
                    
                    # If it has a parent class/function, that defines it
                    parent_snippet_id = self.find_containing_snippet_id(node.parent)
                    if parent_snippet_id and parent_snippet_id != snippet_id:
                        relationships.append(Relationship(
                            source_id=parent_snippet_id,
                            target_id=snippet_id,
                            type=RelationType.DEFINES
                        ))

            # 2. CALLS & 9. INSTANTIATES
            elif tag == "call":
                container_id = self.find_containing_snippet_id(node)
                if container_id:
                    func_node = node.child_by_field_name("function")
                    call_name = self.code[func_node.start_byte:func_node.end_byte]
                    
                    target_id = self.resolve_symbol(call_name)
                    
                    rel_type = RelationType.CALLS
                    # Heuristic for instantiation: if it starts with Uppercase
                    if call_name[0].isupper() and "." not in call_name:
                        rel_type = RelationType.INSTANTIATES
                    
                    relationships.append(Relationship(
                        source_id=container_id,
                        target_id=target_id,
                        type=rel_type,
                        metadata={"call_name": call_name}
                    ))

            # 3. IMPORTS
            elif tag == "import":
                import_text = self.code[node.start_byte:node.end_byte].strip()
                target_id = self.resolve_symbol(import_text)
                relationships.append(Relationship(
                    source_id=file_content_hash,
                    target_id=target_id,
                    type=RelationType.IMPORTS
                ))

            # 4. INHERITS
            elif tag == "class.bases":
                class_node = node.parent
                class_snippet_id = self.range_to_snippet_id.get((class_node.start_byte, class_node.end_byte))
                if class_snippet_id:
                    bases_text = self.code[node.start_byte:node.end_byte].strip("()")
                    for base in bases_text.split(","):
                        base = base.strip()
                        if base:
                            target_id = self.resolve_symbol(base)
                            relationships.append(Relationship(
                                source_id=class_snippet_id,
                                target_id=target_id,
                                type=RelationType.INHERITS
                            ))

            # 6. RETURNS
            elif tag == "function.return_type":
                func_node = node.parent
                func_snippet_id = self.range_to_snippet_id.get((func_node.start_byte, func_node.end_byte))
                if func_snippet_id:
                    return_type = self.code[node.start_byte:node.end_byte].strip()
                    target_id = self.resolve_symbol(return_type)
                    relationships.append(Relationship(
                        source_id=func_snippet_id,
                        target_id=target_id,
                        type=RelationType.RETURNS
                    ))

            # 7. DECORATED_BY
            elif tag == "decorator":
                parent_decorated = node.parent # decorated_definition
                if parent_decorated:
                    def_node = parent_decorated.child_by_field_name("definition")
                    if def_node:
                        snippet_id = self.range_to_snippet_id.get((def_node.start_byte, def_node.end_byte))
                        if snippet_id:
                            decorator_name = self.code[node.start_byte:node.end_byte].strip("@")
                            target_id = self.resolve_symbol(decorator_name)
                            relationships.append(Relationship(
                                source_id=snippet_id,
                                target_id=target_id,
                                type=RelationType.DECORATED_BY
                            ))

            # 8. MODIFIES (Writes)
            elif tag == "assignment.target":
                container_id = self.find_containing_snippet_id(node)
                if container_id:
                    target_name = self.code[node.start_byte:node.end_byte]
                    # Heuristic: track mutations to globals (UPPERCASE) or members (self.attr)
                    if target_name.isupper() or "." in target_name:
                        relationships.append(Relationship(
                            source_id=container_id,
                            target_id=target_name,
                            type=RelationType.MODIFIES
                        ))

        return relationships

