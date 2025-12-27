from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from enum import Enum
import json

class SnippetType(Enum):
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    STRUCT = "struct"
    ENUM = "enum"
    MODULE = "module"
    FILE = "file"
    PLACEHOLDER = "placeholder"

class RelationType(Enum):
    DEFINES = "defines"
    CALLS = "calls"
    IMPORTS = "imports"
    INHERITS = "inherits"
    OVERRIDES = "overrides"
    RETURNS = "returns"
    DECORATED_BY = "decorated_by"
    MODIFIES = "modifies"
    INSTANTIATES = "instantiates"

@dataclass
class CodeSnippet:
    id: str
    name: str
    type: SnippetType
    content: str
    summary: Optional[str] = None
    parent_id: Optional[str] = None
    docstring: Optional[str] = None
    signature: Optional[str] = None
    file_path: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    start_byte: Optional[int] = None
    end_byte: Optional[int] = None
    is_skeleton: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_embeddable_text(self, use_summary: bool = True) -> str:
        """
        Constructs a string representation of the snippet for embedding and retrieval.
        Includes metadata like file path and name to provide more context.
        """
        file_info = f"File: {self.file_path}\n" if self.file_path else ""
        name_info = f"Name: {self.name}\n" if self.name else ""
        type_info = f"Type: {self.type.value}\n"
        context = f"{file_info}{name_info}{type_info}"

        if use_summary and self.summary:
            return f"{context}Summary: {self.summary}\n\nCode:\n{self.content}"
        else:
            return f"{context}Code:\n{self.content}"

    def __str__(self):
        lines = f"L{self.start_line + 1}-L{self.end_line + 1}" if self.start_line is not None else "???"
        skel = " [SKEL]" if self.is_skeleton else ""
        parent = f" parent={self.parent_id[:8]}..." if self.parent_id else ""
        file_name = self.file_path.split("/")[-1] if self.file_path else "unknown"
        return f"[{self.type.value.upper()}] {self.name} ({file_name}:{lines}) id={self.id[:8]}...{skel}{parent}"

    def __repr__(self):
        return self.__str__()

@dataclass
class Relationship:
    source_id: str
    target_id: str
    type: RelationType
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_tuple(self):
        return (self.source_id, self.target_id, self.type.value, json.dumps(self.metadata) if self.metadata else None)

    def __str__(self):
        return f"({self.source_id[:8]}...) --[{self.type.value.upper()}]--> ({self.target_id[:8]}...)"

    def __repr__(self):
        return self.__str__()

@dataclass
class GraphNode:
    id: str
    name: str
    type: SnippetType
    file_path: Optional[str] = None

    def __str__(self):
        file_info = f" in {self.file_path.split('/')[-1]}" if self.file_path else ""
        return f"Node({self.name}, {self.type.value}{file_info}, id={self.id[:8]}...)"

    def __repr__(self):
        return self.__str__()
