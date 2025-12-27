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
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        lines = f"{self.start_line + 1}-{self.end_line + 1}" if self.start_line is not None else "unknown"
        return f"<{self.type.value.capitalize()} name='{self.name}' file='{self.file_path}' lines={lines}>"

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

@dataclass
class GraphNode:
    id: str
    name: str
    type: SnippetType
    file_path: Optional[str] = None
