from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from enum import Enum

class SnippetType(Enum):
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    STRUCT = "struct"
    MODULE = "module"

@dataclass
class CodeSnippet:
    id: str
    name: str
    type: SnippetType
    content: str
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
