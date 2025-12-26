import os
from typing import Dict, Optional, List
from src.parsers.base_parser import BaseParser
from src.parsers.python_parser import PythonParser
from src.parsers.c_parser import CParser
from src.IR.models import CodeSnippet

#!TODO use magika to detect the language of the file and return the appropriate parser

class ParserFactory:
    def __init__(self):
        self._parsers: Dict[str, BaseParser] = {
            "py": PythonParser(),
            "c": CParser(),
            "h": CParser(),
        }

    def get_parser_for_extension(self, extension: str) -> Optional[BaseParser]:
        """Returns a parser instance based on the file extension"""
        ext = extension.lower().lstrip(".")
        return self._parsers.get(ext)

    def get_parser_for_file(self, file_path: str) -> Optional[BaseParser]:
        """Returns a parser instance based on the file path"""
        if "." not in file_path:
            return None
        extension = file_path.split(".")[-1]
        return self.get_parser_for_extension(extension)

    def parse_directory(self, directory_path: str, recursive: bool = False) -> List[CodeSnippet]:
        """Parses all supported files in a directory"""
        all_snippets = []
        
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                parser = self.get_parser_for_file(file_path)
                
                if parser:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        snippets = parser.parse_file(content, file_path)
                        all_snippets.extend(snippets)
                    except Exception as e:
                        print(f"Error parsing {file_path}: {e}")
                else:
                    print(f"No parser found for {file_path}")
            if not recursive:
                break
                
        return all_snippets
