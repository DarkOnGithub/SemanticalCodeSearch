import os
import logging
from typing import Dict, Optional, List, Any
from src.parsers.base_parser import BaseParser
from src.parsers.python_parser import PythonParser
from src.parsers.c_parser import CParser
from src.IR.models import CodeSnippet

logger = logging.getLogger(__name__)

class ParserFactory:
    def __init__(self, chunk_size: int = 500, llm: Optional[Any] = None):
        self.chunk_size = chunk_size
        self._parsers: Dict[str, BaseParser] = {
            "py": PythonParser(chunk_size=chunk_size, llm=llm),
            "c": CParser(chunk_size=chunk_size, llm=llm),
            "h": CParser(chunk_size=chunk_size, llm=llm),
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

    def parse_directory(
        self, 
        directory_path: str, 
        recursive: bool = False,
        ignore_dirs: Optional[List[str]] = None,
        ignore_exts: Optional[List[str]] = None,
        should_parse_callback: Optional[callable] = None
    ) -> List[CodeSnippet]:
        """Parses all supported files in a directory"""
        all_snippets = []
        
        default_ignore_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv"}
        default_ignore_exts = {".pyc", ".pyo", ".so", ".dll", ".exe", ".bin"}
            
        ignore_dirs_set = default_ignore_dirs.union(set(ignore_dirs)) if ignore_dirs else default_ignore_dirs
        ignore_exts_set = default_ignore_exts.union(set(ignore_exts)) if ignore_exts else default_ignore_exts

        for root, dirs, files in os.walk(directory_path):
            dirs[:] = [d for d in dirs if d not in ignore_dirs_set]
            
            for file in files:
                file_path = os.path.join(root, file)
                extension = f".{file.split('.')[-1]}" if "." in file else ""
                
                if extension in ignore_exts_set:
                    continue
                    
                parser = self.get_parser_for_file(file_path)
                
                if parser:
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        import hashlib
                        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                        
                        snippets = None
                        if should_parse_callback:
                            snippets = should_parse_callback(file_path, content_hash)
                        
                        if snippets is None:
                            logger.info(f"Parsing file: {file_path}")
                            snippets = parser.parse_file(content, file_path)
                        else:
                            logger.info(f"Skipping parsing for unchanged file: {file_path}")
                            
                        all_snippets.extend(snippets)
                    except Exception as e:
                        logger.error(f"Error parsing {file_path}: {e}")
            
            if not recursive:
                break
                
        return all_snippets
