import os
import hashlib
import logging
from dataclasses import dataclass
from typing import List, Optional
from src.parsers.factory import ParserFactory
from src.storage.sqlite_storage import SQLiteStorage
from src.storage.falkordb_storage import FalkorDBStorage
from src.graph.manager import GraphManager
from src.IR.models import CodeSnippet, Relationship, SnippetType
from src.model.LLM import get_llm

logger = logging.getLogger(__name__)

@dataclass
class ProjectContext:
    src_path: str
    project_id: str
    data_dir: str
    sqlite_path: str

class ProjectIndexer:
    """
    Handles indexing for a single specific folder/workspace.
    All operations are manual, allowing for fine-grained control.
    """
    def __init__(self, src_path: str, chunk_size: int = 1000):
        self.src_path = os.path.abspath(src_path)
        self.chunk_size = chunk_size
        
        self.context = self._create_context(self.src_path)
        self.llm = get_llm()
        self.factory = ParserFactory(chunk_size=chunk_size, llm=self.llm)
        self.graph_manager = GraphManager(self.factory)
        
        self.sqlite: Optional[SQLiteStorage] = None
        self.graph_db: Optional[FalkorDBStorage] = None

    def _create_context(self, path: str) -> ProjectContext:
        folder_name = os.path.basename(path.rstrip(os.sep))
        path_hash = hashlib.md5(path.encode()).hexdigest()[:8]
        project_id = f"{folder_name}_{path_hash}"
        data_dir = os.path.join("data", "projects", project_id)
        
        return ProjectContext(
            src_path=path,
            project_id=project_id,
            data_dir=data_dir,
            sqlite_path=os.path.join(data_dir, "codebase.db")
        )

    def initialize_storage(self):
        """Prepares directories and establishes database connections."""
        logger.info(f"Initializing storage for project: {self.context.project_id}")
        logger.debug(f"Data directory: {self.context.data_dir}")
        os.makedirs(self.context.data_dir, exist_ok=True)
        self.sqlite = SQLiteStorage(self.context.sqlite_path)
        self.graph_db = FalkorDBStorage(
            db_path=os.path.join(self.context.data_dir, "graph.db"),
            graph_name=self.context.project_id
        )

    def extract_snippets(self) -> List[CodeSnippet]:
        """Pass 1: Parses files into discrete code snippets."""
        logger.info(f"Pass 1: Extracting snippets from {self.src_path}")
        snippets = self.factory.parse_directory(self.src_path, recursive=True)
        
        file_snippets = self.graph_manager.create_file_snippets(snippets)
        snippets.extend(file_snippets)
        
        logger.info(f"Successfully extracted {len(snippets)} snippets")
        return snippets

    def extract_relationships(self, snippets: List[CodeSnippet]) -> List[Relationship]:
        """Pass 2: Builds the semantic graph between snippets."""
        logger.info("Pass 2: Extracting semantic relationships")
        relationships = self.graph_manager.build_graph(snippets)
        logger.info(f"Successfully extracted {len(relationships)} relationships")
        return relationships

    def save(self, snippets: List[CodeSnippet], relationships: List[Relationship]):
        """Persists extracted data to both relational and graph storage."""
        if not self.sqlite or not self.graph_db:
            logger.error("Attempted to save before storage initialization")
            raise RuntimeError("Storage not initialized. Call initialize_storage() first.")
            
        logger.info("Saving data to SQLite and FalkorDB...")
        self.sqlite.save_snippets(snippets)
        self.graph_db.save_snippets(snippets)
        self.graph_db.save_relationships(relationships)
        logger.info("Save completed successfully")

    def cleanup(self, current_snippets: List[CodeSnippet]):
        """Removes data for files that no longer exist on disk."""
        if not self.sqlite or not self.graph_db:
            logger.warning("Cleanup skipped: storage not initialized")
            return

        logger.info("Starting cleanup pass...")
        current_files = {s.file_path for s in current_snippets if s.file_path}
        
        # SQLite Cleanup
        old_sqlite = set(self.sqlite.get_all_file_paths())
        for f in (old_sqlite - current_files):
            logger.info(f"Removing deleted file from SQLite: {f}")
            self.sqlite.delete_file_snippets(f)
            
        # FalkorDB Cleanup
        old_graph = set(self.graph_db.get_all_file_paths())
        for f in (old_graph - current_files):
            logger.info(f"Removing deleted file from FalkorDB: {f}")
            self.graph_db.delete_file_data(f)
        
        logger.info("Cleanup completed")

    def verify(self):
        """Prints a summary of the current project state in storage."""
        if not self.graph_db:
            return
        db_nodes = self.graph_db.get_all_nodes()
        placeholders = [s for s in db_nodes if s.type == SnippetType.PLACEHOLDER]
        
        logger.info(f"Project Index Summary [{self.context.project_id}]:")
        logger.info(f"  - Total Nodes: {len(db_nodes)}")
        logger.info(f"  - Local Snippets: {len(db_nodes) - len(placeholders)}")
        logger.info(f"  - External Symbols: {len(placeholders)}")
