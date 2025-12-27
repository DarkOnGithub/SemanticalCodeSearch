import os
import hashlib
import logging
from dataclasses import dataclass
from typing import List, Optional
from src.parsers.factory import ParserFactory
from src.storage.sqlite_storage import SQLiteStorage
from src.storage.neo4j_storage import Neo4jStorage
from src.graph.manager import GraphManager
from src.IR.models import CodeSnippet, Relationship, SnippetType

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
    def __init__(self, src_path: str, chunk_size: int = 1000, 
                 neo4j_uri: str = "bolt://localhost:7687", 
                 neo4j_user: str = "neo4j", 
                 neo4j_password: str = "password",
                 auto_start_neo4j: bool = True):
        self.src_path = os.path.abspath(src_path)
        self.chunk_size = chunk_size
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.auto_start_neo4j = auto_start_neo4j
        
        self.context = self._create_context(self.src_path)
        self.factory = ParserFactory(chunk_size=chunk_size)
        self.graph_manager = GraphManager(self.factory)
        
        self.sqlite: Optional[SQLiteStorage] = None
        self.neo4j: Optional[Neo4jStorage] = None

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
        self.neo4j = Neo4jStorage(
            uri=self.neo4j_uri, 
            user=self.neo4j_user, 
            password=self.neo4j_password,
            auto_start=self.auto_start_neo4j
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
        if not self.sqlite or not self.neo4j:
            logger.error("Attempted to save before storage initialization")
            raise RuntimeError("Storage not initialized. Call initialize_storage() first.")
            
        logger.info("Saving data to SQLite and Neo4j...")
        self.sqlite.save_snippets(snippets)
        self.neo4j.save_snippets(snippets)
        self.neo4j.save_relationships(relationships)
        logger.info("Save completed successfully")

    def cleanup(self, current_snippets: List[CodeSnippet]):
        """Removes data for files that no longer exist on disk."""
        if not self.sqlite or not self.neo4j:
            logger.warning("Cleanup skipped: storage not initialized")
            return

        logger.info("Starting cleanup pass...")
        current_files = {s.file_path for s in current_snippets if s.file_path}
        
        # SQLite Cleanup
        old_sqlite = set(self.sqlite.get_all_file_paths())
        for f in (old_sqlite - current_files):
            logger.info(f"Removing deleted file from SQLite: {f}")
            self.sqlite.delete_file_snippets(f)
            
        # Neo4j Cleanup
        old_neo4j = set(self.neo4j.get_all_file_paths())
        for f in (old_neo4j - current_files):
            logger.info(f"Removing deleted file from Neo4j: {f}")
            self.neo4j.delete_file_data(f)
        
        logger.info("Cleanup completed")

    def verify(self):
        """Prints a summary of the current project state in storage."""
        if not self.neo4j:
            return
        db_snippets = self.neo4j.get_all_snippets()
        placeholders = [s for s in db_snippets if s.type == SnippetType.PLACEHOLDER]
        
        logger.info(f"Project Index Summary [{self.context.project_id}]:")
        logger.info(f"  - Total Nodes: {len(db_snippets)}")
        logger.info(f"  - Local Snippets: {len(db_snippets) - len(placeholders)}")
        logger.info(f"  - External Symbols: {len(placeholders)}")
