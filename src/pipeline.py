import os
import time
import hashlib
from dataclasses import dataclass
from typing import List, Tuple
from src.parsers.factory import ParserFactory
from src.storage.sqlite_storage import SQLiteStorage
from src.storage.kuzu_storage import KuzuStorage
from src.graph.manager import GraphManager
from src.IR.models import CodeSnippet, Relationship, SnippetType

@dataclass
class ProjectContext:
    src_path: str
    project_id: str
    data_dir: str
    sqlite_path: str
    kuzu_path: str

class IndexingPipeline:
    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size
        self.factory = ParserFactory(chunk_size=chunk_size)
        self.graph_manager = GraphManager(self.factory)

    def _get_project_id(self, path: str) -> str:
        abs_path = os.path.abspath(path)
        folder_name = os.path.basename(abs_path.rstrip(os.sep))
        path_hash = hashlib.md5(abs_path.encode()).hexdigest()[:8]
        return f"{folder_name}_{path_hash}"

    def create_project_context(self, src_path: str) -> ProjectContext:
        src_path = os.path.abspath(src_path)
        project_id = self._get_project_id(src_path)
        data_dir = os.path.join("data", "projects", project_id)
        
        return ProjectContext(
            src_path=src_path,
            project_id=project_id,
            data_dir=data_dir,
            sqlite_path=os.path.join(data_dir, "codebase.db"),
            kuzu_path=os.path.join(data_dir, "graph")
        )

    def run(self, src_path: str):
        context = self.create_project_context(src_path)
        if not os.path.exists(context.src_path):
            print(f"Error: Path {context.src_path} does not exist.")
            return

        print(f"--- Starting Pipeline for Project: {context.project_id} ---")
        
        # 1. Initialize Storage
        sqlite, kuzu = self.initialize_storage(context)
        
        # 2. Pass 1: Parsing
        start_time = time.time()
        snippets = self.extract_snippets(context)
        
        # 3. Pass 2: Relationships
        relationships = self.extract_relationships(snippets)
        
        # 4. Persistence
        self.save_all(sqlite, kuzu, snippets, relationships)
        
        # 5. Maintenance
        self.cleanup(sqlite, kuzu, snippets)
        
        # 6. Verification
        self.verify(kuzu, time.time() - start_time)

    def initialize_storage(self, context: ProjectContext) -> Tuple[SQLiteStorage, KuzuStorage]:
        os.makedirs(context.data_dir, exist_ok=True)
        sqlite = SQLiteStorage(context.sqlite_path)
        kuzu = KuzuStorage(context.kuzu_path)
        return sqlite, kuzu

    def extract_snippets(self, context: ProjectContext) -> List[CodeSnippet]:
        print("--- Pass 1: Parsing Snippets ---")
        snippets = self.factory.parse_directory(context.src_path, recursive=True)
        
        # Add file-level nodes
        file_snippets = self.graph_manager.create_file_snippets(snippets)
        snippets.extend(file_snippets)
        
        print(f"Extracted {len(snippets)} snippets.")
        return snippets

    def extract_relationships(self, snippets: List[CodeSnippet]) -> List[Relationship]:
        print("--- Pass 2: Extracting Relationships ---")
        relationships = self.graph_manager.build_graph(snippets)
        print(f"Extracted {len(relationships)} relationships.")
        return relationships

    def save_all(self, sqlite: SQLiteStorage, kuzu: KuzuStorage, snippets: List[CodeSnippet], relationships: List[Relationship]):
        print("\nSaving to storage engines...")
        sqlite.save_snippets(snippets)
        kuzu.save_snippets(snippets)
        kuzu.save_relationships(relationships)
        print("Save complete.")

    def cleanup(self, sqlite: SQLiteStorage, kuzu: KuzuStorage, current_snippets: List[CodeSnippet]):
        print("\n--- Cleanup Pass: Removing Deleted Files ---")
        
        current_files = {s.file_path for s in current_snippets if s.file_path}
        
        # Cleanup SQLite
        old_sqlite = set(sqlite.get_all_file_paths())
        for f in (old_sqlite - current_files):
            print(f"  - Removing {f} from SQLite")
            sqlite.delete_file_snippets(f)
            
        # Cleanup Kuzu
        old_kuzu = set(kuzu.get_all_file_paths())
        for f in (old_kuzu - current_files):
            print(f"  - Removing {f} from Kuzu")
            kuzu.delete_file_data(f)
        
        print("Cleanup complete.")

    def verify(self, kuzu: KuzuStorage, duration: float):
        print("\n--- Pipeline Summary ---")
        db_snippets = kuzu.get_all_snippets()
        placeholders = [s for s in db_snippets if s.type == SnippetType.PLACEHOLDER]
        
        print(f"Total Graph Nodes: {len(db_snippets)}")
        print(f"  - Actual Snippets: {len(db_snippets) - len(placeholders)}")
        print(f"  - External Symbols: {len(placeholders)}")
        print(f"Time Taken: {duration:.2f} seconds\n")
