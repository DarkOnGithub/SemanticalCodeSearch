import os
import hashlib
import logging
import threading
from dataclasses import dataclass
from typing import List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
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
    def __init__(self, src_path: str, chunk_size: int = 1000, max_workers: int = 10):
        self.src_path = os.path.abspath(src_path)
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        
        self.context = self._create_context(self.src_path)
        self.llm = get_llm()
        self.factory = ParserFactory(chunk_size=chunk_size, llm=self.llm)
        self.graph_manager = GraphManager(self.factory)
        
        self.sqlite: Optional[SQLiteStorage] = None
        self.graph_db: Optional[FalkorDBStorage] = None
        self.changed_files: set[str] = set()
        self.all_encountered_files: Dict[str, str] = {}

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
        """Pass 1: Parses files into discrete code snippets, skipping unchanged files."""
        logger.info(f"Pass 1: Extracting snippets from {self.src_path}")
        
        self.changed_files = set()
        self.all_encountered_files = {} 
        
        def should_parse_callback(file_path, content_hash):
            self.all_encountered_files[file_path] = content_hash
            
            if not self.sqlite:
                return None
            
            old_hash = self.sqlite.get_file_hash(file_path)
            if old_hash == content_hash:
                return self.sqlite.get_file_snippets(file_path)
            
            self.changed_files.add(file_path)
            return None

        snippets = self.factory.parse_directory(
            self.src_path, 
            recursive=True, 
            should_parse_callback=should_parse_callback
        )
        
        file_snippets = self.graph_manager.create_file_snippets(snippets, changed_files=self.changed_files)
        snippets.extend(file_snippets)
        
        # Link top-level snippets to their file snippets
        file_id_map = {s.file_path: s.id for s in snippets if s.type == SnippetType.FILE}
        id_to_snippet = {s.id: s for s in snippets}
        for s in snippets:
            if s.type != SnippetType.FILE and s.file_path in file_id_map:
                # If it has no parent or parent is stale (not in current snippets), link to FILE
                if not s.parent_id or s.parent_id not in id_to_snippet:
                    s.parent_id = file_id_map[s.file_path]
        
        logger.info(f"Successfully extracted {len(snippets)} snippets")
        return snippets

    def summarize_snippets(self, snippets: List[CodeSnippet], batch_size: int = 5):
        """Pass 3: Generates semantic summaries using a parallel batched bottom-up approach."""
        if not self.llm:
            logger.warning("LLM not initialized, skipping summarization")
            return

        # Ensure no duplicates in the snippets list (prevents deadlocks in in_degree logic)
        seen_ids = set()
        unique_snippets = []
        for s in snippets:
            if s.id not in seen_ids:
                seen_ids.add(s.id)
                unique_snippets.append(s)
        snippets = unique_snippets

        logger.info(f"Pass 3: Generating hierarchical summaries (batched, size={batch_size}, workers={self.max_workers})")
        
        id_to_snippet = {s.id: s for s in snippets}
        parent_to_children = {}
        in_degree = {s.id: 0 for s in snippets}
        
        # Track which snippets need re-summarization
        needs_llm = set()
        for s in snippets:
            if s.file_path in self.changed_files:
                needs_llm.add(s.id)
            elif not s.summary:
                needs_llm.add(s.id)
        
        for s in snippets:
            if s.parent_id and s.parent_id in id_to_snippet:
                # Prevent self-parent cycles
                if s.parent_id == s.id:
                    logger.warning(f"Self-parent detected for snippet {s.name} ({s.id[:8]}). Breaking cycle.")
                    s.parent_id = None
                    continue
                if s.parent_id not in parent_to_children:
                    parent_to_children[s.parent_id] = []
                parent_to_children[s.parent_id].append(s.id)
                in_degree[s.parent_id] += 1
        
        lock = threading.Lock()
        ready_pool = [s.id for s in snippets if in_degree[s.id] == 0]
        processed_count = 0
        summarized_ids = set()

        def do_batch(batch_ids):
            to_summarize = []
            child_summaries_map = {}
            
            with lock:
                # Decide which ones in this batch actually need LLM
                for sid in batch_ids:
                    snippet = id_to_snippet[sid]
                    child_ids = parent_to_children.get(sid, [])
                    # Parent needs update if its file changed OR if any child was updated
                    if sid in needs_llm or any(cid in needs_llm for cid in child_ids):
                        needs_llm.add(sid)
                        to_summarize.append(snippet)
                        child_summaries = []
                        for cid in child_ids:
                            child = id_to_snippet.get(cid)
                            if child and child.summary:
                                child_summaries.append(f"{child.name}: {child.summary}")
                        child_summaries_map[sid] = child_summaries

            if to_summarize:
                try:
                    snippet_names = [s.name for s in to_summarize]
                    logger.info(f"Summarizing batch of {len(to_summarize)}: {snippet_names}")
                    self.llm.summarize_batch(to_summarize, child_summaries_map)
                except Exception as e:
                    logger.error(f"Error in batch summarization thread: {e}")
            
            return batch_ids

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            while processed_count < len(snippets):
                with lock:
                    while ready_pool:
                        batch = ready_pool[:batch_size]
                        ready_pool = ready_pool[batch_size:]
                        futures.append(executor.submit(do_batch, batch))
                
                if not futures:
                    break
                
                done, remaining = wait(futures, return_when=FIRST_COMPLETED)
                futures = list(remaining)
                
                for f in done:
                    try:
                        batch_ids = f.result()
                        with lock:
                            for sid in batch_ids:
                                if sid not in summarized_ids:
                                    summarized_ids.add(sid)
                                    processed_count += 1
                                    snippet = id_to_snippet[sid]
                                    if snippet.parent_id and snippet.parent_id in id_to_snippet:
                                        pid = snippet.parent_id
                                        in_degree[pid] -= 1
                                        if in_degree[pid] == 0:
                                            ready_pool.append(pid)
                    except Exception as e:
                        logger.error(f"Batch processing error: {e}")

        unprocessed_ids = [sid for sid in id_to_snippet if sid not in summarized_ids]
        if unprocessed_ids:
            unprocessed_names = [id_to_snippet[sid].name for sid in unprocessed_ids[:5]]
            logger.warning(f"Summarization incomplete. {len(unprocessed_ids)} nodes were skipped (likely a cycle): {unprocessed_names}...")

        logger.info("Hierarchical summarization completed")

    def extract_relationships(self, snippets: List[CodeSnippet]) -> List[Relationship]:
        """Pass 2: Builds the semantic graph between snippets, skipping unchanged files."""
        logger.info("Pass 2: Extracting semantic relationships")
        
        # Only process files that have changed
        # We still need all snippets to build the symbol table for resolution
        relationships = self.graph_manager.build_graph(snippets, changed_files=self.changed_files)
        logger.info(f"Successfully extracted {len(relationships)} relationships")
        return relationships

    def save(self, snippets: List[CodeSnippet], relationships: List[Relationship]):
        """Persists extracted data to both relational and graph storage."""
        if not self.sqlite or not self.graph_db:
            logger.error("Attempted to save before storage initialization")
            raise RuntimeError("Storage not initialized. Call initialize_storage() first.")
            
        logger.info("Saving data to SQLite and FalkorDB...")
        
        # For changed files, delete old data first to ensure clean state (especially for relationships)
        for file_path in self.changed_files:
            logger.info(f"Updating changed file: {file_path}")
            self.sqlite.delete_file_snippets(file_path)
            self.graph_db.delete_file_data(file_path)

        self.sqlite.save_snippets(snippets)
        self.graph_db.save_snippets(snippets)
        self.graph_db.save_relationships(relationships)
        
        # Save file hashes for ALL encountered files, even those with 0 snippets
        for file_path, content_hash in self.all_encountered_files.items():
            self.sqlite.save_file_hash(file_path, content_hash)
                
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
