import os
import hashlib
import logging
import threading
import queue
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Set
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

from tqdm import tqdm

from src.parsers.factory import ParserFactory
from src.storage.sqlite_storage import SQLiteStorage
from src.storage.falkordb_storage import FalkorDBStorage
from src.storage.chroma_storage import ChromaStorage
from src.graph.manager import GraphManager
from src.IR.models import CodeSnippet, Relationship, SnippetType
from src.model.LLM import get_llm
from src.model.embedding import get_embedding_model

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
    def __init__(self, src_path: str, chunk_size: int = 1000, max_workers: int = 10, disable_summary: bool = False):
        self.src_path = os.path.abspath(src_path)
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        self.disable_summary = disable_summary
        
        self.context = self._create_context(self.src_path)
        self.llm = get_llm()
        self.embedding_model = get_embedding_model()
        self.factory = ParserFactory(chunk_size=chunk_size, llm=self.llm)
        self.graph_manager = GraphManager(self.factory)
        
        self.sqlite: Optional[SQLiteStorage] = None
        self.graph_db: Optional[FalkorDBStorage] = None
        self.chroma: Optional[ChromaStorage] = None
        
        self.changed_files: Set[str] = set()
        self.all_encountered_files: Dict[str, str] = {}
        self._embedding_cache: Dict[str, Any] = {}
        self._embedding_queue: queue.Queue = queue.Queue()

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
        os.makedirs(self.context.data_dir, exist_ok=True)
        
        self.sqlite = SQLiteStorage(self.context.sqlite_path)
        self.graph_db = FalkorDBStorage(
            db_path=os.path.join(self.context.data_dir, "graph.db"),
            graph_name=self.context.project_id
        )
        self.chroma = ChromaStorage(
            path=os.path.join(self.context.data_dir, "chroma"),
            collection_name=self.context.project_id
        )

    def extract_snippets(self) -> List[CodeSnippet]:
        """Pass 1: Parses files into discrete code snippets, skipping unchanged files."""
        logger.info(f"Pass 1: Extracting snippets from {self.src_path}")
        self.changed_files.clear()
        self.all_encountered_files.clear()
        
        def should_parse_callback(file_path: str, content_hash: str):
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
        self._link_orphan_snippets(snippets)
        
        logger.info(f"Successfully extracted {len(snippets)} snippets")
        return snippets

    def extract_relationships(self, snippets: List[CodeSnippet]) -> List[Relationship]:
        """Pass 2: Builds the semantic graph between snippets, skipping unchanged files."""
        logger.info("Pass 2: Extracting semantic relationships")
        relationships = self.graph_manager.build_graph(snippets, changed_files=self.changed_files)
        logger.info(f"Successfully extracted {len(relationships)} relationships")
        return relationships

    def summarize_snippets(self, snippets: List[CodeSnippet], batch_size: int = 5, embed_batch_size: int = 1):
        """Pass 3: Generates semantic summaries and pipelines embeddings to the GPU."""
        
        if self.disable_summary:
            logger.info("Pass 3: Summarization disabled. Pipelining embeddings.")
            self._process_embeddings([s for s in snippets if s.file_path in self.changed_files], embed_batch_size)
            return

        if not self.llm:
            logger.warning("LLM not initialized, skipping summarization")
            return

        logger.info(f"Pass 3: Generating hierarchical summaries (LLM batch={batch_size})")
        

        # Deduplicate snippets to prevent graph cycles
        unique_snippets = {s.id: s for s in snippets}.values()
        snippets = list(unique_snippets)
        
        # Build dependency graph (Parent -> Children)
        id_to_snippet = {s.id: s for s in snippets}
        parent_to_children = {}
        in_degree = {s.id: 0 for s in snippets}
        needs_llm = {s.id for s in snippets if s.file_path in self.changed_files or not s.summary}

        for s in snippets:
            if s.parent_id and s.parent_id in id_to_snippet:
                if s.parent_id == s.id:
                    continue 
                parent_to_children.setdefault(s.parent_id, []).append(s.id)
                in_degree[s.parent_id] += 1

        embed_thread = threading.Thread(
            target=self._run_embedding_worker, 
            args=(embed_batch_size,), 
            daemon=True
        )
        embed_thread.start()

        # 2. Bottom-Up Summarization
        lock = threading.Lock()
        ready_pool = [s.id for s in snippets if in_degree[s.id] == 0]
        processed_count = 0
        summarized_ids = set()

        def process_batch(batch_ids):
            to_summarize = []
            child_context = {}
            
            with lock:
                for sid in batch_ids:
                    # If this node OR any child changed, we must re-summarize
                    children = parent_to_children.get(sid, [])
                    if sid in needs_llm or any(cid in needs_llm for cid in children):
                        needs_llm.add(sid)
                        to_summarize.append(id_to_snippet[sid])
                        child_context[sid] = [
                            f"{id_to_snippet[c].name}: {id_to_snippet[c].summary}" 
                            for c in children if id_to_snippet.get(c) and id_to_snippet[c].summary
                        ]
            
            if to_summarize:
                try:
                    self.llm.summarize_batch(to_summarize, child_context)
                except Exception as e:
                    logger.error(f"Batch summarization failed: {e}")
            return batch_ids

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            pbar = tqdm(total=len(snippets), desc="Summarizing", unit="snippet")
            
            while processed_count < len(snippets):
                with lock:
                    while ready_pool:
                        batch = ready_pool[:batch_size]
                        ready_pool = ready_pool[batch_size:]
                        futures.append(executor.submit(process_batch, batch))

                if not futures:
                    break # Cycle detected or done

                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                futures = list(futures) # Convert set back to list

                for f in done:
                    try:
                        for sid in f.result():
                            if sid not in summarized_ids:
                                summarized_ids.add(sid)
                                processed_count += 1
                                pbar.update(1)
                                
                                # Unlock parent
                                s = id_to_snippet[sid]
                                if s.parent_id and s.parent_id in id_to_snippet:
                                    pid = s.parent_id
                                    in_degree[pid] -= 1
                                    if in_degree[pid] == 0:
                                        ready_pool.append(pid)
                    except Exception as e:
                        logger.error(f"Error in future result: {e}")
            pbar.close()

        # 3. Top-Down Context Propagation
        self._propagate_context(snippets, id_to_snippet)

        # 4. Pipeline Embeddings for Changed Files
        self._process_embeddings([s for s in snippets if s.file_path in self.changed_files], embed_batch_size, thread_started=True, thread_obj=embed_thread)
        
        logger.info("Summarization and Contextual Embedding completed")

    def embed_snippets(self, snippets: List[CodeSnippet], batch_size: int = 1, use_summary: Optional[bool] = None) -> List[Optional[Any]]:
        """Pass 4: Generates semantic embeddings. Uses cache if pipelining was used."""
        if use_summary is None:
            use_summary = not self.disable_summary

        if not self.embedding_model:
            return [None] * len(snippets)

        logger.info(f"Pass 4: Resolving embeddings for {len(snippets)} snippets")
        
        results = [None] * len(snippets)
        to_embed = []
        to_embed_indices = []

        for i, snippet in enumerate(snippets):
            if snippet.id in self._embedding_cache:
                results[i] = self._embedding_cache[snippet.id]
            elif snippet.file_path in self.changed_files:
                to_embed.append(snippet)
                to_embed_indices.append(i)

        if to_embed:
            logger.info(f"Embedding {len(to_embed)} remaining snippets")
            try:
                embeddings = self.embedding_model.embed_snippets(to_embed, batch_size=batch_size, use_summary=use_summary)
                for idx, emb in zip(to_embed_indices, embeddings):
                    results[idx] = emb
                    self._embedding_cache[snippets[idx].id] = emb
            except Exception as e:
                logger.error(f"Final embedding pass failed: {e}")

        if hasattr(self.embedding_model, 'clear_cache'):
            self.embedding_model.clear_cache()
            
        return results

    def save(self, snippets: List[CodeSnippet], relationships: List[Relationship], embeddings: Optional[List[Any]] = None):
        """Persists extracted data to both relational and graph storage."""
        if not all([self.sqlite, self.graph_db, self.chroma]):
            raise RuntimeError("Storage not initialized.")
            
        logger.info("Saving data to SQLite, FalkorDB, and ChromaDB...")
        
        # Clean old data for changed files
        for file_path in self.changed_files:
            self.sqlite.delete_file_snippets(file_path)
            self.graph_db.delete_file_data(file_path)
            self.chroma.delete_file_snippets(file_path)

        # Save snippets (Changed only)
        changed_snippets = [s for s in snippets if s.file_path in self.changed_files]
        if changed_snippets:
            self.sqlite.save_snippets(changed_snippets)
            self.graph_db.save_snippets(changed_snippets)
            
            if embeddings:
                # Align embeddings with changed snippets
                valid_pairs = [
                    (s, embeddings[i]) 
                    for i, s in enumerate(snippets) 
                    if s.file_path in self.changed_files and embeddings[i] is not None
                ]
                if valid_pairs:
                    v_snips, v_embs = zip(*valid_pairs)
                    self.chroma.save_snippets(list(v_snips), list(v_embs))

        # Always save all relationships for complete graph connectivity
        self.graph_db.save_relationships(relationships)
        
        # Save hashes
        for f_path, c_hash in self.all_encountered_files.items():
            self.sqlite.save_file_hash(f_path, c_hash)
                
        logger.info("Save completed successfully")

    def cleanup(self, current_snippets: List[CodeSnippet]):
        """Removes data for files that no longer exist on disk."""
        if not self.sqlite: return

        logger.info("Starting cleanup pass...")
        current_files = {s.file_path for s in current_snippets if s.file_path}
        
        for storage, name in [(self.sqlite, "SQLite"), (self.graph_db, "FalkorDB"), (self.chroma, "ChromaDB")]:
            stored_files = set(storage.get_all_file_paths())
            for f in (stored_files - current_files):
                logger.info(f"Removing deleted file from {name}: {f}")
                if hasattr(storage, 'delete_file_data'):
                    storage.delete_file_data(f)
                else:
                    storage.delete_file_snippets(f)

    def verify(self):
        """Prints a summary of the current project state in storage."""
        if not self.graph_db: return
        
        nodes = self.graph_db.get_all_nodes()
        placeholders = sum(1 for n in nodes if n.type == SnippetType.PLACEHOLDER)
        
        logger.info(f"Project Index Summary [{self.context.project_id}]:")
        logger.info(f"  - Total Nodes: {len(nodes)}")
        logger.info(f"  - Local Snippets: {len(nodes) - placeholders}")
        logger.info(f"  - External Symbols: {placeholders}")

    # --- Private Helpers ---

    def _link_orphan_snippets(self, snippets: List[CodeSnippet]):
        """Ensures non-file snippets are linked to their parent file if missing a structural parent."""
        file_id_map = {s.file_path: s.id for s in snippets if s.type == SnippetType.FILE}
        id_map = {s.id: s for s in snippets}
        
        for s in snippets:
            if s.type != SnippetType.FILE and s.file_path in file_id_map:
                if not s.parent_id or s.parent_id not in id_map:
                    s.parent_id = file_id_map[s.file_path]
            
            # inherit signature for context
            if s.parent_id and s.parent_id in id_map:
                parent = id_map[s.parent_id]
                s.metadata["parent_signature"] = parent.signature or parent.name

    def _propagate_context(self, snippets: List[CodeSnippet], id_map: Dict[str, CodeSnippet]):
        """Propagates summaries top-down (File -> Class -> Method)."""
        logger.info("Propagating summaries top-down...")
        for _ in range(3): # Sufficient depth for standard code nesting
            for s in snippets:
                if s.parent_id and s.parent_id in id_map:
                    parent = id_map[s.parent_id]
                    if parent.summary and not s.metadata.get("parent_summary"):
                        s.metadata["parent_summary"] = parent.summary

    def _process_embeddings(self, snippets: List[CodeSnippet], batch_size: int, thread_started: bool = False, thread_obj: Optional[threading.Thread] = None):
        """Orchestrates the background embedding worker."""
        if not thread_started:
            thread_obj = threading.Thread(target=self._run_embedding_worker, args=(batch_size,), daemon=True)
            thread_obj.start()

        pbar = tqdm(total=len(snippets), desc="Pipelining Embeddings", unit="snippet")
        for s in snippets:
            self._embedding_queue.put(s)
            pbar.update(0) # Keep pbar alive
        
        self._embedding_queue.put(None) # Sentinel
        if thread_obj:
            thread_obj.join()
        pbar.close()

    def _run_embedding_worker(self, batch_size: int):
        """Consumer loop for the embedding queue."""
        while True:
            batch = self._collect_batch(batch_size)
            if not batch: break
            
            try:
                # use_summary=True default assumed for context
                embeddings = self.embedding_model.embed_snippets(batch, batch_size=len(batch))
                for s, emb in zip(batch, embeddings):
                    self._embedding_cache[s.id] = emb
            except Exception as e:
                logger.error(f"Embedding pipeline error: {e}")
            finally:
                for _ in batch:
                    self._embedding_queue.task_done()

    def _collect_batch(self, size: int) -> List[CodeSnippet]:
        """Helper to gather a batch from the queue with a sentinel check."""
        batch = []
        try:
            item = self._embedding_queue.get()
            if item is None: return [] # Sentinel received
            batch.append(item)
            
            while len(batch) < size:
                try:
                    item = self._embedding_queue.get_nowait()
                    if item is None:
                        self._embedding_queue.put(None) # Re-queue sentinel
                        break
                    batch.append(item)
                except queue.Empty:
                    break
        except Exception:
            return []
        return batch