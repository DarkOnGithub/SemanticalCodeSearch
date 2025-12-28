import os
import logging
import chromadb
from typing import List, Optional, Dict, Any
from src.IR.models import CodeSnippet

logger = logging.getLogger(__name__)

class ChromaStorage:
    def __init__(self, path: str = "data/chroma", collection_name: str = "codebase"):
        os.makedirs(path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"ChromaDB initialized at {path}, collection: {collection_name} (cosine similarity)")

    def save_snippets(self, snippets: List[CodeSnippet], embeddings: List[Any]):
        """
        Saves snippets and their embeddings to ChromaDB.
        """
        if not snippets:
            return

        if len(snippets) != len(embeddings):
            raise ValueError("Number of snippets and embeddings must match")

        ids = [s.id for s in snippets]
        metadatas = []
        for s in snippets:
            meta = {
                "name": s.name,
                "type": s.type.value,
                "file_path": s.file_path or "",
                "start_line": s.start_line if s.start_line is not None else -1,
                "end_line": s.end_line if s.end_line is not None else -1,
                "is_skeleton": s.is_skeleton
            }
            metadatas.append(meta)
        
        documents = [s.to_embeddable_text(use_summary=True) for s in snippets]

        batch_size = 500
        for i in range(0, len(snippets), batch_size):
            end = min(i + batch_size, len(snippets))
            self.collection.upsert(
                ids=ids[i:end],
                embeddings=[e.tolist() if hasattr(e, "tolist") else e for e in embeddings[i:end]],
                metadatas=metadatas[i:end],
                documents=documents[i:end]
            )
        
        logger.info(f"Saved {len(snippets)} snippets to ChromaDB")

    def delete_file_snippets(self, file_path: str):
        """
        Deletes all snippets associated with a specific file path.
        """
        self.collection.delete(where={"file_path": file_path})
        logger.debug(f"Deleted snippets for file {file_path} from ChromaDB")

    def query(self, query_embedding: Any, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Searches for snippets similar to the query embedding.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist() if hasattr(query_embedding, "tolist") else query_embedding],
            n_results=n_results
        )
        
        parsed_results = []
        if results["ids"]:
            for i in range(len(results["ids"][0])):
                parsed_results.append({
                    "id": results["ids"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "document": results["documents"][0][i],
                    "distance": results["distances"][0][i]
                })
        
        return parsed_results

    def get_all_file_paths(self) -> List[str]:
        """
        Retrieves all unique file paths stored in ChromaDB.
        """
        results = self.collection.get(include=["metadatas"])
        if not results["metadatas"]:
            return []
        
        file_paths = {m.get("file_path") for m in results["metadatas"] if m.get("file_path")}
        return list(file_paths)

