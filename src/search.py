import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from src.indexer import ProjectIndexer
from src.model.orchestrator import Orchestrator
from src.model.reranker import get_reranker

logger = logging.getLogger(__name__)

ANSWER_PROMPT = """You are an expert software engineer assistant. Answer the user's question about the codebase based on the provided code snippets and their context.

User Question: {query}

Relevant Code Snippets:
{snippets_context}

Instructions:
1. Provide a direct and technical answer to the user's question.
2. Use markdown formatting for better readability (e.g., **bold** for emphasis, `code` for identifiers, lists for steps).
3. Reference specific file paths and line numbers where appropriate.
4. If the code snippets don't contain enough information to answer the question, state that clearly.
5. Use the summary information if available to provide high-level context.

Answer:"""

class SearchManager:
    def __init__(self, indexer: ProjectIndexer, orchestrator: Optional[Orchestrator] = None, use_reranker: bool = True):
        self.indexer = indexer
        self.llm = indexer.llm
        self.embedding_model = indexer.embedding_model
        self.sqlite = indexer.sqlite
        self.graph_db = indexer.graph_db
        self.chroma = indexer.chroma
        self.orchestrator = orchestrator
        self.reranker = get_reranker() if use_reranker else None

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Performs a hybrid search (Semantic + Keyword) with RRF fusion and Re-ranking.
        """
        start_time = time.time()
        if not self.chroma or not self.embedding_model:
            logger.error("Search components not initialized.")
            return {"results": [], "final_query": query, "hyde_used": False}

        # 1. Orchestration (HyDE)
        final_query, hyde_used = self._orchestrate_query(query)

        # 2. Retrieval (Vector + Keyword)
        initial_n = max(n_results * 10, 100)
        vector_results, keyword_results = self._retrieve_candidates(query, final_query, hyde_used, initial_n)

        # 3. Fusion (Reciprocal Rank Fusion)
        sorted_ids = self._fuse_results(vector_results, keyword_results)
        
        # 4. Hydration & Re-ranking
        final_results = self._hydrate_and_rerank(sorted_ids[:initial_n], vector_results, keyword_results, query, n_results)

        logger.info(f"Hybrid Search complete in {time.time() - start_time:.2f}s")
        return {
            "results": final_results,
            "final_query": final_query,
            "hyde_used": hyde_used
        }

    def answer_query(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Generates a complete answer using the LLM."""
        if not self.llm: return "LLM not available."
        if not results: return "No search results found."

        context = self._build_context_string(results)
        prompt = ANSWER_PROMPT.format(query=query, snippets_context=context)
        
        logger.info("Generating answer with Gemini...")
        return self.llm.complete(prompt)

    def stream_answer_query(self, query: str, results: List[Dict[str, Any]]):
        """Yields tokens for the answer as they are generated."""
        if not self.llm:
            yield "LLM not available."
            return
        if not results:
            yield "No search results found."
            return

        context = self._build_context_string(results)
        prompt = ANSWER_PROMPT.format(query=query, snippets_context=context)
        
        logger.info("Streaming answer with Gemini...")
        yield from self.llm.stream_complete(prompt)

    # --- Pipeline Steps ---

    def _orchestrate_query(self, query: str) -> Tuple[str, bool]:
        if not self.orchestrator:
            return query, False
            
        logger.debug(f"Orchestrating query: {query}")
        final_query = self.orchestrator.process_query(query)
        return final_query, (final_query != query)

    def _retrieve_candidates(self, original_query: str, final_query: str, hyde_used: bool, limit: int) -> Tuple[List[Dict], List[Dict]]:
        # Vector Search
        query_embedding = self.embedding_model.embed_text(final_query)
        vector_results = self.chroma.query(query_embedding, n_results=limit)
        
        # Keyword Search
        keyword_results = []
        if self.sqlite:
            search_queries = [original_query]
            if hyde_used:
                search_queries.append(final_query)
            
            seen_ids = set()
            for q in search_queries:
                snippets = self.sqlite.search_by_content(q, limit=limit)
                for s in snippets:
                    if s.id not in seen_ids:
                        keyword_results.append({
                            "id": s.id, 
                            "document": s.to_embeddable_text(use_summary=True)
                        })
                        seen_ids.add(s.id)
                if len(keyword_results) >= limit:
                    break
                    
        return vector_results, keyword_results

    def _fuse_results(self, vector_results: List[Dict], keyword_results: List[Dict], k: int = 60) -> List[str]:
        """Implements Reciprocal Rank Fusion (RRF)."""
        combined_scores = {}
        
        for rank, res in enumerate(vector_results, 1):
            combined_scores[res["id"]] = combined_scores.get(res["id"], 0) + (1.0 / (k + rank))
            
        for rank, res in enumerate(keyword_results, 1):
            combined_scores[res["id"]] = combined_scores.get(res["id"], 0) + (1.0 / (k + rank))

        return sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)

    def _hydrate_and_rerank(self, top_ids: List[str], vector_res: List[Dict], keyword_res: List[Dict], query: str, final_k: int) -> List[Dict]:
        candidates = []
        
        # Hydrate snippet objects
        for sid in top_ids:
            if not self.sqlite: continue
            snippet = self.sqlite.get_snippet(sid)
            if not snippet: continue

            # Find the best available document text for reranking
            doc_text = next((r["document"] for r in vector_res if r["id"] == sid), None)
            if not doc_text:
                doc_text = next((r["document"] for r in keyword_res if r["id"] == sid), None)
            if not doc_text:
                doc_text = snippet.to_embeddable_text(use_summary=True)
            
            candidates.append({
                "id": sid,
                "snippet": snippet,
                "document": doc_text
            })

        # Re-rank
        if self.reranker and candidates:
            logger.info(f"Reranking {len(candidates)} candidates...")
            docs = [c["document"] for c in candidates]
            reranked_meta = self.reranker.rerank(query, docs, top_n=final_k)
            
            # Reorder candidates based on reranker output
            final_candidates = []
            for meta in reranked_meta:
                cand = candidates[meta["index"]]
                cand["score"] = meta["score"]
                final_candidates.append(cand)
            candidates = final_candidates
        else:
            candidates = candidates[:final_k]

        # Add Parents & Relations
        results = []
        for c in candidates:
            snippet = c["snippet"]
            parent = self.sqlite.get_snippet(snippet.parent_id) if snippet.parent_id else None
            relations = self.graph_db.get_snippet_relationships(snippet.id) if self.graph_db else []
            
            results.append({
                "snippet": snippet,
                "parent": parent,
                "relations": relations,
                "score": c.get("score"),
                "document": c["document"]
            })
            
        return results

    def _build_context_string(self, results: List[Dict[str, Any]]) -> str:
        context_parts = []
        for i, res in enumerate(results, 1):
            s = res["snippet"]
            parent = res.get("parent")
            relations = res.get("relations", [])
            
            parent_info = f" [in {parent.name} ({parent.type.value})]" if parent else ""
            header = f"--- Snippet {i} [{s.name}{parent_info} at {s.file_path}:{s.start_line + 1}] ---"
            
            content = [header]
            if s.summary:
                content.append(f"Summary: {s.summary}")
            
            if relations:
                rel_str = ", ".join([f"{r_type} {target}" for r_type, target in relations])
                content.append(f"Relationships: {rel_str}")
                
            content.append(f"Code:\n{s.content}\n")
            context_parts.append("\n".join(content))
            
        return "\n".join(context_parts)