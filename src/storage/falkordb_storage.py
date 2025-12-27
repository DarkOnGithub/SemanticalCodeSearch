import logging
import os
from typing import List
from redislite import FalkorDB
from src.IR.models import CodeSnippet, SnippetType, Relationship, GraphNode

logger = logging.getLogger(__name__)

class FalkorDBStorage:
    def __init__(self, db_path: str = "data/graph.db", graph_name: str = "codebase"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db = FalkorDB(db_path)
        self.graph = self.db.select_graph(graph_name)
        self._init_indices()

    def _init_indices(self):
        """Creates indices for Snippets."""
        try:
            self.graph.query("CREATE INDEX FOR (s:Snippet) ON (s.id)")
        except Exception as e:
            # Index might already exist
            logger.debug(f"Index creation note: {e}")

    def save_snippets(self, snippets: List[CodeSnippet]):
        if not snippets:
            return

        for s in snippets:
            query = """
            MERGE (s:Snippet {id: $id})
            SET s.name = $name, 
                s.type = $type, 
                s.file_path = $file_path
            """
            params = {
                "id": s.id,
                "name": s.name,
                "type": s.type.value,
                "file_path": s.file_path or ""
            }
            try:
                self.graph.query(query, params)
            except Exception as e:
                logger.error(f"Error saving snippet {s.id} to FalkorDB: {e}")

    def save_relationships(self, relationships: List[Relationship]):
        if not relationships:
            return

        # 1. Ensure all targets exist (placeholders)
        target_ids = list(set(r.target_id for r in relationships))
        for target_id in target_ids:
            placeholder_query = """
            MERGE (s:Snippet {id: $id})
            ON CREATE SET s.name = $id, s.type = 'placeholder', s.file_path = ''
            """
            try:
                self.graph.query(placeholder_query, {"id": target_id})
            except Exception as e:
                logger.error(f"Error ensuring FalkorDB placeholder {target_id}: {e}")

        for r in relationships:
            rel_type = r.type.value.upper()
            rel_query = f"""
            MATCH (src:Snippet {{id: $src}})
            MATCH (dst:Snippet {{id: $dst}})
            MERGE (src)-[:{rel_type}]->(dst)
            """
            params = {"src": r.source_id, "dst": r.target_id}
            try:
                self.graph.query(rel_query, params)
            except Exception as e:
                logger.error(f"Error saving FalkorDB relationship {rel_type}: {e}")

    def get_all_file_paths(self) -> List[str]:
        query = "MATCH (s:Snippet) WHERE s.file_path <> '' RETURN DISTINCT s.file_path"
        try:
            result = self.graph.query(query)
            # FalkorDB result structure might differ, usually result.result_set
            return [record[0] for record in result.result_set]
        except Exception as e:
            logger.error(f"Error fetching file paths from FalkorDB: {e}")
            return []

    def delete_file_data(self, file_path: str):
        query = "MATCH (s:Snippet {file_path: $path}) DETACH DELETE s"
        try:
            self.graph.query(query, {"path": file_path})
        except Exception as e:
            logger.error(f"Error deleting FalkorDB data for file {file_path}: {e}")

    def get_snippet_relationships(self, snippet_id: str) -> List[tuple]:
        """Returns all outgoing relationships for a snippet as (rel_type, target_name)"""
        query = """
        MATCH (s:Snippet {id: $id})-[r]->(t:Snippet)
        RETURN type(r) AS rel_type, t.name AS target_name
        """
        relationships = []
        try:
            result = self.graph.query(query, {"id": snippet_id})
            for record in result.result_set:
                relationships.append((record[0].lower(), record[1]))
        except Exception as e:
            logger.error(f"Error fetching FalkorDB relationships for {snippet_id}: {e}")
        return relationships

    def get_all_nodes(self) -> List[GraphNode]:
        query = """
        MATCH (s:Snippet) 
        RETURN s.id AS id, s.name AS name, s.type AS type, s.file_path AS file_path
        """
        nodes = []
        try:
            result = self.graph.query(query)
            for record in result.result_set:
                nodes.append(GraphNode(
                    id=record[0],
                    name=record[1],
                    type=SnippetType(record[2]),
                    file_path=record[3] if record[3] else None
                ))
        except Exception as e:
            logger.error(f"Error fetching all nodes from FalkorDB: {e}")
        return nodes

    def close(self):
        # FalkorDBLite might not need explicit close, but good to have
        pass

