import json
import logging
import subprocess
import time
from typing import List
from neo4j import GraphDatabase, Driver
from src.IR.models import CodeSnippet, SnippetType, Relationship

logger = logging.getLogger(__name__)

class Neo4jStorage:
    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password", auto_start: bool = True):
        self.uri = uri
        self.user = user
        self.password = password
        
        if auto_start:
            self._ensure_neo4j_running()
            
        self.driver: Driver = GraphDatabase.driver(uri, auth=(user, password))
        self._init_constraints()

    def _ensure_neo4j_running(self):
        """Checks if Neo4j is running via Docker, starts it if not, and waits for health."""
        try:
            # Check if container is running
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", "semantical-code-search-neo4j"],
                capture_output=True, text=True
            )
            
            if result.returncode != 0:
                if "permission denied" in result.stderr.lower():
                    logger.error("Docker permission denied. Run: sudo usermod -aG docker $USER && newgrp docker")
                    return
            
            if result.returncode != 0 or "true" not in result.stdout:
                logger.info("Neo4j not running. Starting via docker-compose...")
                subprocess.run(["docker-compose", "up", "-d"], check=True, capture_output=True, text=True)
            
            # Wait for Neo4j health
            for _ in range(30):
                health = subprocess.run(
                    ["docker", "inspect", "-f", "{{.State.Health.Status}}", "semantical-code-search-neo4j"],
                    capture_output=True, text=True
                )
                if "healthy" in health.stdout:
                    logger.info("Neo4j is healthy.")
                    return
                time.sleep(2)
            
            logger.warning("Neo4j health check timed out, connecting anyway...")
            
        except FileNotFoundError:
            logger.error("Docker or docker-compose not found. Please ensure they are installed to use auto-start.")
        except Exception as e:
            logger.error(f"Failed to automatically start Neo4j: {e}")

    def close(self):
        self.driver.close()

    def _init_constraints(self):
        """Creates uniqueness constraints for Snippets."""
        with self.driver.session() as session:
            try:
                # Snippet ID should be unique
                session.run("CREATE CONSTRAINT snippet_id_unique IF NOT EXISTS FOR (s:Snippet) REQUIRE s.id IS UNIQUE")
            except Exception as e:
                logger.error(f"Error creating Neo4j constraints: {e}")

    def save_snippets(self, snippets: List[CodeSnippet]):
        if not snippets:
            return

        batch = []
        for s in snippets:
            batch.append({
                "id": s.id,
                "name": s.name,
                "type": s.type.value,
                "parent_id": s.parent_id or "",
                "signature": s.signature or "",
                "file_path": s.file_path or "",
                "start_line": s.start_line if s.start_line is not None else -1,
                "end_line": s.end_line if s.end_line is not None else -1,
                "metadata_json": json.dumps(s.metadata)
            })

        query = """
        UNWIND $batch AS map
        MERGE (s:Snippet {id: map.id})
        SET s += map
        """
        
        with self.driver.session() as session:
            try:
                session.run(query, batch=batch)
            except Exception as e:
                logger.error(f"Error batch saving snippets to Neo4j: {e}")

    def save_relationships(self, relationships: List[Relationship]):
        if not relationships:
            return

        # 1. Ensure all targets exist (placeholders)
        target_ids = list(set(r.target_id for r in relationships))
        placeholder_query = """
        UNWIND $ids AS id
        MERGE (s:Snippet {id: id})
        ON CREATE SET s.name = id, s.type = 'placeholder',
                     s.parent_id = '', s.signature = '',
                     s.file_path = '', s.start_line = -1, s.end_line = -1,
                     s.metadata_json = '{}'
        """
        
        with self.driver.session() as session:
            try:
                session.run(placeholder_query, ids=target_ids)
            except Exception as e:
                logger.error(f"Error ensuring Neo4j placeholders: {e}")

        rel_groups = {}
        for r in relationships:
            rel_type = r.type.value.upper()
            if rel_type not in rel_groups:
                rel_groups[rel_type] = []
            rel_groups[rel_type].append({"src": r.source_id, "dst": r.target_id})

        for rel_name, batch in rel_groups.items():
            rel_query = f"""
            UNWIND $batch AS edge
            MATCH (src:Snippet {{id: edge.src}}), (dst:Snippet {{id: edge.dst}})
            MERGE (src)-[:{rel_name}]->(dst)
            """
            try:
                with self.driver.session() as session:
                    session.run(rel_query, batch=batch)
            except Exception as e:
                logger.error(f"Error batch saving Neo4j relationships {rel_name}: {e}")

    def get_all_file_paths(self) -> List[str]:
        query = "MATCH (s:Snippet) WHERE s.file_path <> '' RETURN DISTINCT s.file_path"
        with self.driver.session() as session:
            result = session.run(query)
            return [record["s.file_path"] for record in result]

    def delete_file_data(self, file_path: str):
        query = "MATCH (s:Snippet {file_path: $path}) DETACH DELETE s"
        with self.driver.session() as session:
            try:
                session.run(query, path=file_path)
            except Exception as e:
                logger.error(f"Error deleting Neo4j data for file {file_path}: {e}")

    def get_snippet_relationships(self, snippet_id: str) -> List[tuple]:
        """Returns all outgoing relationships for a snippet as (rel_type, target_name)"""
        query = """
        MATCH (s:Snippet {id: $id})-[r]->(t:Snippet)
        RETURN type(r) AS rel_type, t.name AS target_name
        """
        relationships = []
        with self.driver.session() as session:
            try:
                result = session.run(query, id=snippet_id)
                for record in result:
                    relationships.append((record["rel_type"].lower(), record["target_name"]))
            except Exception as e:
                logger.error(f"Error fetching Neo4j relationships for {snippet_id}: {e}")
        return relationships

    def get_all_snippets(self) -> List[CodeSnippet]:
        query = """
        MATCH (s:Snippet) 
        RETURN s.id AS id, s.name AS name, s.type AS type, s.parent_id AS parent_id, 
               s.signature AS signature, s.file_path AS file_path, 
               s.start_line AS start_line, s.end_line AS end_line, 
               s.metadata_json AS metadata_json
        """
        snippets = []
        with self.driver.session() as session:
            result = session.run(query)
            for record in result:
                snippets.append(CodeSnippet(
                    id=record["id"],
                    name=record["name"],
                    type=SnippetType(record["type"]),
                    content="", # Lean: content is in SQLite
                    parent_id=record["parent_id"] if record["parent_id"] else None,
                    docstring=None, # Lean: docstring is in SQLite
                    signature=record["signature"] if record["signature"] else None,
                    file_path=record["file_path"] if record["file_path"] else None,
                    start_line=int(record["start_line"]) if record["start_line"] != -1 else None,
                    end_line=int(record["end_line"]) if record["end_line"] != -1 else None,
                    start_byte=None,
                    end_byte=None,
                    metadata=json.loads(record["metadata_json"])
                ))
        return snippets

