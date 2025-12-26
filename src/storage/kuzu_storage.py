import kuzu
import os
import json
from typing import List
from src.IR.models import CodeSnippet, SnippetType, Relationship, RelationType

class KuzuStorage:
    def __init__(self, db_path: str = "data/kuzu"):
        # Kuzu handles directory creation. It fails if the directory already exists.
        parent_dir = os.path.dirname(db_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        
        # If it exists, we just open it. Kuzu Database(path) opens if exists,
        # but apparently in 0.11.3 it fails if it's an EMPTY directory we just created.
        # So we only create the parent.
        self.db = kuzu.Database(db_path)
        self.conn = kuzu.Connection(self.db)
        self._init_schema()

    def _init_schema(self):
        # Create Snippet node table
        try:
            self.conn.execute("""
                CREATE NODE TABLE Snippet (
                    id STRING,
                    name STRING,
                    type STRING,
                    parent_id STRING,
                    signature STRING,
                    file_path STRING,
                    start_line INT64,
                    end_line INT64,
                    metadata_json STRING,
                    PRIMARY KEY (id)
                )
            """)
        except Exception as e:
            if "already exists" not in str(e):
                print(f"Error creating Snippet table: {e}")

        # Create relationship tables for each RelationType
        for rel in RelationType:
            rel_name = rel.value.upper()
            try:
                self.conn.execute(f"CREATE REL TABLE {rel_name} (FROM Snippet TO Snippet)")
            except Exception as e:
                if "already exists" not in str(e):
                    print(f"Error creating relationship table {rel_name}: {e}")

    def save_snippets(self, snippets: List[CodeSnippet]):
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
        
        if not batch:
            return

        try:
            self.conn.execute("""
                UNWIND $batch AS map
                MERGE (s:Snippet {id: map.id})
                ON CREATE SET 
                    s.name = map.name, s.type = map.type,
                    s.parent_id = map.parent_id, s.signature = map.signature,
                    s.file_path = map.file_path, s.start_line = map.start_line, s.end_line = map.end_line,
                    s.metadata_json = map.metadata_json
                ON MATCH SET
                    s.name = map.name, s.type = map.type,
                    s.parent_id = map.parent_id, s.signature = map.signature,
                    s.file_path = map.file_path, s.start_line = map.start_line, s.end_line = map.end_line,
                    s.metadata_json = map.metadata_json
            """, {"batch": batch})
        except Exception as e:
            print(f"Error batch saving snippets: {e}")

    def save_relationships(self, relationships: List[Relationship]):
        if not relationships:
            return

        # 1. Ensure all targets exist (placeholders)
        target_ids = list(set(r.target_id for r in relationships))
        # Batch check/create placeholders
        try:
            self.conn.execute("""
                UNWIND $ids AS id
                MERGE (s:Snippet {id: id})
                ON CREATE SET s.name = id, s.type = 'placeholder',
                             s.parent_id = '', s.signature = '',
                             s.file_path = '', s.start_line = -1, s.end_line = -1,
                             s.metadata_json = '{}'
            """, {"ids": target_ids})
        except Exception as e:
            print(f"Error ensuring relationship placeholders: {e}")

        # 2. Batch create relationships by type
        rel_groups = {}
        for r in relationships:
            rel_type = r.type.value.upper()
            if rel_type not in rel_groups:
                rel_groups[rel_type] = []
            rel_groups[rel_type].append({"src": r.source_id, "dst": r.target_id})

        for rel_name, batch in rel_groups.items():
            try:
                # Kuzu doesn't support dynamic relationship types in Cypher yet ([:$rel_name] is usually invalid)
                # So we have to loop through types but can batch within each type.
                self.conn.execute(f"""
                    UNWIND $batch AS edge
                    MATCH (src:Snippet {{id: edge.src}}), (dst:Snippet {{id: edge.dst}})
                    MERGE (src)-[:{rel_name}]->(dst)
                """, {"batch": batch})
            except Exception as e:
                print(f"Error batch saving relationships {rel_name}: {e}")

    def get_all_file_paths(self) -> List[str]:
        res = self.conn.execute("MATCH (s:Snippet) RETURN DISTINCT s.file_path")
        paths = []
        while res.has_next():
            path = res.get_next()[0]
            if path:
                paths.append(path)
        return paths

    def delete_file_data(self, file_path: str):
        try:
            self.conn.execute("MATCH (s:Snippet {file_path: $path}) DETACH DELETE s", {"path": file_path})
        except Exception as e:
            print(f"Error deleting data for file {file_path}: {e}")

    def get_snippet_relationships(self, snippet_id: str) -> List[tuple]:
        """Returns all outgoing relationships for a snippet as (rel_type, target_name)"""
        relationships = []
        for rel in RelationType:
            rel_name = rel.value.upper()
            try:
                res = self.conn.execute(f"""
                    MATCH (s:Snippet {{id: $id}})-[r:{rel_name}]->(t:Snippet)
                    RETURN t.name
                """, {"id": snippet_id})
                while res.has_next():
                    target_name = res.get_next()[0]
                    relationships.append((rel.value, target_name))
            except Exception:
                continue
        return relationships

    def get_all_snippets(self) -> List[CodeSnippet]:
        snippets = []
        res = self.conn.execute("""
            MATCH (s:Snippet) 
            RETURN s.id, s.name, s.type, s.parent_id, 
                   s.signature, s.file_path, s.start_line, s.end_line, 
                   s.metadata_json
        """)
        cols = [c.split('.')[-1] for c in res.get_column_names()]
        while res.has_next():
            row = res.get_next()
            d = dict(zip(cols, row))
            
            snippets.append(CodeSnippet(
                id=d["id"],
                name=d["name"],
                type=SnippetType(d["type"]),
                content="", # Lean: content is in SQLite
                parent_id=d["parent_id"] if d["parent_id"] else None,
                docstring=None, # Lean: docstring is in SQLite
                signature=d["signature"] if d["signature"] else None,
                file_path=d["file_path"] if d["file_path"] else None,
                start_line=d["start_line"] if d["start_line"] != -1 else None,
                end_line=d["end_line"] if d["end_line"] != -1 else None,
                start_byte=None,
                end_byte=None,
                metadata=json.loads(d["metadata_json"])
            ))
        return snippets

