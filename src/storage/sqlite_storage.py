import os
import sqlite3
import json
import logging
import re
from typing import List, Optional, Dict

from src.IR.models import CodeSnippet, SnippetType

logger = logging.getLogger(__name__)

class SQLiteStorage:
    def __init__(self, db_path: str = "data/codebase.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Helper to get a configured connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize the database schema, FTS, and triggers."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            self._create_tables(cursor)
            self._setup_fts(cursor)
            self._create_triggers(cursor)
            conn.commit()

    def _create_tables(self, cursor: sqlite3.Cursor):
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS snippets (
                id TEXT PRIMARY KEY,
                name TEXT,
                type TEXT,
                content TEXT,
                summary TEXT,
                parent_id TEXT,
                docstring TEXT,
                signature TEXT,
                file_path TEXT,
                start_line INTEGER,
                end_line INTEGER,
                start_byte INTEGER,
                end_byte INTEGER,
                is_skeleton INTEGER DEFAULT 0,
                metadata_json TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS file_hashes (
                file_path TEXT PRIMARY KEY,
                content_hash TEXT
            )
        """)

    def _setup_fts(self, cursor: sqlite3.Cursor):
        try:
            cursor.execute("PRAGMA integrity_check")
            if cursor.fetchone()[0] != "ok":
                logger.warning("Database corruption detected during integrity check!")

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='snippets_fts'")
            fts_exists = cursor.fetchone()

            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS snippets_fts 
                USING fts5(id UNINDEXED, name, content, summary, content='snippets', content_rowid='rowid')
            """)

            if not fts_exists:
                cursor.execute("INSERT INTO snippets_fts(snippets_fts) VALUES('rebuild')")
            else:
                cursor.execute("SELECT COUNT(*) FROM snippets")
                s_count = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM snippets_fts")
                f_count = cursor.fetchone()[0]
                
                if s_count != f_count:
                    logger.warning(f"FTS index out of sync (snippets: {s_count}, FTS: {f_count}). Rebuilding...")
                    cursor.execute("INSERT INTO snippets_fts(snippets_fts) VALUES('rebuild')")

        except sqlite3.OperationalError as e:
            if "malformed" in str(e).lower():
                logger.error(f"FTS index is malformed: {e}. Attempting recovery...")
                try:
                    cursor.execute("INSERT INTO snippets_fts(snippets_fts) VALUES('rebuild')")
                except Exception as recovery_e:
                    logger.error(f"Recovery failed: {recovery_e}")
            else:
                logger.warning(f"FTS5 initialization failed: {e}. Falling back to LIKE.")

    def _create_triggers(self, cursor: sqlite3.Cursor):
        """Ensure FTS index stays updated automatically."""
        triggers = [
            ("snippets_ai", """
                CREATE TRIGGER snippets_ai AFTER INSERT ON snippets BEGIN
                  INSERT INTO snippets_fts(rowid, id, name, content, summary) 
                  VALUES (new.rowid, new.id, new.name, new.content, new.summary);
                END;
            """),
            ("snippets_ad", """
                CREATE TRIGGER snippets_ad AFTER DELETE ON snippets BEGIN
                  INSERT INTO snippets_fts(snippets_fts, rowid, id, name, content, summary) 
                  VALUES('delete', old.rowid, old.id, old.name, old.content, old.summary);
                END;
            """),
            ("snippets_au", """
                CREATE TRIGGER snippets_au AFTER UPDATE ON snippets BEGIN
                  INSERT INTO snippets_fts(snippets_fts, rowid, id, name, content, summary) 
                  VALUES('delete', old.rowid, old.id, old.name, old.content, old.summary);
                  INSERT INTO snippets_fts(rowid, id, name, content, summary) 
                  VALUES (new.rowid, new.id, new.name, new.content, new.summary);
                END;
            """)
        ]
        
        for name, sql in triggers:
            cursor.execute(f"DROP TRIGGER IF EXISTS {name}")
            cursor.execute(sql)

    def _row_to_snippet(self, row: sqlite3.Row) -> CodeSnippet:
        """Centralized converter from DB row to CodeSnippet object."""
        return CodeSnippet(
            id=row["id"],
            name=row["name"],
            type=SnippetType(row["type"]),
            content=row["content"],
            summary=row["summary"],
            parent_id=row["parent_id"],
            docstring=row["docstring"],
            signature=row["signature"],
            file_path=row["file_path"],
            start_line=row["start_line"],
            end_line=row["end_line"],
            start_byte=row["start_byte"],
            end_byte=row["end_byte"],
            is_skeleton=bool(row["is_skeleton"]),
            metadata=json.loads(row["metadata_json"]) if row["metadata_json"] else {}
        )

    def save_snippets(self, snippets: List[CodeSnippet], _retry_count: int = 0):
        if not snippets:
            return

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                sql = """
                    INSERT OR REPLACE INTO snippets (
                        id, name, type, content, summary, parent_id, docstring, signature, 
                        file_path, start_line, end_line, start_byte, end_byte, is_skeleton, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                
                batch_data = []
                for s in snippets:
                    summary = s.summary
                    if summary is not None and not isinstance(summary, str):
                        summary = json.dumps(summary) if isinstance(summary, (dict, list)) else str(summary)

                    parent_id = str(s.parent_id) if s.parent_id is not None and not isinstance(s.parent_id, str) else s.parent_id

                    batch_data.append((
                        s.id, s.name, s.type.value, s.content, summary, parent_id,
                        s.docstring, s.signature, s.file_path, s.start_line, s.end_line,
                        s.start_byte, s.end_byte, 1 if s.is_skeleton else 0, json.dumps(s.metadata)
                    ))
                
                cursor.executemany(sql, batch_data)
                conn.commit()

        except sqlite3.OperationalError as e:
            if "malformed" in str(e).lower() and _retry_count < 1:
                logger.error(f"Corruption detected during save: {e}. Attempting FTS rebuild and retry...")
                self._rebuild_fts_index()
                self.save_snippets(snippets, _retry_count + 1)
            else:
                logger.error(f"Failed to save snippets after retries: {e}")
                raise

    def get_file_hash(self, file_path: str) -> Optional[str]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT content_hash FROM file_hashes WHERE file_path = ?", (file_path,))
            row = cursor.fetchone()
            return row["content_hash"] if row else None

    def save_file_hash(self, file_path: str, content_hash: str):
        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO file_hashes (file_path, content_hash) VALUES (?, ?)", 
                (file_path, content_hash)
            )
            conn.commit()

    def get_file_snippets(self, file_path: str) -> List[CodeSnippet]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM snippets WHERE file_path = ?", (file_path,))
            return [self._row_to_snippet(row) for row in cursor.fetchall()]

    def delete_file_snippets(self, file_path: str, _retry_count: int = 0):
        try:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM snippets WHERE file_path = ?", (file_path,))
                conn.execute("DELETE FROM file_hashes WHERE file_path = ?", (file_path,))
                conn.commit()
        except sqlite3.OperationalError as e:
            if "malformed" in str(e).lower() and _retry_count < 1:
                logger.error(f"Corruption detected during delete: {e}. Attempting FTS rebuild...")
                self._rebuild_fts_index()
                self.delete_file_snippets(file_path, _retry_count + 1)
            else:
                raise

    def get_all_file_paths(self) -> List[str]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT file_path FROM snippets WHERE file_path IS NOT NULL")
            return [row[0] for row in cursor.fetchall()]

    def get_all_snippets(self) -> List[CodeSnippet]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM snippets")
            return [self._row_to_snippet(row) for row in cursor.fetchall()]

    def get_snippet(self, snippet_id: str) -> Optional[CodeSnippet]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM snippets WHERE id = ?", (snippet_id,))
            row = cursor.fetchone()
            return self._row_to_snippet(row) if row else None

    def get_snippets(self, snippet_ids: List[str]) -> Dict[str, CodeSnippet]:
        """Bulk fetch snippets by ID."""
        if not snippet_ids:
            return {}
        
        placeholders = ",".join(["?"] * len(snippet_ids))
        query = f"SELECT * FROM snippets WHERE id IN ({placeholders})"
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, snippet_ids)
            return {row["id"]: self._row_to_snippet(row) for row in cursor.fetchall()}

    def search_by_content(self, query: str, limit: int = 50) -> List[CodeSnippet]:
        """
        Multi-stage search strategy:
        1. Exact identifier match
        2. FTS Exact Phrase
        3. FTS AND (all terms)
        4. FTS OR (any term)
        5. Fallback: Standard LIKE
        """
        original_terms = re.findall(r'[a-zA-Z0-9_\.]+', query.strip())
        if not original_terms:
            return []

        tech_terms = [t for t in original_terms if len(t) > 1]
        
        results_map: Dict[str, sqlite3.Row] = {}

        with self._get_connection() as conn:
            cursor = conn.cursor()

            for term in tech_terms:
                if len(term) < 3:
                    continue
                # We use a fake rank of -100.0 to push these to the top if we were sorting later
                cursor.execute("SELECT *, -100.0 as rank FROM snippets WHERE name = ? LIMIT ?", (term, limit))
                for r in cursor.fetchall():
                    results_map.setdefault(r["id"], r)

            fts_strategies = []
            if len(tech_terms) > 1:
                fts_strategies.append(f'"{ " ".join(tech_terms) }"') # Exact phrase
            
            if tech_terms:
                fts_strategies.append(" AND ".join(tech_terms))
                fts_strategies.append(" OR ".join(tech_terms))
            else:
                fts_strategies.append(" OR ".join(original_terms))

            for fts_query in fts_strategies:
                if len(results_map) >= limit:
                    break
                    
                try:
                    sql = """
                        SELECT s.*, f.rank FROM snippets s
                        JOIN snippets_fts f ON s.rowid = f.rowid
                        WHERE snippets_fts MATCH ?
                        ORDER BY rank
                        LIMIT ?
                    """
                    cursor.execute(sql, (fts_query, limit))
                    for r in cursor.fetchall():
                        results_map.setdefault(r["id"], r)
                except sqlite3.OperationalError:
                    continue 

            if not results_map:
                sql_like = "SELECT * FROM snippets WHERE name LIKE ? OR content LIKE ? OR summary LIKE ? LIMIT ?"
                pattern = f"%{query}%"
                cursor.execute(sql_like, (pattern, pattern, pattern, limit))
                for r in cursor.fetchall():
                    results_map.setdefault(r["id"], r)

        return [self._row_to_snippet(row) for row in results_map.values()]

    def search_by_name(self, name_query: str) -> List[CodeSnippet]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM snippets WHERE name LIKE ?", (f"%{name_query}%",))
            return [self._row_to_snippet(row) for row in cursor.fetchall()]

    def _rebuild_fts_index(self):
        """Helper to force a rebuild of the FTS index."""
        try:
            with self._get_connection() as conn:
                conn.execute("INSERT INTO snippets_fts(snippets_fts) VALUES('rebuild')")
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to rebuild FTS index: {e}")