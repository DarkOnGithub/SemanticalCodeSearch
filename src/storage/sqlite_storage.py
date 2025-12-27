import os
import sqlite3
import json
import logging
from typing import List, Optional
from src.IR.models import CodeSnippet, SnippetType

logger = logging.getLogger(__name__)

class SQLiteStorage:
    def __init__(self, db_path: str = "data/codebase.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
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
            conn.commit()

    def save_snippets(self, snippets: List[CodeSnippet]):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for s in snippets:
                cursor.execute("""
                    INSERT OR REPLACE INTO snippets (
                        id, name, type, content, summary, parent_id, docstring, signature, 
                        file_path, start_line, end_line, start_byte, end_byte, is_skeleton, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    s.id,
                    s.name,
                    s.type.value,
                    s.content,
                    s.summary,
                    s.parent_id,
                    s.docstring,
                    s.signature,
                    s.file_path,
                    s.start_line,
                    s.end_line,
                    s.start_byte,
                    s.end_byte,
                    1 if s.is_skeleton else 0,
                    json.dumps(s.metadata)
                ))
            conn.commit()

    def get_file_hash(self, file_path: str) -> Optional[str]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT content_hash FROM file_hashes WHERE file_path = ?", (file_path,))
            row = cursor.fetchone()
            return row[0] if row else None

    def save_file_hash(self, file_path: str, content_hash: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT OR REPLACE INTO file_hashes (file_path, content_hash) VALUES (?, ?)", (file_path, content_hash))
            conn.commit()

    def get_file_snippets(self, file_path: str) -> List[CodeSnippet]:
        snippets = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM snippets WHERE file_path = ?", (file_path,))
            rows = cursor.fetchall()
            for row in rows:
                snippets.append(CodeSnippet(
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
                    metadata=json.loads(row["metadata_json"])
                ))
        return snippets

    def delete_file_snippets(self, file_path: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM snippets WHERE file_path = ?", (file_path,))
            cursor.execute("DELETE FROM file_hashes WHERE file_path = ?", (file_path,))
            conn.commit()

    def get_all_file_paths(self) -> List[str]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT file_path FROM snippets WHERE file_path IS NOT NULL")
            return [row[0] for row in cursor.fetchall()]

    def get_all_snippets(self) -> List[CodeSnippet]:
        snippets = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM snippets")
            rows = cursor.fetchall()
            for row in rows:
                snippets.append(CodeSnippet(
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
                    metadata=json.loads(row["metadata_json"])
                ))
        return snippets

    def get_snippet(self, snippet_id: str) -> Optional[CodeSnippet]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM snippets WHERE id = ?", (snippet_id,))
            row = cursor.fetchone()
            if row:
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
                    metadata=json.loads(row["metadata_json"])
                )
        return None

    def search_by_name(self, name_query: str) -> List[CodeSnippet]:
        snippets = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM snippets WHERE name LIKE ?", (f"%{name_query}%",))
            rows = cursor.fetchall()
            for row in rows:
                snippets.append(CodeSnippet(
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
                    metadata=json.loads(row["metadata_json"])
                ))
        return snippets

