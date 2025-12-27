import os
import sqlite3
import json
import logging
from typing import List
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
                    metadata_json TEXT
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
                        file_path, start_line, end_line, start_byte, end_byte, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    json.dumps(s.metadata)
                ))
            conn.commit()

    def get_all_file_paths(self) -> List[str]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT file_path FROM snippets WHERE file_path IS NOT NULL")
            return [row[0] for row in cursor.fetchall()]

    def delete_file_snippets(self, file_path: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM snippets WHERE file_path = ?", (file_path,))
            conn.commit()

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
                    metadata=json.loads(row["metadata_json"])
                ))
        return snippets

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
                    metadata=json.loads(row["metadata_json"])
                ))
        return snippets

