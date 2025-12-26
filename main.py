from src.parsers.factory import ParserFactory
from src.storage.sqlite_storage import SQLiteStorage
import os
import time

CHUNK_SIZE = 1000

def main():
    factory = ParserFactory(chunk_size=CHUNK_SIZE)
    storage = SQLiteStorage("data/codebase.db")
    
    # Path to the src directory in the project
    project_root = os.path.abspath(os.path.dirname(__file__))
    src_path = os.path.join(project_root, "src")
    
    print(f"--- Parsing and Saving to SQL ---")
    
    # Parse directory
    all_snippets = factory.parse_directory(src_path, recursive=True)
    print(f"Parsed {len(all_snippets)} snippets.")

    # Save to SQLite
    print("Saving to database...")
    storage.save_snippets(all_snippets)
    print("Save complete.")

    # Verify retrieval
    print("\n--- Verifying Database Storage ---")
    db_snippets = storage.get_all_snippets()
    print(f"Retrieved {len(db_snippets)} snippets from the database.")

    for snippet in db_snippets:
        print(f"Snippet: {snippet.name} (Length: {len(snippet.content)})")
        print(f"Content: {snippet.content}")
        print("-" * 50)

if __name__ == "__main__":
    main()
