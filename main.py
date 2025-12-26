from src.parsers.factory import ParserFactory
import os
import time

CHUNK_SIZE = 8000

def main():
    factory = ParserFactory(chunk_size=CHUNK_SIZE)
    
    # Path to the src directory in the project
    project_root = os.path.abspath(os.path.dirname(__file__))
    src_path = os.path.join(project_root, "src")
    
    print(f"--- Benchmarking Caching & Chunking Performance ---")
    
    # Run 1: Cold start (no cache)
    print("\nRun 1: Cold Start (Parsing from disk)...")
    start1 = time.perf_counter()
    all_snippets1 = factory.parse_directory(src_path, recursive=True)
    end1 = time.perf_counter()
    print(f"Time: {end1 - start1:.4f}s | Snippets: {len(all_snippets1)}")
    
    chunked = [s for s in all_snippets1 if s.metadata.get("chunk_index") is not None]
    print(f"Number of chunked snippets: {len(chunked)}")
    if chunked:
        print(f"Example chunk: {chunked[0].name} (Length: {len(chunked[0].content)})")
        entity_name = chunked[0].name.split("_chunk_")[0]
        entity_chunks = [s for s in chunked if s.name.startswith(entity_name)]
        print(f"Total chunks for '{entity_name}': {len(entity_chunks)}")
        for i, chunk in enumerate(entity_chunks):
            print(f"Chunk {i+1}: {chunk.name} (Length: {len(chunk.content)})")
            print(f"Content: {chunk.content}")
            print("-" * 50)

if __name__ == "__main__":
    main()
