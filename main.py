from src.IR.models import SnippetType
from src.parsers.factory import ParserFactory
import os
import time

def main():
    factory = ParserFactory()
    
    src_path = "/home/user/Downloads/WordSearchOCR-master/"
    
    print(f"--- Benchmarking ParserFactory.parse_directory ---")
    print(f"Target: {src_path}")
    
    start_time = time.perf_counter()
    all_snippets = factory.parse_directory(src_path, recursive=True)
    end_time = time.perf_counter()
    
    duration = end_time - start_time
    
    
    print(f"\nBenchmark Results:")
    print(f"Total Snippets Extracted: {len(all_snippets)}")
    print(f"Total Time Taken:         {duration:.4f} seconds")
    
    if all_snippets:
        avg_time = duration / len(all_snippets)
        print(f"Average Time per Snippet: {avg_time:.6f} seconds")
        files = set()
        for s in all_snippets:
            files.add(s.file_path)
            if s.type == SnippetType.STRUCT:
                print(s)
        # print("\n".join(files))
if __name__ == "__main__":
    main()
