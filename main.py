from src.parsers.factory import ParserFactory
import os
import time

def main():
    factory = ParserFactory()
    
    # Path to the src directory in the project
    project_root = os.path.abspath(os.path.dirname(__file__))
    src_path = os.path.join(project_root, "src")
    
    print(f"--- Benchmarking Caching Performance ---")
    
    # Run 1: Cold start (no cache)
    print("\nRun 1: Cold Start (Parsing from disk)...")
    start1 = time.perf_counter()
    all_snippets1 = factory.parse_directory(src_path, recursive=True)
    end1 = time.perf_counter()
    print(f"Time: {end1 - start1:.4f}s | Snippets: {len(all_snippets1)}")
    os.remove("./src/test.py")
    with open("./src/test.py", "w") as f:
        f.write("""
def test():
    print("Hello, World!")
        """)
    # Run 2: Hot start (from cache)
    print("\nRun 2: Hot Start (Identical content, should skip parsing)...")
    start2 = time.perf_counter()
    all_snippets2 = factory.parse_directory(src_path, recursive=True)
    end2 = time.perf_counter()
    
    print(f"Time: {end2 - start2:.4f}s | Snippets: {len(all_snippets2)}")
    
    improvement = (end1 - start1) / (end2 - start2)
    print(f"\nSpeedup: {improvement:.1f}x faster!")

if __name__ == "__main__":
    main()
