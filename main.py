from src.parsers.factory import ParserFactory
import os

def main():
    factory = ParserFactory()
    
    project_root = os.path.abspath(os.path.dirname(__file__))
    src_path = os.path.join(project_root, "src")
    
    print(f"--- Recursive Directory Parsing: {src_path} ---")
    
    all_snippets = factory.parse_directory(src_path, recursive=True)
    
    print(f"Found total of {len(all_snippets)} snippets across all files.\n")
    
    for s in all_snippets[:10]:
        print(s)
    
    if len(all_snippets) > 10:
        print(f"\n... and {len(all_snippets) - 10} more.")

if __name__ == "__main__":
    main()
