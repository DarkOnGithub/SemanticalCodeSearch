import argparse
import os
import logging
from src.indexer import ProjectIndexer
from src.search import SearchManager
from dotenv import load_dotenv

def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Semantical Code Search Indexer")
    parser.add_argument("dir", nargs="?", default=os.getcwd(), help="Directory to index")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for parsing")
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel workers for summarization")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose (DEBUG) logging")
    parser.add_argument("--query", type=str, help="Search the codebase using natural language")
    args = parser.parse_args()

    setup_logging(args.verbose)
    
    # 1. Create an indexer instance for this specific folder
    indexer = ProjectIndexer(
        src_path=args.dir, 
        chunk_size=args.chunk_size,
        max_workers=args.workers
    )
    
    # 2. Manually trigger the pipeline steps
    indexer.initialize_storage()
    
    if args.query:
        from src.model.orchestrator import get_orchestrator
        orchestrator = get_orchestrator()
        searcher = SearchManager(indexer, orchestrator=orchestrator)
        
        # Check if indexed
        if searcher.chroma.collection.count() == 0:
            print("\n[!] The codebase has not been indexed yet. Please run indexing first by running 'python main.py' without the --query argument.")
            return

        print(f"\n--- Searching for: '{args.query}' ---")
        results = searcher.search(args.query)
        
        if not results:
            print("No relevant code found.")
        else:
            for i, res in enumerate(results, 1):
                s = res["snippet"]
                parent = res.get("parent")
                score = res["score"]
                relations = res["relations"]
                
                parent_info = f" [in {parent.name} ({parent.type.value})]" if parent else ""
                print(f"\n[{i}] {s.name}{parent_info} ({s.file_path}:{s.start_line + 1}) - Score: {score:.4f}")
                if s.summary:
                    print(f"    Summary: {s.summary}")
                
                if relations:
                    print("    Relations:")
                    for rel_type, target in relations:
                        print(f"      - {rel_type.upper()} -> {target}")
                
                # print(f"\n    Code:\n{s.content[:200]}...") # Optional: show snippet start
            
            # 3. Generate natural language answer using DeepSeek
            print("\n" + "="*50)
            print("--- DeepSeek Analysis ---")
            answer = searcher.answer_query(args.query, results)
            print(answer)
            print("="*50 + "\n")
            
        return

    snippets = indexer.extract_snippets()
    relationships = indexer.extract_relationships(snippets)
    
    indexer.summarize_snippets(snippets)
    embeddings = indexer.embed_snippets(snippets)
    indexer.save(snippets, relationships, embeddings=embeddings)
    indexer.cleanup(snippets)

    indexer.verify()

if __name__ == "__main__":
    main()
