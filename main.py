import argparse
import os
import logging
from src.indexer import ProjectIndexer
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
    
    snippets = indexer.extract_snippets()
    relationships = indexer.extract_relationships(snippets)
    
    indexer.summarize_snippets(snippets)
    indexer.save(snippets, relationships)
    indexer.cleanup(snippets)

    for s in snippets:
        if not (s.summary is None or s.summary == ""):
            print(s.summary)
    indexer.verify()

if __name__ == "__main__":
    main()
