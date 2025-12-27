import argparse
import os
import logging
from src.indexer import ProjectIndexer

def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def main():
    parser = argparse.ArgumentParser(description="Semantical Code Search Indexer")
    parser.add_argument("dir", nargs="?", default=os.getcwd(), help="Directory to index")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for parsing")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose (DEBUG) logging")
    args = parser.parse_args()

    setup_logging(args.verbose)
    
    # 1. Create an indexer instance for this specific folder
    indexer = ProjectIndexer(
        src_path=args.dir, 
        chunk_size=args.chunk_size
    )
    
    # 2. Manually trigger the pipeline steps
    indexer.initialize_storage()
    
    snippets = indexer.extract_snippets()
    relationships = indexer.extract_relationships(snippets)
    
    indexer.save(snippets, relationships)
    indexer.cleanup(snippets)
    
    # 3. Final verification
    indexer.verify()

if __name__ == "__main__":
    main()
