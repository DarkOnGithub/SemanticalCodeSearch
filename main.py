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
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--neo4j-user", default="neo4j", help="Neo4j username")
    parser.add_argument("--neo4j-password", default="password", help="Neo4j password")
    parser.add_argument("--no-auto-start", action="store_false", dest="auto_start", help="Disable automatic Neo4j starting via Docker")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose (DEBUG) logging")
    parser.set_defaults(auto_start=True)
    args = parser.parse_args()

    setup_logging(args.verbose)
    
    # 1. Create an indexer instance for this specific folder
    indexer = ProjectIndexer(
        src_path=args.dir, 
        chunk_size=args.chunk_size,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        auto_start_neo4j=args.auto_start
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
