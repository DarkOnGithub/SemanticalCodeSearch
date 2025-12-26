import argparse
import os
from src.pipeline import IndexingPipeline

def main():
    parser = argparse.ArgumentParser(description="Semantical Code Search Indexer")
    parser.add_argument("dir", nargs="?", default=os.getcwd(), help="Directory to index")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for parsing")
    args = parser.parse_args()

    pipeline = IndexingPipeline(chunk_size=args.chunk_size)
    pipeline.run(args.dir)

if __name__ == "__main__":
    main()
