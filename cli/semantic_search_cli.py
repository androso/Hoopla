#!/usr/bin/env python3
import argparse
from lib.semantic_search import embed_query_text, embed_text, verify_embeddings

from lib.semantic_search import (
    verify_model,
    chunk_text,
    semantic_chunk_text,
    embed_chunks_command,
    search_chunked_command,
    semantic_search
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")
    
    subparser.add_parser("verify")

    embed_text_parser = subparser.add_parser("embed_text", help="Generate embeddings from text")
    embed_text_parser.add_argument("text", type=str, help="Text to generate embeddings from") 

    subparser.add_parser("verify_embeddings", help="Helps verify generated embeddings")
    embed_query_parser = subparser.add_parser("embedquery", help="Helps embedding a query")
    embed_query_parser.add_argument("query", type=str, help="The query to embed")

    search_parser = subparser.add_parser("search", help="starts search")
    search_parser.add_argument("query", type=str, help="text to search")
    search_parser.add_argument("--limit", type=int, help="limit of results", default=5)

    chunking_parser = subparser.add_parser("chunk", help="Start chunking")
    chunking_parser.add_argument("text", type=str, help="Text to chunk")
    chunking_parser.add_argument("--chunk-size", type=int, default=200, help="Size of each chunk in words") 
    chunking_parser.add_argument("--overlap", type=int, help="Overlap between chunks") 

    semantic_chunking_parser = subparser.add_parser("semantic_chunk", help="Start semantic chunking")
    semantic_chunking_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunking_parser.add_argument("--max-chunk-size", type=int, default=4)
    semantic_chunking_parser.add_argument("--overlap", type=int, default=0)

    embed_chunks_parser = subparser.add_parser("embed_chunks", help="generate chunked embeddings")
    
    search_chunked_parser = subparser.add_parser("search_chunked", help="Search chunked")
    search_chunked_parser.add_argument("query", type=str, help="query to use")
    search_chunked_parser.add_argument("--limit", type=int, default=5, help="number of results to return")

    args = parser.parse_args()
    
    match args.command:
        case "verify":
            verify_model()
            
        case "verify_embeddings":
            verify_embeddings()
            
        case "chunk":
            chunk_text(args.text, args.max_chunk_size, args.overlap)

        case "semantic_chunk": 
            semantic_chunk_text(args.text, args.max_chunk_size, args.overlap)

        case "embed_chunks":
            embeddings = embed_chunks_command()
            print(f"Generated {len(embeddings)} chunked embeddings") 

        case "search_chunked":
            result = search_chunked_command(args.query, args.limit)
            print(f"Query: {result["query"]}")
            print("Results:")

            for i, res in enumerate(result["results"], 1):
                print(f"\n{i}. {res['title']} (score: {res['score']:.4f})")
                print(f"       {res['document']}...")

        case "embed_text":
            text = embed_text(args.text)
            print(f"Text: {args.text}")
            print(f"First 3 dimensions: {text[:3]}")
            print(f"Dimensions: {text.shape[0]}")
            
        case "search":
            semantic_search(args.query, args.limit)

        case "embedquery":
            embed_query_text(args.query)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
