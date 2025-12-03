#!/usr/bin/env python3

import argparse

from lib.search_utils import load_movies
from lib.semantic_search import embed_query_text, embed_text, get_model_info, get_semantic_search, verify_embeddings

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
    args = parser.parse_args()
    
    match args.command:
        case "verify":
            info = get_model_info()
            print(f"Model loaded: {info['model_name']}")
            print(f"Max sequence length: {info['max_seq_length']}")
            
        case "verify_embeddings":
            verify_embeddings()
            
        case "chunk":
            words = args.text.split()
            chunks = []
            
            if args.overlap and args.overlap >= args.chunk_size:
                print(f"Error: overlap ({args.overlap}) must be less than chunk_size ({args.chunk_size})")
                return
            
            if args.overlap and args.overlap > 0:
                step = args.chunk_size - args.overlap
                for chunk_idx in range(0, len(words), step):
                   chunk = " ".join(words[chunk_idx:chunk_idx + args.chunk_size])
                   chunks.append(chunk)
            else:
                for chunk_idx in range(0, len(words), args.chunk_size):
                   chunk = " ".join(words[chunk_idx:chunk_idx + args.chunk_size])
                   chunks.append(chunk)
                    
            print(f"Chunking {len(args.text)} characters")
            for idx, chunk in enumerate(chunks, 1):
                print(f"{idx}. {chunk}")
                
        case "embed_text":
            text = embed_text(args.text)
            print(f"Text: {args.text}")
            print(f"First 3 dimensions: {text[:3]}")
            print(f"Dimensions, {text.shape[0]}")
            
        case "search":
            search = get_semantic_search()
            movies = load_movies()
            search.load_or_create_embeddings(movies)
            results = search.search(args.query, args.limit)
            
            for idx, result in enumerate(results):
               print(f"{idx + 1}. {result['title']} (score: {result['score']})") 
                
        case "embedquery":
            embed_query_text(args.query)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()