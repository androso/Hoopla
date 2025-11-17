#!/usr/bin/env python3

import argparse

from lib.semantic_search import embed_text, get_model_info

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")
    subparser.add_parser("verify")

    embed_text_parser = subparser.add_parser("embed_text", help="Generate embeddings from text")
    embed_text_parser.add_argument("text", type=str, help="Text to generate embeddings from") 
    
    args = parser.parse_args()
    
    match args.command:
        case "verify":
            info = get_model_info()
            print(f"Model loaded: {info["model_name"]}")
            print(f"Max sequence length: {info["max_seq_length"]}")
        case "embed_text":
            text = embed_text(args.text)
            print(f"Text: {args.text}")
            print(f"First 3 dimensions: {text[:3]}")
            print(f"Dimensions, {text.shape[0]}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()