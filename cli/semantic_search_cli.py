#!/usr/bin/env python3

import argparse

from lib.semantic_search import embed_text, get_model_info


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")
    subparser.add_parser("verify", help="Verify the model is loaded correctly")

    embed_text_parser = subparser.add_parser("embed_text", help="Generate embeddings from text")
    embed_text_parser.add_argument("text", type=str, help="Text to generate embeddings from") 
    
    args = parser.parse_args()
    
    match args.command:
        case "verify":
            try:
                model_info = get_model_info()
                print(f"Model loaded: {model_info['model_name']}")
                print(f"Max sequence length: {model_info['max_seq_length']}")
            except Exception as e:
                print(f"Error: {e}")
                return 1
                
        case "embed_text":
            try:
                text_embedding = embed_text(args.text)
                print(f"Text: {args.text}")
                print(f"First 3 dimensions: {text_embedding[:3]}")
                print(f"Dimensions: {text_embedding.shape[0]}")
            except Exception as e:
                print(f"Error: {e}")
                return 1
                
        case _:
            parser.print_help()
            return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
