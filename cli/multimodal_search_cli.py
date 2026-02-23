
import argparse
from lib.multimodal_search import verify_image_embedding, image_search_command

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    img_parser = subparsers.add_parser(
        "verify_image_embedding",
    )
    img_parser.add_argument("img_path", type=str, help="Path for the image")

    img_search = subparsers.add_parser(
        "image_search"
    )
    img_search.add_argument("img_path", type=str)

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.img_path) 
        case "image_search":
            results = image_search_command(args.img_path)
            for idx, res in enumerate(results, 1):
                doc = res[0]
                score = res[1]
                print(f"{idx}. {doc['title']} (similarity: {score:.3f})")
                print(f"{doc['description'][:100]}...")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()