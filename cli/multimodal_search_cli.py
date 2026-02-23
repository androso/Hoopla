
import argparse
import sys

from lib.multimodal_search import image_search_command, verify_image_embedding
from lib.search.constants import DEFAULT_SEARCH_LIMIT, SCORE_PRECISION


def _print_results(results: list[dict]) -> None:
    """Pretty-print a list of formatted search results."""
    for idx, res in enumerate(results, 1):
        title = res["title"]
        score = res["score"]
        description = res["document"]
        print(f"{idx}. {title} (similarity: {score:.{SCORE_PRECISION}f})")
        print(f"   {description[:100]}...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    img_parser = subparsers.add_parser(
        "verify_image_embedding",
        help="Print the embedding dimensions for a given image",
    )
    img_parser.add_argument("img_path", type=str, help="Path to the image file")

    img_search = subparsers.add_parser(
        "image_search",
        help="Search movies by image similarity",
    )
    img_search.add_argument("img_path", type=str, help="Path to the query image")
    img_search.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help=f"Max results to return (default: {DEFAULT_SEARCH_LIMIT})",
    )

    args = parser.parse_args()

    try:
        match args.command:
            case "verify_image_embedding":
                verify_image_embedding(args.img_path)
            case "image_search":
                results = image_search_command(args.img_path, limit=args.limit)
                _print_results(results)
            case _:
                parser.print_help()
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()