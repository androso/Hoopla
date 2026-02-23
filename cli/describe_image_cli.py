import argparse
import mimetypes
from google.genai import types
from gemini import client, model

def main():
    parser = argparse.ArgumentParser(description="Describe an image using a text query")
    parser.add_argument("--image", required=True, help="Path to the image file")
    parser.add_argument("--query", required=True, help="Text query to describe the image")

    args = parser.parse_args()


    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    with open(args.image, "rb") as f:
        image_data = f.read()
        prompt = f"""Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
            - Synthesize visual and textual information
            - Focus on movie-specific details (actors, scenes, style, etc.)
            - Return only the rewritten query, without any additional commentary"""
        parts = [
            prompt,
            types.Part.from_bytes(data=image_data, mime_type=mime),
            args.query.strip()
        ]

        response = client.models.generate_content(model=model, contents=parts)
        new_query = (response.text).strip()
        print(f"Rewritten query: {response.text.strip()}")
        if response.usage_metadata is not None:
            print(f"Total tokens: {response.usage_metadata.total_token_count}")


    print(f"Image: {args.image}")
    print(f"Query: {args.query}")

if __name__ == "__main__":
    main()