import os
from dotenv import load_dotenv
from google import genai
from typing import Optional

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.5-flash"

def spellcheck_query(query: str):
    prompt = f"""Fix any spelling errors in this movie search query.

        Only correct obvious typos. Don't change correctly spelled words.

        Query: "{query}"

        If no errors, return the original query.

        Corrected:"""

    response = client.models.generate_content(
        model=model,
        contents=prompt
    )

    corrected = (response.text).strip().strip('"')
    return corrected if corrected else query

def enhance_query(query: str, method: Optional[str] = None):
    match method:
        case "spell":
            return spellcheck_query(query)
        case _:
            return query