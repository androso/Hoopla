import os
from dotenv import load_dotenv
from google import genai
from typing import Optional
import json 
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-3-flash-preview"

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

def rewrite_query(query:str):
    prompt = f"""Rewrite this movie search query to be more specific and searchable.

    Original: "{query}"

    Consider:
    - Common movie knowledge (famous actors, popular films)
    - Genre conventions (horror = scary, animation = cartoon)
    - Keep it concise (under 10 words)
    - It should be a google style search query that's very specific
    - Don't use boolean logic

    Examples:

    - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
    - "movie about bear in london with marmalade" -> "Paddington London marmalade"
    - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

    Rewritten query:"""
    
    response = client.models.generate_content(
        model=model,
        contents=prompt
    )

    corrected = (response.text).strip().strip('"')
    return corrected if corrected else query

def expand_query(query:str):
    prompt = f"""Expand this movie search query with related terms.

    Add synonyms and related concepts that might appear in movie descriptions.
    Keep expansions relevant and focused.
    This will be appended to the original query.

    Examples:

    - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
    - "action movie with bear" -> "action thriller bear chase fight adventure"
    - "comedy with bear" -> "comedy funny bear humor lighthearted"

    Query: "{query}"
    """
    
    response = client.models.generate_content(
        model=model,
        contents=prompt
    )

    corrected = (response.text).strip().strip('"')
    return corrected if corrected else query


def score_batch_documents(query: str, docs):
    doc_list_str = json.dumps(docs, indent=2)
    prompt = f"""Rank these movies by relevance to the search query.

    Query: "{query}"

    Movies:
    {doc_list_str}

    Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

    [75, 12, 34, 2, 1]
    """

    response = client.models.generate_content(
        model=model,
        contents=prompt
    )
    corrected = (response.text).strip().strip('"')
    ids_list = json.loads(corrected)
    id_to_position = {id_val: idx for idx, id_val in enumerate(ids_list)}
    
    return sorted(docs, key=lambda doc: id_to_position.get(doc['id'], float('inf')))

def score_document(query: str, doc):
    prompt = f"""Rate how well this movie matches the search query.

    Query: "{query}"
    Movie: {doc.get("title", "")} - {doc.get("document", "")}

    Consider:
    - Direct relevance to query
    - User intent (what they're looking for)
    - Content appropriateness

    Rate 0-10 (10 = perfect match).
    Give me ONLY the number in your response, no other text or explanation.

    Score:"""

    response = client.models.generate_content(
        model=model,
        contents=prompt
    )

    corrected = int((response.text).strip().strip('"'))
    return corrected if corrected else 0

def evaluate_results(query: str, results):
    formatted_results = []
    for res in results:
        formatted_results.append({
            "title": res["title"],
            "document": res["document"]
        })
    formatted_results = json.dumps(formatted_results)
    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

    Query: "{query}"

    Results:
    {chr(10).join(formatted_results)}

    Scale:
    - 3: Highly relevant
    - 2: Relevant
    - 1: Marginally relevant
    - 0: Not relevant

    Do NOT give any numbers out than 0, 1, 2, or 3.

    Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

    [2, 0, 3, 2, 0, 1]""" 
    
    response = client.models.generate_content(
        model=model,
        contents=prompt
    )
    
    corrected = (response.text).strip().strip('"')
    scores = json.loads(corrected)
    for doc_idx, doc in enumerate(results):
        doc["metadata"]["llm_score"] = scores[doc_idx] 

    return results

def enhance_query(query: str, method: Optional[str] = None):
    match method:
        case "spell":
            return spellcheck_query(query)
        case "rewrite":
            return rewrite_query(query)
        case "expand":
            return expand_query(query)
        case _:
            return query
    

def generate_answer(query: str, docs):
    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

    Query: {query}

    Documents:
    {docs}

    Provide a comprehensive answer that addresses the query:"""
    response = client.models.generate_content(
        model=model,
        contents=prompt
    )


    return response.text 

def generate_summary(query: str, search_results, limit=5):
    docs_text = ""
    for i, result in enumerate(search_results[:limit], start=1):
        docs_text += f"Document {i}: {result['title']}; {result['document']}\n\n"

    prompt = f"""
        Provide information useful to this query by synthesizing information from multiple search results in detail.
        The goal is to provide comprehensive information so that users know what their options are.
        Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
        This should be tailored to Hoopla users. Hoopla is a movie streaming service.
        Query: {query}
        Search Results:
        {docs_text}
        Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
    """

    response = client.models.generate_content(
        model=model,
        contents=prompt
    )

    return (response.text or "").strip()