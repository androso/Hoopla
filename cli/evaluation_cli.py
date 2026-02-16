import argparse
from lib.search.loaders import load_dataset, load_movies
from lib.hybrid_search import HybridSearch

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)"
    )

    args = parser.parse_args()
    limit = args.limit

    # run evaluation logic here:
    dataset = load_dataset()
    movies = load_movies()
    instance_search = HybridSearch(movies)
    
    for test_case in dataset["test_cases"]:
        query = test_case["query"]
        relevant_docs = test_case["relevant_docs"]
        results = instance_search.rrf_search(query, 60, limit)
        relevant_docs_found_n = 0
        for result in results:
            if result["title"] in relevant_docs:    
                relevant_docs_found_n += 1
        precision = relevant_docs_found_n / len(results)
        recall = relevant_docs_found_n / len(relevant_docs)
        f1 = 2 * (precision * recall) / (precision + recall)

        print(f"k={limit}")
        print(f"- {query}")
        print(f"    - Precision@{limit}: {precision:.4f}")
        print(f"    - Recall@{limit}: {recall:.4f}")
        print(f"    - F1 Score: {f1}")
        print(f"    - Retrieved: {[result['title'] for result in results]}")
        print(f"    - Relevant: {[doc for doc in relevant_docs]}")

if __name__ == "__main__":
    main()