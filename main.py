# Line 1:  Module docstring — this is the demo entry point for ToyVectorDB.
"""ToyVectorDB - main.py: Demo showing insert, search, and delete on a vector store."""

# Line 3:  Import the VectorStore and SearchResult from our core module.
from vector_store import VectorStore, SearchResult, cosine_similarity

# Line 5:  Import the EmbeddingEngine for text-to-vector conversion in the demo.
from embedding import EmbeddingEngine


# Line 8: Define a helper to print a search result in a readable format.
def print_result(result: SearchResult, rank: int) -> None:
    # Line 9: Format and print each result with its rank, ID, score, and metadata.
    print(
        f"    #{rank}  id={result.id:<10}  score={result.score:+.4f}  meta={result.metadata}"
    )


# Line 12: Main execution block — runs the demo when this file is executed directly.
if __name__ == "__main__":
    # Line 13: Print a header to identify which module is running.
    print("=" * 60)
    # Line 14: Module title.
    print("  TOYVECTORDB — Module #6: Store & Search by Similarity")
    # Line 15: Separator line.
    print("=" * 60)

    # ────────────────────────────────────────────────────────────
    #  PHASE 1: Build the embedding engine from a small corpus
    # ────────────────────────────────────────────────────────────
    # Line 20: Define a small corpus of documents (simulating a knowledge base).
    documents = {
        "doc_1": "The cat sat on the mat and purred softly",
        "doc_2": "A dog chased the ball in the park all afternoon",
        "doc_3": "The cat played with a red yarn ball near the window",
        "doc_4": "Python is a programming language used for data science",
        "doc_5": "Machine learning models require large datasets to train",
    }

    # Line 27: Print phase header.
    print("\n[PHASE 1] Building embedding engine from corpus...")
    # Line 28: Extract just the text values (discard IDs for vocabulary building).
    doc_texts = list(documents.values())
    # Line 29: Build the EmbeddingEngine — it learns the vocabulary from all documents.
    engine = EmbeddingEngine.build(doc_texts)
    # Line 30: Print the learned dimensionality so user sees how many "features" we have.
    print(f"    Vocabulary size (vector dimension): {engine.dimension}")
    # Line 31: Print first 10 vocabulary words as a sample.
    print(f"    Sample vocab: {engine.vocabulary[:10]}")

    # ────────────────────────────────────────────────────────────
    #  PHASE 2: Embed documents and insert into VectorStore
    # ────────────────────────────────────────────────────────────
    # Line 36: Print phase header.
    print("\n[PHASE 2] Embedding documents and inserting into store...")
    # Line 37: Create a fresh, empty VectorStore instance.
    store = VectorStore()

    # Line 39: Iterate over each document ID and its text content.
    for doc_id, text in documents.items():
        # Line 40: Convert the text into a numeric vector using the embedding engine.
        vector = engine.embed(text)
        # Line 41: Create metadata dict with the original text (useful for display in results).
        metadata = {"text": text, "length": len(text.split())}
        # Line 42: Insert the vector into the store with its ID and metadata.
        store.insert(doc_id, vector, metadata)
        # Line 43: Print confirmation of insertion with vector magnitude for intuition.
        print(
            f"    INSERTED {doc_id}: dim={len(vector)}, magnitude={sum(x*x for x in vector)**0.5:.4f}"
        )

    # Line 45: Print total count stored to confirm all insertions succeeded.
    print(f"    Store size: {len(store)} vectors")

    # ────────────────────────────────────────────────────────────
    #  PHASE 3: Search by similarity
    # ────────────────────────────────────────────────────────────
    # Line 50: Print phase header.
    print("\n[PHASE 3] Searching by similarity...")

    # Line 52: Define a set of query texts to search with.
    queries = [
        ("pet_cat", "The cat is sleeping on the warm mat"),
        ("pet_dog", "A happy dog runs in the green park"),
        ("tech_ai", "Neural networks and deep learning algorithms"),
    ]

    # Line 57: Iterate over each query to demonstrate different search results.
    for query_id, query_text in queries:
        # Line 58: Print which query we're running.
        print(f"\n    QUERY '{query_id}': \"{query_text}\"")
        # Line 59: Embed the query text into a vector (same engine = same space).
        query_vector = engine.embed(query_text)
        # Line 60: Search for top 3 most similar documents.
        results = store.search_top_k(query_vector, k=3)
        # Line 61: If no results (shouldn't happen here), print a message.
        if not results:
            # Line 62: Inform user that nothing was found.
            print("      (no results)")
        # Line 64: Otherwise, print each result with its rank.
        else:
            # Line 65: Enumerate results starting from rank 1.
            for rank, result in enumerate(results, start=1):
                # Line 66: Print the formatted result.
                print_result(result, rank)

    # ────────────────────────────────────────────────────────────
    #  PHASE 4: Delete a vector and verify
    # ────────────────────────────────────────────────────────────
    # Line 73: Print phase header.
    print("\n[PHASE 4] Deleting a vector...")
    # Line 74: Show store size before deletion.
    print(f"    Before delete: {len(store)} vectors")
    # Line 75: Delete doc_2 (the dog document).
    deleted = store.delete("doc_2")
    # Line 76: Print whether the deletion succeeded.
    print(f"    delete('doc_2') → {deleted}")
    # Line 77: Show store size after deletion.
    print(f"    After delete: {len(store)} vectors")
    # Line 78: Verify doc_2 is gone by checking contains().
    print(f"    contains('doc_2') → {store.contains('doc_2')}")

    # Line 80: Search again with the dog query to show doc_2 no longer appears.
    print("\n    Re-searching for 'dog' after deletion:")
    # Line 81: Embed the dog query.
    dog_query_vector = engine.embed("A happy dog runs in the green park")
    # Line 82: Search top 3.
    results_after = store.search_top_k(dog_query_vector, k=3)
    # Line 83: Print results — doc_2 should be absent.
    for rank, result in enumerate(results_after, start=1):
        # Line 84: Print each remaining result.
        print_result(result, rank)

    # ────────────────────────────────────────────────────────────
    #  PHASE 5: Edge case demonstration — zero vector search
    # ────────────────────────────────────────────────────────────
    # Line 91: Print phase header.
    print("\n[PHASE 5] Edge case: searching with a zero-magnitude vector...")
    # Line 92: Create a zero vector of the same dimension as our store.
    zero_vector = [0.0] * engine.dimension
    # Line 93: Attempt search — should return results all with score 0.0 (no crash).
    zero_results = store.search_top_k(zero_vector, k=3)
    # Line 94: Print the scores to show they're all 0.0 (cosine with zero vector is defined as 0).
    print(f"    Scores: {[f'{r.score:.4f}' for r in zero_results]}")
    # Line 95: Confirm no exception was raised.
    print("    No crash — zero-magnitude handled gracefully.")

    # Line 97: Final separator and summary.
    print("\n" + "=" * 60)
    # Line 98: Summary line.
    print("  DEMO COMPLETE")
    # Line 99: Key takeaway reminder.
    print("  A vector DB = list of vectors + similarity function + sorting.")
    # Line 100: Closing separator.
    print("=" * 60)
