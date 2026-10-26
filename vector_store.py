# Line 1:  Module docstring identifying this as the core data structure for ToyVectorDB.
"""ToyVectorDB - vector_store.py: In-memory vector database with cosine similarity search."""

# Line 3:  Import math for sqrt (needed in cosine similarity magnitude calculation).
import math

# Line 5:  Import dataclass for structured SearchResult objects.
from dataclasses import dataclass

# Line 7:  Import typing for type hints (List, Dict, Optional, Any, Tuple).
from typing import Any, Dict, List, Optional, Tuple


# Line 10: Define cosine_similarity as a standalone function (reusable by any module that needs it).
def cosine_similarity(a: List[float], b: List[float]) -> float:
    # Line 11: Compute the dot product of vectors a and b (sum of element-wise products).
    dot = sum(x * y for x, y in zip(a, b))
    # Line 12: Compute magnitude (L2 norm) of vector a using Pythagorean theorem in N dimensions.
    mag_a = math.sqrt(sum(x * x for x in a))
    # Line 13: Compute magnitude (L2 norm) of vector b.
    mag_b = math.sqrt(sum(y * y for y in b))
    # Line 14: Guard against division by zero — if either vector is all zeros, similarity is undefined; return 0.0.
    if mag_a == 0.0 or mag_b == 0.0:
        # Line 15: Return 0.0 as a safe default when magnitude is zero (avoids ZeroDivisionError).
        return 0.0
    # Line 16: Return the normalized dot product — this is cosine of the angle between vectors, range [-1, 1].
    return dot / (mag_a * mag_b)


# Line 19: Define SearchResult as a dataclass to represent one row returned by a search query.
@dataclass
class SearchResult:
    # Line 20: The unique identifier of the stored vector that matched.
    id: str
    # Line 21: The actual vector values (list of floats) for transparency/debugging.
    vector: List[float]
    # Line 22: Optional metadata dict attached at insert time (e.g., source document, timestamp).
    metadata: Optional[Dict[str, Any]]
    # Line 23: The cosine similarity score between query and this result (higher = more similar).
    score: float


# Line 26: Define the VectorStore class — the central data structure of this module.
class VectorStore:
    # Line 27: Docstring explaining what this class does in plain language.
    """In-memory vector database. Supports insert, top-k similarity search, and delete.

    Think of it as a spreadsheet where each row is (id, [numbers], notes).
    Search = "find the K rows whose numbers are most like my query numbers."
    """

    # Line 32: Constructor initializes an empty dictionary as the backing store.
    def __init__(self):
        # Line 33: _store maps id (str) → tuple of (vector, metadata). Dict gives O(1) lookup by ID.
        self._store: Dict[str, Tuple[List[float], Optional[Dict[str, Any]]]] = {}

    # Line 35: Insert a new vector into the store with validation.
    def insert(
        self, id: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        # Line 36: Check if this ID already exists — raise to prevent silent data corruption from overwrites.
        if id in self._store:
            # Line 37: Raise ValueError with a clear message identifying the duplicate.
            raise ValueError(f"Duplicate ID: '{id}' already exists in store.")
        # Line 39: Validate that vector is actually a list (not None, not a string, etc.).
        if not isinstance(vector, list) or len(vector) == 0:
            # Line 40: Reject empty or non-list vectors with a descriptive error.
            raise ValueError("Vector must be a non-empty list of numbers.")
        # Line 42: Iterate over each element to verify it is numeric (int or float).
        for i, val in enumerate(vector):
            # Line 43: If any element is not int/float (e.g., string, None, bool), reject the whole vector.
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                # Line 44: Raise with index and type info so the caller knows exactly what went wrong.
                raise ValueError(
                    f"Vector element at index {i} is not numeric: {type(val).__name__}"
                )
        # Line 46: Store the vector and metadata under the given ID.
        self._store[id] = (vector, metadata)

    # Line 48: Search for the K most similar vectors to a query vector using cosine similarity.
    def search_top_k(self, query_vector: List[float], k: int = 3) -> List[SearchResult]:
        # Line 49: If the store is empty, return an empty list immediately (no work to do).
        if not self._store:
            # Line 50: Empty result — caller can handle this gracefully.
            return []
        # Line 52: Validate the query vector is a non-empty list of numbers.
        if not isinstance(query_vector, list) or len(query_vector) == 0:
            # Line 53: Reject invalid query vectors before doing any computation.
            raise ValueError("Query vector must be a non-empty list of numbers.")
        # Line 55: Accumulate all (result, score) pairs here before sorting.
        results: List[SearchResult] = []
        # Line 56: Iterate over every stored vector to compute its similarity to the query.
        for id, (vector, metadata) in self._store.items():
            # Line 57: Skip vectors whose dimensionality doesn't match the query (can't compare apples to oranges).
            if len(vector) != len(query_vector):
                # Line 58: Continue to next vector — incompatible dimensions are silently skipped.
                continue
            # Line 60: Compute cosine similarity between query and this stored vector.
            score = cosine_similarity(query_vector, vector)
            # Line 61: Create a SearchResult object bundling id, vector, metadata, and score together.
            results.append(
                SearchResult(id=id, vector=vector, metadata=metadata, score=score)
            )
        # Line 63: Sort all results by score in descending order (most similar first).
        results.sort(key=lambda r: r.score, reverse=True)
        # Line 64: Slice to return only the top K results (k is 1-indexed from best).
        return results[:k]

    # Line 66: Remove a vector by its ID. Returns True if found and removed, False if not found.
    def delete(self, id: str) -> bool:
        # Line 67: Check if the ID exists before attempting deletion.
        if id in self._store:
            # Line 68: Remove the entry from the dictionary.
            del self._store[id]
            # Line 69: Signal success to the caller.
            return True
        # Line 71: ID was not found — return False so caller can log a warning without crashing.
        return False

    # Line 73: Return all stored vectors as a list of SearchResult objects (score set to 0.0 since no query context).
    def get_all(self) -> List[SearchResult]:
        # Line 74: Comprehension that wraps each stored entry into a SearchResult with score=0.0.
        return [
            SearchResult(id=id, vector=vec, metadata=meta, score=0.0)
            for id, (vec, meta) in self._store.items()
        ]

    # Line 76: Implement len() protocol so `len(store)` works naturally.
    def __len__(self) -> int:
        # Line 77: Return the number of vectors currently stored.
        return len(self._store)

    # Line 79: Check whether a given ID exists in the store (useful for conditional logic).
    def contains(self, id: str) -> bool:
        # Line 80: O(1) dictionary membership check.
        return id in self._store

    # Line 82: Return the dimensionality of stored vectors (assumes all are same length; returns 0 if empty).
    def get_dimension(self) -> int:
        # Line 83: If store is empty, there's no vector to measure.
        if not self._store:
            # Line 84: Return 0 as a sentinel for "no vectors stored yet."
            return 0
        # Line 86: Grab the first vector and return its length (all inserted vectors should be same dimension).
        first_vector = next(iter(self._store.values()))[0]
        # Line 87: Return the number of elements in that vector.
        return len(first_vector)
