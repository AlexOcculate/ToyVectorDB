# Line 1:  Module docstring — this file provides a minimal text-to-vector engine for demo purposes.
"""ToyVectorDB - embedding.py: Minimal bag-of-words embedding (re-implemented from Module #5 pattern)."""

# Line 3:  Import math for potential future use (kept for consistency with vector_store.py).
import math

# Line 5:  Import Counter from collections for efficient word frequency counting.
from collections import Counter

# Line 7:  Import typing for type hints.
from typing import List, Optional


# Line 10: Define the EmbeddingEngine class — converts raw text into fixed-length numeric vectors.
class EmbeddingEngine:
    # Line 11: Docstring explaining the approach in plain language.
    """Converts text to numeric vectors using Term Frequency (bag of words).

    Analogy: Imagine each word in your vocabulary is a column in a spreadsheet.
    Each document is a row where cell values = how often that word appears (normalized).
    Two documents with similar topics will have similar rows → high cosine similarity.
    """

    # Line 17: Constructor accepts an optional pre-built vocabulary list.
    def __init__(self, vocabulary: Optional[List[str]] = None):
        # Line 18: Store the vocabulary (ordered list of unique words) — this defines vector dimensionality.
        self._vocabulary: List[str] = vocabulary if vocabulary is not None else []
        # Line 19: Build a lookup map from word → column index for O(1) access during embedding.
        self._word_to_index: dict = {w: i for i, w in enumerate(self._vocabulary)}

    # Line 21: Class method to build an EmbeddingEngine from a corpus of documents (learns vocabulary).
    @classmethod
    def build(cls, documents: List[str]) -> "EmbeddingEngine":
        # Line 22: Collect all unique words across all documents into a set.
        all_words = set()
        # Line 23: Iterate over each document in the corpus.
        for doc in documents:
            # Line 24: Tokenize by lowercasing and splitting on whitespace (simple but effective for demo).
            words = doc.lower().split()
            # Line 25: Add all tokens from this document to the global word set.
            all_words.update(words)
        # Line 26: Sort the vocabulary alphabetically to ensure deterministic vector ordering across runs.
        vocab = sorted(all_words)
        # Line 27: Create and return a new EmbeddingEngine with the learned vocabulary.
        return cls(vocabulary=vocab)

    # Line 29: Convert a single text string into a vector of term frequencies.
    def embed(self, text: str) -> List[float]:
        # Line 30: If no vocabulary has been built yet, return an empty vector (caller should handle this).
        if not self._vocabulary:
            # Line 31: Empty vector signals "engine not initialized" — prevents silent zero vectors.
            return []
        # Line 33: Tokenize the input text (lowercase + split on whitespace).
        words = text.lower().split()
        # Line 34: Count occurrences of each word using Counter (efficient single-pass counting).
        counts: Counter = Counter(words)
        # Line 35: Total number of tokens — used to normalize frequencies into [0, 1] range.
        total = len(words) if words else 1
        # Line 36: Initialize the output vector with zeros, one slot per vocabulary word.
        vector: List[float] = [0.0] * len(self._vocabulary)
        # Line 37: For each word that appeared in the text, fill in its column.
        for word, count in counts.items():
            # Line 38: Only process words that are in our vocabulary (out-of-vocab words get 0).
            if word in self._word_to_index:
                # Line 39: Set the value to term frequency = count / total_tokens (normalized proportion).
                vector[self._word_to_index[word]] = count / total
        # Line 41: Return the completed vector — ready to be stored or compared.
        return vector

    # Line 43: Property that returns the dimensionality of produced vectors (= vocabulary size).
    @property
    def dimension(self) -> int:
        # Line 44: The number of unique words in the vocabulary equals the vector length.
        return len(self._vocabulary)

    # Line 46: Return the vocabulary list (useful for debugging or display).
    @property
    def vocabulary(self) -> List[str]:
        # Line 47: Return a copy to prevent external mutation of internal state.
        return list(self._vocabulary)
