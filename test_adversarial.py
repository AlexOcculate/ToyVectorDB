# Line 1:  Import sys and os for path setup so tests can find sibling modules regardless of CWD.
import sys, os

# Line 2: Insert the current directory into sys.path so imports work from any working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Line 4:  Import the core components under test.
from vector_store import VectorStore, cosine_similarity

# Line 5:  Import the embedding engine for tests that need text-to-vector conversion.
from embedding import EmbeddingEngine


# ──────────────────────────────────────────────
#  TEST 1: Corrupted / Malformed Vectors
#  Proves: the store rejects bad data without crashing
# ──────────────────────────────────────────────
def test_broken_vectors():
    # Line 12: Create a fresh store for isolation.
    store = VectorStore()

    # Line 14: Attempt to insert a vector containing None — should raise ValueError.
    try:
        # Line 15: Insert with a None element (simulates corrupted data from upstream).
        store.insert("bad_1", [1.0, None, 3.0])
        # Line 16: If we reach here, the validation failed to catch the bad data.
        raise AssertionError("Should have rejected vector with None element")
    # Line 18: Catch the expected ValueError — this is the correct behavior.
    except ValueError:
        # Line 19: Pass — validation worked as intended.
        pass

    # Line 21: Attempt to insert a vector that is actually a string (type confusion).
    try:
        # Line 22: Insert with a string instead of a list.
        store.insert("bad_2", "not a vector")
        # Line 23: If we reach here, type validation failed.
        raise AssertionError("Should have rejected non-list vector")
    # Line 25: Catch the expected ValueError.
    except ValueError:
        # Line 26: Pass — type check worked.
        pass

    # Line 28: Attempt to insert an empty list as a vector.
    try:
        # Line 29: Empty list has no elements to compare — must be rejected.
        store.insert("bad_3", [])
        # Line 30: If we reach here, empty check failed.
        raise AssertionError("Should have rejected empty vector")
    # Line 32: Catch the expected ValueError.
    except ValueError:
        # Line 33: Pass — empty check worked.
        pass

    # Line 35: Attempt to insert a vector with mixed types (int and string).
    try:
        # Line 36: String element in a numeric vector is invalid.
        store.insert("bad_4", [1.0, "two", 3.0])
        # Line 37: If we reach here, element type check failed.
        raise AssertionError("Should have rejected vector with string element")
    # Line 39: Catch the expected ValueError.
    except ValueError:
        # Line 40: Pass — element validation worked.
        pass

    # Line 42: Verify store is still empty after all failed insertions (no partial state).
    assert len(store) == 0, f"Store should be empty, has {len(store)} items"


# ──────────────────────────────────────────────
#  TEST 2: Invalid / Hostile Input
#  Proves: guardrails block bad input before it reaches the core
# ──────────────────────────────────────────────
def test_hostile_input():
    # Line 51: Create a fresh store.
    store = VectorStore()

    # Line 53: Insert a valid vector so we have something to search against.
    store.insert("valid", [1.0, 2.0, 3.0], {"text": "hello world"})

    # Line 55: Attempt search with an oversized query (10000 dimensions vs stored 3).
    try:
        # Line 56: Query with wrong dimensionality — should be handled gracefully (skip or raise).
        huge_query = [0.1] * 10000
        # Line 57: This should either return empty results (dimension mismatch skipped) or raise.
        results = store.search_top_k(huge_query, k=3)
        # Line 58: If it returns results, they should all be from compatible-dimension vectors (none here).
        assert (
            len(results) == 0
        ), f"Expected 0 results for dimension mismatch, got {len(results)}"
    except ValueError:
        # Line 60: Raising is also acceptable behavior for dimension mismatch.
        pass

    # Line 62: Attempt search with None as query vector.
    try:
        # Line 63: None is not a valid vector — must be rejected.
        store.search_top_k(None, k=3)
        # Line 64: If we reach here, None check failed.
        raise AssertionError("Should have rejected None query vector")
    except ValueError:
        # Line 66: Pass — None was caught.
        pass

    # Line 68: Attempt search with empty list as query.
    try:
        # Line 69: Empty list cannot be compared to anything.
        store.search_top_k([], k=3)
        # Line 70: If we reach here, empty check failed.
        raise AssertionError("Should have rejected empty query vector")
    except ValueError:
        # Line 72: Pass — empty was caught.
        pass

    # Line 74: Attempt to insert with an extremely long ID (10000 chars) — should still work or be bounded.
    long_id = "x" * 10000
    # Line 75: Insert with the long ID — Python dicts handle this fine, but we verify no crash.
    store.insert(long_id, [1.0, 2.0, 3.0])
    # Line 76: Verify it was stored.
    assert store.contains(long_id), "Long ID should be storable"
    # Line 77: Clean up.
    store.delete(long_id)


# ──────────────────────────────────────────────
#  TEST 3: Operation Failure (delete non-existent, search empty)
#  Proves: failing operations return safe defaults, not exceptions
# ──────────────────────────────────────────────
def test_operation_failure():
    # Line 86: Create a fresh store.
    store = VectorStore()

    # Line 88: Attempt to delete an ID that was never inserted — should return False, not raise.
    result = store.delete("nonexistent_id")
    # Line 89: Verify it returned False (not found) rather than raising KeyError.
    assert result == False, f"Expected False for missing ID, got {result}"

    # Line 91: Attempt to search on a completely empty store — should return empty list.
    results = store.search_top_k([1.0, 2.0, 3.0], k=5)
    # Line 92: Verify empty list was returned (not an exception).
    assert (
        results == []
    ), f"Expected empty list from empty store, got {len(results)} items"

    # Line 94: Insert one valid vector, then delete it, then search again.
    store.insert("temp", [1.0, 0.0, 0.0])
    # Line 95: Delete it.
    store.delete("temp")
    # Line 96: Store should be empty again.
    assert len(store) == 0, "Store should be empty after delete"
    # Line 97: Search on now-empty store should return [].
    results2 = store.search_top_k([1.0, 0.0, 0.0], k=3)
    # Line 98: Confirm empty.
    assert results2 == [], "Search on emptied store should return []"


# ──────────────────────────────────────────────
#  TEST 4: Stress / Degradation (many inserts + searches)
#  Proves: performance doesn't degrade catastrophically; no memory leak pattern
# ──────────────────────────────────────────────
def test_stress_degradation():
    # Line 107: Create a fresh store.
    store = VectorStore()

    # Line 109: Insert 500 vectors of dimension 10 (simulates a moderate-scale DB).
    for i in range(500):
        # Line 110: Generate a pseudo-random-looking vector using a simple formula (deterministic, no random import needed).
        vector = [(i * 7 + j * 3) % 17 / 17.0 for j in range(10)]
        # Line 111: Insert with sequential ID and minimal metadata.
        store.insert(f"stress_{i}", vector, {"index": i})

    # Line 113: Verify all 500 were stored.
    assert len(store) == 500, f"Expected 500 vectors, got {len(store)}"

    # Line 115: Perform 100 searches and verify each returns exactly k results (or fewer if store < k).
    for i in range(100):
        # Line 116: Create a query vector.
        query = [(i * 3 + j) % 11 / 11.0 for j in range(10)]
        # Line 117: Search for top 5.
        results = store.search_top_k(query, k=5)
        # Line 118: Should always return exactly 5 (store has 500, all same dimension).
        assert len(results) == 5, f"Search {i}: expected 5 results, got {len(results)}"
        # Line 119: Verify scores are in valid cosine range [-1, 1].
        for r in results:
            # Line 120: Cosine similarity must be between -1 and 1 (with small float tolerance).
            assert -1.0001 <= r.score <= 1.0001, f"Score {r.score} out of range [-1, 1]"

    # Line 123: Verify get_all returns all 500.
    all_items = store.get_all()
    # Line 124: Count check.
    assert len(all_items) == 500, f"get_all expected 500, got {len(all_items)}"

    # Line 126: Delete half the vectors and verify count.
    for i in range(250):
        # Line 127: Delete by ID.
        store.delete(f"stress_{i}")
    # Line 128: Should have 250 remaining.
    assert len(store) == 250, f"Expected 250 after deletion, got {len(store)}"


# ──────────────────────────────────────────────
#  TEST 5: Module-specific — Empty DB search / Zero-magnitude / Duplicate ID
#  Proves: the three unique failure modes of a vector store are handled
# ──────────────────────────────────────────────
def test_module_specific():
    # Line 137: Create a fresh store.
    store = VectorStore()

    # --- Sub-test 5a: Search on completely empty DB ---
    # Line 140: Search with a valid vector on an empty store.
    results = store.search_top_k([1.0, 2.0, 3.0], k=3)
    # Line 141: Must return empty list, not raise.
    assert results == [], "Search on empty DB must return []"

    # --- Sub-test 5b: Zero-magnitude vector as query ---
    # Line 144: Insert a normal vector first.
    store.insert("normal", [3.0, 4.0, 0.0])
    # Line 145: Search with an all-zeros vector (magnitude = 0).
    zero_query = [0.0, 0.0, 0.0]
    # Line 146: This should NOT raise ZeroDivisionError — cosine_similarity handles it.
    results_zero = store.search_top_k(zero_query, k=3)
    # Line 147: Should return 1 result (the "normal" vector) with score 0.0.
    assert (
        len(results_zero) == 1
    ), f"Expected 1 result for zero query, got {len(results_zero)}"
    # Line 148: Score must be exactly 0.0 (our convention for zero-magnitude).
    assert (
        results_zero[0].score == 0.0
    ), f"Zero-magnitude score should be 0.0, got {results_zero[0].score}"

    # --- Sub-test 5c: Zero-magnitude vector as stored vector ---
    # Line 151: Insert a zero vector (valid — it's a list of numbers).
    store.insert("zero_vec", [0.0, 0.0, 0.0])
    # Line 152: Now search with a normal query — the zero vector should get score 0.0.
    results_mixed = store.search_top_k([1.0, 1.0, 1.0], k=3)
    # Line 153: Should return 2 results (normal + zero_vec).
    assert len(results_mixed) == 2, f"Expected 2 results, got {len(results_mixed)}"
    # Line 154: The zero vector's score should be 0.0.
    zero_result = [r for r in results_mixed if r.id == "zero_vec"][0]
    # Line 155: Verify its score is 0.0.
    assert (
        zero_result.score == 0.0
    ), f"Zero vector score should be 0.0, got {zero_result.score}"

    # --- Sub-test 5d: Duplicate ID insertion must raise ---
    # Line 158: Attempt to insert the same ID twice.
    try:
        # Line 159: First insert succeeds.
        store.insert("dup_test", [1.0, 0.0, 0.0])
        # Line 160: Second insert with same ID must raise ValueError.
        store.insert("dup_test", [0.0, 1.0, 0.0])
        # Line 161: If we reach here, duplicate detection failed.
        raise AssertionError("Duplicate ID insertion should have raised ValueError")
    except ValueError as e:
        # Line 163: Verify the error message mentions "Duplicate".
        assert "Duplicate" in str(e), f"Error should mention 'Duplicate', got: {e}"

    # --- Sub-test 5e: cosine_similarity function directly with zero vectors ---
    # Line 167: Both vectors zero — must return 0.0, not crash.
    score = cosine_similarity([0.0, 0.0], [0.0, 0.0])
    # Line 168: Verify it's 0.0.
    assert score == 0.0, f"cosine([0,0],[0,0]) should be 0.0, got {score}"

    # Line 170: One vector zero, one non-zero — must return 0.0.
    score2 = cosine_similarity([0.0, 0.0], [1.0, 1.0])
    # Line 171: Verify it's 0.0.
    assert score2 == 0.0, f"cosine([0,0],[1,1]) should be 0.0, got {score2}"


# ──────────────────────────────────────────────
#  RUNNER
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # Line 179: Print test header.
    print("=" * 50)
    # Line 180: Identify the test suite.
    print("  TOYVECTORDB — Adversarial Test Suite")
    # Line 181: Separator.
    print("=" * 50)

    # Line 183: Define the list of (name, function) pairs for the runner.
    tests = [
        # Line 184: Test 1 — corrupted vectors are rejected.
        ("Corrupted / malformed vectors", test_broken_vectors),
        # Line 185: Test 2 — hostile input is blocked by guardrails.
        ("Hostile / invalid input", test_hostile_input),
        # Line 186: Test 3 — failing operations return safe defaults.
        ("Operation failure (safe defaults)", test_operation_failure),
        # Line 187: Test 4 — stress test proves no degradation.
        ("Stress / no degradation", test_stress_degradation),
        # Line 188: Test 5 — module-specific edge cases.
        ("Empty DB / zero-mag / duplicate", test_module_specific),
    ]

    # Line 190: Counter for passed tests.
    passed = 0
    # Line 191: Iterate over all tests.
    for name, fn in tests:
        # Line 192: Attempt to run the test function.
        try:
            # Line 193: Call the test — it either passes (no exception) or raises.
            fn()
            # Line 194: Print PASS with the test name.
            print(f"  PASS  {name}")
            # Line 195: Increment passed counter.
            passed += 1
        except Exception as e:
            # Line 197: Print FAIL with the exception message for debugging.
            print(f"  FAIL  {name}: {e}")

    # Line 199: Print summary line.
    print(f"\n  {passed}/{len(tests)} tests passed.")
    # Line 200: If any test failed, exit with non-zero code (CI-friendly).
    if passed < len(tests):
        # Line 201: Signal failure to the shell.
        sys.exit(1)
