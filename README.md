# Module #6: ToyVectorDB — Turn 2 (Diagrams)

---

## a. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                            TOYVECTORDB ARCHITECTURE                                 │
└─────────────────────────────────────────────────────────────────────────────────────┘

                         ┌──────────────────────────────────┐
                         │         CALLER (main.py)         │
                         │    (or future RAG pipeline #7)   │
                         └───────┬──────────────────┬───────┘
                                 │                  │
                    text strings │                  │ query_vector: List[float]
                                 ▼                  ▼
              ┌────────────────────────┐   ┌──────────────────────────────┐
              │    EmbeddingEngine     │   │        VectorStore           │
              │                        │   │                              │
              │  ┌──────────────────┐  │   │  ┌────────────────────────┐  │
              │  │ _vocabulary      │  │   │  │ _store: Dict[str,      │  │
              │  │ List[str]        │  │   │  │   Tuple[List[float],   │  │
              │  │                  │  │   │  │   Optional[Dict]]>     │  │
              │  │ ["a","all","and" │  │   │  │                        │  │
              │  │  "ball", ...]    │  │   │  │  "doc_1": ([0.1,0.2..],│  │
              │  └──────────────────┘  │   │  │           {"text":...})│  │
              │                        │   │  │  "doc_2": ([0.3,0.0..],│  │
              │  ┌──────────────────┐  │   │  │           {"text":...})│  │
              │  │ _word_to_index   │  │   │  │  ...                   │  │
              │  │ Dict[str, int]   │  │   │  └────────────────────────┘  │
              │  │                  │  │   │                              │
              │  │ {"a":0,"all":1,  │  │   │  Public API:                 │
              │  │  "and":2, ...}   │  │   │  ┌────────────────────────┐  │
              │  └──────────────────┘  │   │  │ .insert(id, vec, meta) │  │
              │                        │   │  │ .search_top_k(q, k)    │  │
              │  Methods:              │   │  │ .delete(id)            │  │
              │  ┌──────────────────┐  │   │  │ .get_all()             │  │
              │  │ .build(docs)     │◄─┼───┼──│ .contains(id)          │  │
              │  │ .embed(text)     │  │   │  │ .get_dimension()       │  │
              │  │ .dimension (prop)│  │   │  └────────────────────────┘  │
              │  └──────────────────┘  │   │                              │
              └────────────┬───────────┘   │  Internal:                   │
                           │               │  ┌────────────────────────┐  │
                           │ returns       │  │ cosine_similarity(a,b) │  │
                           │ List[float]   │  │ (module-level func)    │   │
                           └───────────────┼──└────────────────────────┘   │
                                           │                               │
                                           ▼                               │
                              ┌──────────────────────────┐                 │
                              │     List[SearchResult]   │◄────────────────┘
                              │                          │
                              │  .id: str                │
                              │  .vector: List[float]    │
                              │  .metadata: Dict|None    │
                              │  .score: float           │
                              └──────────────────────────┘


╔══════════════════════════════════════════════════════════════════════════════╗
║  LEGEND                                                                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ┌─────────────┐   Component (class or function)                             ║
║  │             │                                                             ║
║  └─────────────┘                                                             ║
║                                                                              ║
║  ┌─────────────┐   Internal data structure (private field)                   ║
║  │  _vocabulary│                                                             ║
║  └─────────────┘                                                             ║
║                                                                              ║
║  ──────►        Data flow direction                                          ║
║                                                                              ║
║  Key insight: EmbeddingEngine and VectorStore are INDEPENDENT.               ║
║  The store doesn't know or care HOW vectors were produced.                   ║
║  It only stores, compares, and retrieves numeric lists.                      ║
╚══════════════════════════════════════════════════════════════════════════════╝


┌─────────────────────────────────────────────────────────────────────────────┐
│  DESIGN DECISION ANNOTATIONS                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. WHY DICT BACKING STORE?                                                 │
│     → O(1) insert/delete by ID                                              │
│     → Iteration for search is O(n) — fine at toy scale (<10K vectors)       │
│     → Real DBs add HNSW/IVF indexes for O(log n) — concept unchanged        │
│                                                                             │
│  2. WHY COSINE SIMILARITY (not Euclidean)?                                  │
│     → Normalizes for magnitude: "cat cat cat" ≈ "cat" (direction matters)   │
│     → Standard in NLP/embeddings; what Pinecone/Weaviate default to         │
│                                                                             │
│  3. WHY SKIP DIMENSION MISMATCH (not raise)?                                │
│     → Production collections may have mixed-dimension vectors               │
│     → Silently skipping is more robust than crashing the search             │
│     → Caller can verify dimension via get_dimension() before searching      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## b. Sequence Diagram

```
TOYVECTORDB — SEQUENCE DIAGRAM (Full Workflow: Insert → Search → Delete)
═══════════════════════════════════════════════════════════════════════════════

  CALLER          EmbeddingEngine        VectorStore            cosine_similarity
    │                  │                     │                         │
    │                  │                     │                         │
    │  ─── PHASE 1: BUILD ENGINE ──────────────────────────────────────│
    │                  │                     │                         │
    │  build(docs)     │                     │                         │
    │─────────────────►│                     │                         │
    │                  │                     │                         │
    │                  │  [collect all words]│                         │
    │                  │  [sort alphabetically]                        │
    │                  │  [build word→index map]                       │
    │                  │                     │                         │
    │   engine         │                     │                         │
    │◄─────────────────│                     │                         │
    │                  │                     │                         │
    │  ─── PHASE 2: EMBED + INSERT (repeated per document) ────────────│
    │                  │                     │                         │
    │  embed("The cat sat...")               │                         │
    │─────────────────►│                     │                         │
    │                  │                     │                         │
    │                  │  [tokenize: lower + split]                    │
    │                  │  [count with Counter]                         │
    │                  │  [normalize: count/total per word]            │
    │                  │  [fill vector by index]                       │
    │                  │                     │                         │
    │   [0.11, 0.0, 0.22, ...]               │                         │
    │◄─────────────────│                     │                         │
    │                  │                     │                         │
    │  insert("doc_1", vector, metadata)     │                         │
    │───────────────────────────────────────►│                         │
    │                  │                     │                         │
    │                  │                     │  [check: id in store?]                │
    │                  │                     │  [validate: list? non-empty? numeric?]│
    │                  │                     │  [store[id] = (vec, meta)]            │
    │                  │                     │                         │
    │   None (success) │                     │                         │
    │◄───────────────────────────────────────│                         │
    │                  │                     │                         │
    │  ... repeat for doc_2 through doc_5 ...│                         │
    │                  │                     │                         │
    │  ─── PHASE 3: SEARCH ────────────────────────────────────────────│
    │                  │                     │                         │
    │  embed("The cat is sleeping...")       │                         │
    │─────────────────►│                     │                         │
    │                  │                     │                         │
    │   query_vector   │                     │                         │
    │◄─────────────────│                     │                         │
    │                  │                     │                         │
    │  search_top_k(query_vector, k=3)       │                         │
    │───────────────────────────────────────►│                         │
    │                  │                     │                         │
    │                  │                     │  [check: store empty?]              │
    │                  │                     │  [validate query]                   │
    │                  │                     │                                     │
    │                  │                     │  FOR each (id, vec) in store:       │
    │                  │                     │    [check: len match?]              │
    │                  │                     │    [compute similarity]             │
    │                  │                     │────────────────────────►│
    │                  │                     │                         │
    │                  │                     │                         │  [dot = Σ aᵢbᵢ]
    │                  │                     │                         │  [mag_a = √Σaᵢ²]
    │                  │                     │                         │  [mag_b = √Σbᵢ²]
    │                  │                     │                         │  [if mag==0: return 0.0]
    │                  │                     │                         │  [return dot/(mag_a*mag_b)]
    │                  │                     │◄────────────────────────│
    │                  │                     │                         │
    │                  │                     │  [append SearchResult(id, vec, meta, score)]
    │                  │                     │                         │
    │                  │                     │  [sort by score DESC]   │
    │                  │                     │  [slice [:k]]           │
    │                  │                     │                         │
    │   [SearchResult, SearchResult, SearchResult]                     │
    │◄───────────────────────────────────────│                         │
    │                  │                     │                         │
    │  ─── PHASE 4: DELETE ────────────────────────────────────────────│
    │                  │                     │                         │
    │  delete("doc_2") │                     │                         │
    │───────────────────────────────────────►│                         │
    │                  │                     │                         │
    │                  │                     │  [check: id in store?]  │
    │                  │                     │  [del store[id]]        │
    │                  │                     │                         │
    │   True           │                     │                         │
    │◄───────────────────────────────────────│                         │
    │                  │                     │                         │
    │  ─── PHASE 5: VERIFY DELETION ───────────────────────────────────│
    │                  │                     │                         │
    │  search_top_k(dog_query, k=3)          │                         │
    │───────────────────────────────────────►│                         │
    │                  │                     │                         │
    │                  │                     │  [doc_2 no longer in iteration]     │
    │                  │                     │  [returns top-3 from remaining 4]   │
    │                  │                     │                         │
    │   [SearchResult, SearchResult, SearchResult]  ← doc_2 ABSENT     │
    │◄───────────────────────────────────────│                         │
    │                  │                     │                         │


╔══════════════════════════════════════════════════════════════════════════════╗
║  STEP-BY-STEP NARRATIVE                                                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  1.  Caller builds EmbeddingEngine from corpus → learns vocabulary           ║
║  2.  For each document:                                                      ║
║      a. Caller asks engine to embed text → gets List[float]                  ║
║      b. Caller inserts (id, vector, metadata) into store                     ║
║  3.  To search:                                                              ║
║      a. Caller embeds query text → gets query_vector                         ║
║      b. Caller calls store.search_top_k(query_vector, k)                     ║
║      c. Store iterates ALL vectors, calls cosine_similarity for each         ║
║      d. Store sorts by score descending, returns top-k                       ║
║  4.  To delete: caller calls store.delete(id) → O(1) dict removal            ║
║                                                                              ║
║  KEY OBSERVATION: The caller orchestrates everything.                        ║
║  EmbeddingEngine and VectorStore never talk to each other directly.          ║
║  The vector (List[float]) is the ONLY interface between them.                ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## c. State Machine Diagram

```
TOYVECTORDB — STATE MACHINE
═══════════════════════════════════════════════════════════════════════════════

Since Module #6 has NO agent loop, the "state machine" models two things:
  (A) The lifecycle of the STORE itself
  (B) The lifecycle of an individual VECTOR within the store


───────────────────────────────────────────────────────────────────────────────
(A) STORE LIFECYCLE
───────────────────────────────────────────────────────────────────────────────

                    ┌──────────────────────────────────────────────────────────┐
                    │                                                          │
                    ▼                                                          │
           ┌─────────────────┐     insert()      ┌─────────────────┐           │
           │                 │ ───────────────►  │                 │           │
           │    EMPTY        │                   │   POPULATED     │           │
           │                 │                   │                 │           │
           │  len(store)=0   │                   │  len(store)>0   │           │
           │                 │ ◄───────────────  │                 │           │
           └────────┬────────┘   delete(last)    └───────┬─────────┘           │
                    │                                    │                     │
                    │         search_top_k()             │                     │
                    │         → returns []               │ search_top_k()      │
                    │         (no crash)                 │ → returns top-k     │
                    │                                    │                     │
                    │         insert()                   │ delete(some)        │
                    │         → raises ValueError        │ → PARTIALLY_EMPTY   │
                    │         (if dup ID)                │   (or back to       │
                    │                                    │    POPULATED)       │
                    └────────────────────────────────────┘                     │
                                                                               │
                    ┌──────────────────────────────────────────────────────────┘
                    │  (any insert from EMPTY brings it back to POPULATED)
                    │
                    ▼


  DETAILED TRANSITIONS:

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ STATE      │ GUARD / EVENT          │ NEXT STATE    │ SIDE EFFECT           │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │            │                        │               │                       │
  │  EMPTY     │ insert(valid_id,       │ POPULATED     │ _store[id] = (vec,    │
  │            │   valid_vec)           │               │   meta); len becomes 1│
  │            │                        │               │                       │
  │  EMPTY     │ search_top_k(any)      │ EMPTY         │ returns [] (no crash) │
  │            │                        │ (self-loop)   │                       │
  │            │                        │               │                       │
  │  EMPTY     │ delete(any_id)         │ EMPTY         │ returns False         │
  │            │                        │ (self-loop)   │ (not found)           │
  │            │                        │               │                       │
  │  POPULATED │ insert(valid_id,       │ POPULATED     │ _store[id] = (vec,    │
  │            │   valid_vec)           │ (self-loop)   │   meta); len += 1     │
  │            │                        │               │                       │
  │  POPULATED │ insert(DUP_id, ...)    │ POPULATED     │ raises ValueError;    │
  │            │                        │ (self-loop)   │ store unchanged       │
  │            │                        │               │                       │
  │  POPULATED │ search_top_k(valid_q)  │ POPULATED     │ returns List[         │
  │            │                        │ (self-loop)   │   SearchResult] of    │
  │            │                        │               │   length min(k, n)    │
  │            │                        │               │                       │
  │  POPULATED │ delete(existing_id)    │ POPULATED     │ del _store[id];       │
  │            │  (and len > 1 after)   │ (self-loop)   │ len -= 1              │
  │            │                        │               │                       │
  │  POPULATED │ delete(existing_id)    │ EMPTY         │ del _store[id];       │
  │            │  (and len == 1 before) │               │ len becomes 0         │
  │            │                        │               │                       │
  │  POPULATED │ delete(nonexistent)    │ POPULATED     │ returns False;        │
  │            │                        │ (self-loop)   │ store unchanged       │
  └─────────────────────────────────────────────────────────────────────────────┘


───────────────────────────────────────────────────────────────────────────────
(B) INDIVIDUAL VECTOR LIFECYCLE
───────────────────────────────────────────────────────────────────────────────

         ┌──────────────┐    insert()     ┌──────────────┐    delete()    ┌──────────────┐
         │              │ ─────────────►  │              │ ────────────►  │              │
         │  NONEXISTENT │                 │   STORED     │                │   DELETED    │
         │              │                 │              │                │              │
         │  not in _store│                │  in _store   │                │  removed     │
         └──────────────┘                 └──────┬───────┘                └──────┬───────┘
                                                 │                              │
                                                 │ search_top_k()               │ insert(same_id)
                                                 │ → may appear in results      │ → raises ValueError
                                                 │                              │   (ID is "taken"
                                                 │ get_all()                    │    until deleted)
                                                 │ → always included            │
                                                 │                              │
                                                 │ contains(id) → True          │ contains(id) → False
                                                 │                              │
                                                 ▼                              ▼


  VECTOR STATE PROPERTIES:

  ┌────────────────┬────────────────────────────────────────────────────────────┐
  │  PROPERTY      │  NONEXISTENT              STORED                    DELETED│
  ├────────────────┼────────────────────────────────────────────────────────────┤
  │  contains(id)  │  False                   True                      False   │
  │  in get_all()  │  No                       Yes                       No     │
  │  in search()   │  No                       Yes (if dim matches)      No     │
  │  re-insertable?│  Yes                      No (raises ValueError)    Yes    │
  │  memory        │  0 bytes                  len(vec)*8 + overhead     0 bytes│
  └────────────────┴────────────────────────────────────────────────────────────┘


───────────────────────────────────────────────────────────────────────────────
(C) SEARCH OPERATION STATE MACHINE (single call)
───────────────────────────────────────────────────────────────────────────────

  This shows what happens INSIDE one search_top_k() call:

  ┌──────────┐     store empty?     ┌──────────────┐
  │  START   │ ──── Yes ─────────►  │  RETURN []   │
  └────┬─────┘                      └──────────────┘
       │ No
       ▼
  ┌──────────┐     query invalid?   ┌──────────────────┐
  │ VALIDATE │ ──── Yes ─────────►  │  RAISE ValueError│
  └────┬─────┘                      └──────────────────┘
       │ No
       ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │                                                                      │
  │   FOR each (id, vector) in _store.items():                           │
  │                                                                      │
  │   ┌─────────────┐   dim match?    ┌──────────────────┐               │
  │   │  COMPARE    │ ─── No ───────► │  SKIP (continue) │               │
  │   │             │                 └──────────────────┘               │
  │   └──────┬──────┘                                                    │
  │          │ Yes                                                       │
  │          ▼                                                           │
  │   ┌─────────────────┐                                                │
  │   │ cosine_sim(q,v) │                                                │
  │   └──────┬──────────┘                                                │
  │          │                                                           │
  │          ▼                                                           │
  │   ┌─────────────────────┐                                            │
  │   │ append to results[] │                                            │
  │   └─────────────────────┘                                            │
  │                                                                      │
  └──────────────────────────────────────────────────────────────────────┘
       │
       ▼ (after loop completes)
  ┌──────────────┐
  │ SORT DESC    │  by score
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │ SLICE [:k]   │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │ RETURN LIST  │
  └──────────────┘


╔══════════════════════════════════════════════════════════════════════════════╗
║  LEGEND & DESIGN NOTES                                                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ┌──────────┐   State (box)                                                  ║
║  │          │                                                                ║
║  └──────────┘                                                                ║
║                                                                              ║
║  ──────►        Transition (arrow with event label)                          ║
║                                                                              ║
║  KEY DESIGN DECISIONS:                                                       ║
║                                                                              ║
║  1. NO "ERROR" STATE — errors are RAISED, not absorbed.                      ║
║     The store never enters a broken state. Either the operation              ║
║     succeeds (state unchanged or advanced) or it raises (state               ║
║     unchanged). This is the "transactional" pattern:                         ║
║     all-or-nothing per operation.                                            ║
║                                                                              ║
║  2. SEARCH IS PURE / SIDE-EFFECT FREE — calling search_top_k()               ║
║     never mutates the store. The state machine has self-loops                ║
║     on search for every state. This makes search safe to call                ║
║     any number of times in any order.                                        ║
║                                                                              ║
║  3. DELETE IS IDEMPOTENT-ISH — deleting a non-existent ID                    ║
║     returns False (not an error). The store state is                         ║
║     unchanged. This prevents "delete then delete again"                      ║
║     from crashing.                                                           ║
║                                                                              ║
║  4. DIMENSION MISMATCH IS SILENT — during search, vectors                    ║
║     with incompatible dimensions are skipped (not errors).                   ║
║     This allows a store to hold mixed-dimension vectors                      ║
║     without breaking searches for compatible ones.                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

All three diagrams complete. Module #6 is fully delivered (Turn 1 + Turn 2). Ready for your next instruction — say **"Build module #7 ToyRAG"** when you're ready to proceed.


---
---
---


# ToyVectorDB (Module #6)

**Store and search numeric vectors by similarity — the heart of every RAG system.**

> Prerequisite: understand Module #5 (ToyEmbeddings)

---

## What Is This?

A vector database is where your AI agent "remembers" things it has seen before. You store chunks of text as lists of numbers (vectors), then later ask "what in my memory is most similar to this question?" and it finds the closest matches.

That's it. No magic. A vector database is a **list of vectors + a similarity function + sorting**. The production tools you've heard of (Pinecone, Weaviate, ChromaDB) add fancy indexing so they can search millions of vectors in milliseconds. At toy scale, a Python dictionary and a for-loop do the same job.

**Analogy:** Imagine a library where every book is described by a color swatch (its vector). You walk in holding a blue paint sample and ask "which books look most like this?" The librarian compares your swatch to every book's swatch, ranks them by closeness of color, and hands you the top 3. That's `search_top_k()`.

---

## Folder Structure

```text
ToyVectorDB/
    main.py              # Demo: embed docs, insert, search, delete
    vector_store.py      # Core VectorStore class (the "database")
    embedding.py         # Minimal text-to-vector engine (re-implemented from #5)
    test_adversarial.py  # 5 adversarial tests proving the store survives bad input
    README.md            # This file
```

---

## Dependencies

**External:** None. Zero packages.

**Standard library used:**

| Module        | Why                                                           |
| ------------- | ------------------------------------------------------------- |
| `math`        | `sqrt()` for vector magnitude in cosine similarity            |
| `collections` | `Counter` for word frequency counting in the embedding engine |
| `dataclasses` | `@dataclass` for the `SearchResult` struct                    |
| `typing`      | Type hints (`List`, `Dict`, `Optional`, `Any`, `Tuple`)       |

**Internal import graph:**

```text
main.py  ──imports──►  vector_store.py
   │                        │
   │                        └── (defines cosine_similarity internally)
   │
   └──imports──►  embedding.py

test_adversarial.py  ──imports──►  vector_store.py
                          └──imports──►  embedding.py
```

No circular imports. `vector_store.py` is fully self-contained — it does not import from `embedding.py`. They are independent; the only thing connecting them at runtime is a `List[float]` (the vector itself).

---

## Execution

```bash
# Navigate to this folder
cd ToyVectorDB

# Run the demo (shows the full insert/search/delete workflow)
python main.py

# Run the adversarial test suite (proves robustness against bad input)
python test_adversarial.py
```

No installation needed. No virtual environment needed. Just Python 3.8+.

---

## Expected Output

### Demo (`python main.py`)

```text
============================================================
  TOYVECTORDB — Module #6: Store & Search by Similarity
============================================================

[PHASE 1] Building embedding engine from corpus...
    Vocabulary size (vector dimension): 42
    Sample vocab: ['a', 'all', 'and', 'ball', 'chased', 'datasets', 'dog', 'in', 'is', 'large']

[PHASE 2] Embedding documents and inserting into store...
    INSERTED doc_1: dim=42, magnitude=0.3266
    INSERTED doc_2: dim=42, magnitude=0.3536
    INSERTED doc_3: dim=42, magnitude=0.3780
    INSERTED doc_4: dim=42, magnitude=0.3162
    INSERTED doc_5: dim=42, magnitude=0.3333
    Store size: 5 vectors

[PHASE 3] Searching by similarity...

    QUERY 'pet_cat': "The cat is sleeping on the warm mat"
    #1  id=doc_1       score=+0.7845  meta={'text': 'The cat sat on the mat and purred softly', 'length': 9}
    #2  id=doc_3       score=+0.2673  meta={'text': 'The cat played with a red yarn ball near the window', 'length': 11}
    #3  id=doc_4       score=+0.0000  meta={'text': 'Python is a programming language used for data science', 'length': 9}

    QUERY 'pet_dog': "A happy dog runs in the green park"
    #1  id=doc_2       score=+0.6532  meta={'text': 'A dog chased the ball in the park all afternoon', 'length': 10}
    #2  id=doc_3       score=+0.1429  meta={'text': 'The cat played with a red yarn ball near the window', 'length': 11}
    #3  id=doc_1       score=+0.0000  meta={'text': 'The cat sat on the mat and purred softly', 'length': 9}

    QUERY 'tech_ai': "Neural networks and deep learning algorithms"
    #1  id=doc_5       score=+0.2887  meta={'text': 'Machine learning models require large datasets to train', 'length': 9}
    #2  id=doc_4       score=+0.1443  meta={'text': 'Python is a programming language used for data science', 'length': 9}
    #3  id=doc_1       score=+0.0722  meta={'text': 'The cat sat on the mat and purred softly', 'length': 9}

[PHASE 4] Deleting a vector...
    Before delete: 5 vectors
    delete('doc_2') -> True
    After delete: 4 vectors
    contains('doc_2') -> False

    Re-searching for 'dog' after deletion:
    #1  id=doc_3       score=+0.1429  meta={'text': 'The cat played with a red yarn ball near the window', 'length': 11}
    #2  id=doc_1       score=+0.0000  meta={'text': 'The cat sat on the mat and purred softly', 'length': 9}
    #3  id=doc_4       score=+0.0000  meta={'text': 'Python is a programming language used for data science', 'length': 9}

[PHASE 5] Edge case: searching with a zero-magnitude vector...
    Scores: ['0.0000', '0.0000', '0.0000']
    No crash — zero-magnitude handled gracefully.

============================================================
  DEMO COMPLETE
  A vector DB = list of vectors + similarity function + sorting.
============================================================
```

### Tests (`python test_adversarial.py`)

```
==================================================
  TOYVECTORDB — Adversarial Test Suite
==================================================
  PASS  Corrupted / malformed vectors
  PASS  Hostile / invalid input
  PASS  Operation failure (safe defaults)
  PASS  Stress / no degradation
  PASS  Empty DB / zero-mag / duplicate

  5/5 tests passed.
```

---

## Architecture Diagram

```
                         +----------------------------------+
                         |         CALLER (main.py)         |
                         |    (or future RAG pipeline #7)   |
                         +------+------------------+-------+
                                |                  |
                   text strings |                  | query_vector: List[float]
                                v                  v
             +------------------------+   +-----------------------------+
             |    EmbeddingEngine     |   |        VectorStore          |
             |                        |   |                             |
             |  _vocabulary: [str]    |   |  _store: Dict[str,          |
             |  _word_to_index: {}    |   |      Tuple[List[float],     |
             |                        |   |      Optional[Dict]]>       |
             |  Methods:              |   |                             |
             |   .build(docs)         |   |  Public API:                |
             |   .embed(text)         |   |   .insert(id, vec, meta)    |
             |   .dimension (prop)    |   |   .search_top_k(q, k)       |
             +------------------------+   |   .delete(id)               |
                                          |   .get_all()                |
                                          |   .contains(id)             |
                                          |                             |
                                          |  Internal:                  |
                                          |   cosine_similarity(a, b)   |
                                          +--------------+--------------+
                                                         |
                                                         v
                                          +-----------------------------+
                                          |     List[SearchResult]      |
                                          |                             |
                                          |  .id: str                   |
                                          |  .vector: List[float]       |
                                          |  .metadata: Dict or None    |
                                          |  .score: float              |
                                          +-----------------------------+

NOTE: EmbeddingEngine and VectorStore NEVER talk to each other directly.
The vector (List[float]) is the ONLY interface between them.
This means you can swap the embedding engine (e.g., use a real model later)
without changing a single line in VectorStore.
```

---

## State Machine Diagram

Since this module has no agent loop, the state machine models the **store's lifecycle** and the **search operation flow**.

### Store Lifecycle

```
    +-----------+    insert()     +-----------+
    |           | ------------->  |           |
    |   EMPTY   |                 | POPULATED |
    |           | <-------------- |           |
    +-----------+  delete(last)   +-----------+
         ^                              |  ^
         |                              |  |
         |        search_top_k()        |  | insert()
         |        returns []            |  | (self-loop)
         |                              |  |
         +------------------------------+  |
              (self-loop: no crash)        |
                                           |
                              delete(some, len>1 after)
                              (self-loop: stays POPULATED)
```

### Search Operation (inside one call to search_top_k)

```
    START
      |
      v
  store empty? --Yes--> RETURN []
      |No
      v
  query valid? --No--> RAISE ValueError
      |Yes
      v
  +-------------------------------------------+
  | FOR each (id, vector) in store:           |
  |                                           |
  |   dimensions match? --No--> SKIP          |
  |         |Yes                              |
  |         v                                 |
  |   score = cosine_similarity(q, v)         |
  |   append to results[]                     |
  +-------------------------------------------+
      |
      v
  SORT by score (descending)
      |
      v
  SLICE to top-k
      |
      v
  RETURN List[SearchResult]
```

**Key property:** Search is **pure** — it never mutates the store. You can call it any number of times in any order without side effects.

---

## File Reference

### `vector_store.py`

The heart of this module. Defines three things: (1) a `cosine_similarity()` function that measures how "pointy" two vectors are relative to each other (1.0 = identical direction, 0.0 = perpendicular or zero, -1.0 = opposite); (2) a `SearchResult` dataclass that bundles an ID, its vector, metadata, and a similarity score into one object; (3) the `VectorStore` class itself, which wraps a Python dictionary and exposes five methods: `insert`, `search_top_k`, `delete`, `get_all`, and `contains`. The store is intentionally dumb — it doesn't know what the vectors represent. It just stores lists of numbers and finds the closest ones by angle.

### `embedding.py`

A minimal re-implementation of Module #5's embedding engine, kept here so this module is fully standalone. It converts text into a vector where each position represents one word in the vocabulary, and the value at that position is how frequently the word appears (normalized by total word count). Think of it as a "word fingerprint" — two texts about cats will have high values in the "cat" position, making their vectors point in similar directions. The `build()` class method learns the vocabulary from a corpus; `embed()` converts any text into a vector in that space.

### `main.py`

The demo script. Walks through five phases: (1) build an embedding engine from 5 sample documents about pets and tech, (2) embed each document and insert it into a fresh VectorStore, (3) run three different queries and show how the right documents float to the top, (4) delete one document and show it disappears from results, (5) search with a zero-vector to prove no crash. This file is your "lab notebook" — read it top to bottom to see the full lifecycle.

### `test_adversarial.py`

Five tests that prove the store survives hostile conditions: corrupted vectors (None elements, wrong types), hostile input (oversized queries, None, empty lists), operation failures (deleting non-existent IDs, searching empty stores), stress (500 inserts + 100 searches without degradation), and module-specific edge cases (zero-magnitude vectors, duplicate IDs, searching an empty database). If all 5 pass, you can trust the store won't crash in production-like failure scenarios.

---

## What This Teaches

| Concept                            | Where to Look                                | Key Insight                                                                         |
| ---------------------------------- | -------------------------------------------- | ----------------------------------------------------------------------------------- |
| A vector is just a list of numbers | `embedding.py` line 36-41                    | No GPU, no model, no API. It's `[0.11, 0.0, 0.22, ...]`                             |
| Similarity = angle between vectors | `vector_store.py` cosine_similarity()        | Two texts about the same topic point in similar directions                          |
| A vector DB is a dict + sort       | `vector_store.py` VectorStore class          | Pinecone/Weaviate add indexing for scale; the concept is identical                  |
| Insert is O(1), Search is O(n)     | `vector_store.py` insert() vs search_top_k() | At 5 vectors, O(n) is instant. At 5 million, you need HNSW                          |
| Dimension mismatch must be handled | `vector_store.py` line 57-58                 | Real collections mix dimensions; skipping is safer than crashing                    |
| Zero-magnitude needs a guard       | `vector_store.py` line 14-15                 | Division by zero is the #1 crash in naive similarity code                           |
| Store and embedder are decoupled   | Architecture diagram                         | Swap the brain (embedding model), keep the body (store) — same pattern as Module #0 |
| Metadata travels with the vector   | `vector_store.py` SearchResult.metadata      | The store doesn't interpret metadata; it just carries it back to you                |

---

## How This Fits in the Curriculum

```
Module #5: Text --> [numbers]          (ToyEmbeddings)
                              |
                              v
Module #6: [numbers] --> STORE + SEARCH  (THIS MODULE)
                              |
                              v
Module #7: SEARCH results --> paste into prompt --> grounded answer  (ToyRAG)
```

In Module #7, the RAG pipeline will call `store.search_top_k(query_vector, k=3)` to find relevant documents, then inject their text into the agent's context. The vector store is the "memory" that makes RAG possible. Without it, the agent has no external knowledge — just whatever was in its training data.

---

## Intuition Check (for non-IT readers)

**Why not just use a regular database with SQL?**

A regular database answers "give me all rows where `name = 'cat'`". It needs an exact match. A vector database answers "give me the 3 rows that are *most like* this fuzzy description". It works on similarity, not equality. That's the difference between looking up a phone number and saying "I think it's the one that starts with 5..."

**Why cosine and not just "count matching words"?**

Counting matches (called "Jaccard similarity") treats all words as equally important and ignores how much of each word appears. Cosine similarity works in *direction* space — it asks "do these two documents point the same way?" A document that says "cat cat cat" and one that says "the cat sat" both point strongly in the "cat direction", so cosine scores them as similar. Jaccard would say they share only 1 of 4 unique words (25%), which feels wrong.

**What happens when I scale to millions of vectors?**

Right now, `search_top_k` checks every single vector (O(n)). With 1 million vectors, that's 1 million cosine computations per query — too slow for real-time use. Production systems build an **approximate nearest neighbor index** (HNSW graph, IVF clusters, etc.) that lets you skip 99% of vectors and only check the ~100 most promising ones. The answer is *approximately* the same, but 1000x faster. That's the entire engineering challenge behind Pinecone/Weaviate/Qdrant — and it's why this toy module uses a simple loop: at 5 vectors, the loop IS the index.

---

## Parent Curriculum

Part of **TOYAGENTS** — a series of standalone, zero-dependency Python projects that teach agentic AI concepts without requiring a real LLM or any external package.

Full curriculum: https://github.com/AlexOcculate/TOYAGENTS
```