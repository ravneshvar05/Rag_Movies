# System Deep Dive: Movie Transcript RAG

This document details exactly how the system processes data, from raw SRT uploads to the final answer, including specific parameters and internal syntax.

---

## 1. Ingestion & Chunking Process

When a user uploads an SRT file, it goes through **Header Parsing**, **Cleaning**, and **Scene-Aware Chunking**.

### A. Parsing & Cleaning (`srt_parser.py`)
*   **Syntax**: Parses standard SRT format (Index -> Time Range -> Text).
*   **Cleaning**: Removes HTML tags (`<i>`, `<b>`), identifying music/sound effects (brackets `[]`, parentheses `()`).
*   **Grouping**: Merges lines that are close in time (< 2.0s gap) into single "entries" to maintain dialogue flow.

### B. Scene-Aware Chunking (`chunker.py`)
Instead of fixed-size windows (like 500 characters), we use **time-based scene detection**.

**Algorithm (`_scene_aware_chunking`):**
1.  Iterate through all subtitles.
2.  Calculate **gap** between current subtitle end and next subtitle start.
3.  **CUT** if:
    *   `Gap > 3.0 seconds` (Parameter: `scene_break_threshold`): Implies a scene change or long pause.
    *   `Duration > 120 seconds` (Parameter: `max_chunk_seconds`): Enforces max length.
4.  **KEEP** if:
    *   `Duration < 10 seconds` (Parameter: `min_chunk_seconds`): Too short, merge with next.
5.  **Overlap**: Adds `5 seconds` (Parameter: `overlap_seconds`) of audio/text from previous chunk to maintain context at boundaries.

**Result**: A list of `TranscriptChunk` objects with `start_time`, `end_time`, `text`, and `scene_id`.

---

## 2. Retrieval Process (When You Ask a Question)

When a question is asked (e.g., *"Why did Rocky lose?"*), a **Hybrid Retrieval** strategy executes.

### Step 1: Semantic Search (Vector)
*   **Model**: `BAAI/bge-base-en-v1.5`
*   **Dimension**: `768` floats.
*   **Syntax**:
    ```python
    query_vector = model.encode(query, normalize_embeddings=True)
    D, I = index.search(query_vector, k=40)  # Top 40 results
    ```
*   **Index**: `faiss.IndexFlatIP` (Inner Product, equivalent to Cosine Similarity since vectors are normalized).

### Step 2: Keyword Search (Lexical)
*   **Engine**: SQLite `FTS5` (Full-Text Search module).
*   **Query Processing**:
    1.  Lowercases query.
    2.  Removes stop words (`the`, `is`, `at`, etc.).
    3.  **Fuzzy Matching**: Appends `*` (prefix match). Example: `Rocky*` matches `Rocky`, `Rockys`, `Rockies`.
    4.  **Syntax**:
        ```sql
        SELECT * FROM chunks_fts WHERE text MATCH '"Rocky"* OR "lose"*' LIMIT 30
        ```
*   **Filtering**: `score >= 0.5`.

### Step 3: Hybrid Fusion (RRF)
We merge results using **Reciprocal Rank Fusion (RRF)** to balance vector and keyword scores.

**Formula**:
$$ Score(d) = \sum \frac{1}{k + rank(d)} $$
*   **Parameter**: `k=60` (default constant).
*   **Logic**:
    *   If a chunk is #1 in Semantic and #1 in Keyword -> High score.
    *   If a chunk is #1 in Semantic but missing in Keyword -> Moderate score.
    *   This "smooths" the rankings so one bad method doesn't dominate.

**Final Output**: Top `25` chunks (Parameter: `max_final_chunks`).

---

## 3. LLM Pipeline (The "Brain")

The retrieved chunks trigger a **4-step LLM chain** using Groq (`llama-3.1-8b-instant` for speed, `llama-3.3-70b` for final answer).

### Step A: Router (`QuestionRouter`)
*   **Role**: Classifies the question type to guide the answers.
*   **Input**: Question string.
*   **Output JSON**:
    ```json
    {"category": "WHY", "requires_full_narrative": true}
    ```
*   **Categories**: `WHY`, `WHAT`, `HOW`, `QUOTE`, `SUMMARY`, `GLOSSARY`.

### Step B: Relevance Judge (`RelevanceJudge`)
*   **Role**: Filters out "noise" chunks that were retrieved but aren't actually relevant.
*   **Input**: Question + List of 25 Chunks.
*   **System Prompt**: *"Identify which chunks are relevant to the user's question."*
*   **Output JSON**:
    ```json
    {"relevant_chunk_ids": [5, 6, 7, 12]}
    ```
*   **Effect**: Reduces context from ~5000 tokens to ~1000 tokens of pure signal.

### Step C: Context Distiller (`ContextDistiller`)
*   **Role**: Compresses/Rewrites the relevant chunks into a coherent narrative.
*   **Input**: Relevant Chunks.
*   **System Prompt**: *"Compress the transcripts while preserving key facts, dialogue, and timestamps."*
*   **Compression Ratio**: Target ~0.6x of original size.
*   **Output**: A clean string: `[00:10:30] Rocky fights Apollo. [00:10:45] He gets knocked down.`

### Step D: Answerer (`Answerer`)
*   **Role**: Generates the final user-facing answer.
*   **Model**: **`llama-3.3-70b-versatile`** (Smarter, larger model).
*   **Input**: Distilled Context + Original Question.
*   **System Prompt**: 
    1.  Answer ONLY from context.
    2.  Cite timestamps (`[MM:SS]`) for every claim.
    3.  If unknown, say "I cannot find this information."
*   **Final Output**:
    > "Rocky loses the fight because of a split decision [01:58:20]. However, he wins the moral victory by going the distance [01:59:10]."

---

## 4. detailed Code Execution Flow

This section traces the exact Python method calls for each user action.

### Flow A: User Uploads File (Ingestion)

1.  **Streamlit Interface** (`app.py`)
    *   User drags & drops file -> `st.file_uploader`.
    *   Calls `ingest_pipeline.process_movie()`.

2.  **Parsing** (`srt_parser.py`)
    *   Method: `SRTParser.parse_file(path)`
    *   Action: Reads raw text line-by-line.
    *   Param: `group_threshold=2.0` (Merges subs <2s apart).

3.  **Chunking** (`chunker.py`)
    *   Method: `TranscriptChunker.chunk_entries(entries, strategy='scene_aware')`
    *   Logic:
        ```python
        gap = next_start - current_end
        if gap > 3.0:  # scene_break_threshold
            create_new_chunk()
        elif duration > 120:  # max_chunk_seconds
            create_new_chunk()
        ```

4.  **Embedding & Storage** (`embedding_store.py`)
    *   Method: `EmbeddingStore.add_chunks(chunks)`
    *   **Vector**:
        ```python
        vectors = model.encode([c.text for c in chunks])
        index.add(vectors)  # FAISS
        ```
    *   **Metadata**:
        *   Method: `MetadataStore.add_movie(movie_id, title)`
        *   Method: `MetadataStore.add_chunks(chunk_dicts)` -> SQL INSERT.

---

### Flow B: User Asks Question (Pipeline)

1.  **Start** (`app.py`)
    *   User types "Why did he die?" -> `pipeline.answer_question(q)`.

2.  **Routing** (`rag_pipeline.py`)
    *   Call: `router.route(question)`
    *   Model: `llama-3.1-8b`
    *   Prompt: "Classify this question..."

3.  **Retrieval** (`rag_pipeline.py`)
    *   Call: `retriever.retrieve(question, top_k=25)`
    *   **Internal**:
        1.  `embedding_store.search(q, k=40)` -> Semantic IDs
        2.  `keyword_search.search(q, k=30)` -> Keyword IDs
        3.  **RRF Merge**: `score = 1/(60 + rank_sem) + 1/(60 + rank_key)`
    *   Result: List of `TranscriptChunk` objects.

4.  **Relevance Judgment** (`rag_pipeline.py`)
    *   Call: `judge.judge(question, chunks)`
    *   Model: `llama-3.1-8b`
    *   Logic:
        ```python
        if len(chunks) > 20:
            judge_in_batches(batch_size=20)
        ```
    *   Result: `[ID_1, ID_5, ID_9]` (Subset of chunks).

5.  **Distillation** (`rag_pipeline.py`)
    *   Call: `distiller.distill(question, relevant_chunks)`
    *   Model: `llama-3.1-8b`
    *   Action: Rewrites chunks into a single narrative flow.

6.  **Final Answer** (`rag_pipeline.py`)
    *   Call: `answerer.answer(question, distilled_context)`
    *   Model: **`llama-3.3-70b`**
    *   Output: Final text with timestamps.

---

## 5. Concrete Data Flow Example

Here is exactly what happens to the data, step-by-step.

### A. Raw Input (SRT File)
Imagine this file: `rocky.srt`

```text
1
00:10:05,000 --> 00:10:07,500
[Music playing]
Get up, Rocky!

2
00:10:08,000 --> 00:10:10,000
<i>You can do it!</i>

3
00:10:10,500 --> 00:10:13,000
I ain't hear no bell.

4
00:10:18,000 --> 00:10:20,000
One more round.
```

### B. Preprocessing & Cleaning
**Logic**: `SRTParser` reads this.
1.  **Entry 1**: Removes `[Music playing]`. Keeps "Get up, Rocky!"
2.  **Entry 2**: Removes `<i>...</i>`. Keeps "You can do it!"
3.  **Merging**:
    *   Gap between #1 (end 10:07.5) and #2 (start 10:08.0) is **0.5s**.
    *   Gap between #2 (end 10:10.0) and #3 (start 10:10.5) is **0.5s**.
    *   **THRESHOLD** is 2.0s. Since gaps < 2.0s, these merge into one "dialogue flow".

**Resulting Cleaned Entries**:
*   **Entry A**: `00:10:05 -> 00:10:13`: "Get up, Rocky! You can do it! I ain't hear no bell."
*   **Entry B**: `00:10:18 -> 00:10:20`: "One more round." (Because gap from 10:13 to 10:18 is 5s > 2s).

### C. Chunking Strategy
**Logic**: `TranscriptChunker` processes Entries A and B.

1.  **Processing Entry A**: Starts new chunk. Current duration = 8s.
2.  **Gap Analysis**: Gap between Entry A (end 10:13) and Entry B (start 10:18) is **5.0 seconds**.
3.  **Decision**: `Gap (5.0s) > scene_break_threshold (3.0s)`.
4.  **ACTION**: ✂️ **CUT CHUNK**.

**Resulting Chunk**:
```json
{
  "chunk_id": "rocky_chunk_1",
  "start_time": "00:10:05",
  "end_time": "00:10:13",
  "text": "Get up, Rocky! You can do it! I ain't hear no bell.",
  "scene_id": 1
}
```

### D. Final Storage

#### 1. Vector DB (FAISS)
The text *"Get up, Rocky!..."* is sent to the embedding model (`BAAI/bge-base-en-v1.5`).

*   **Input**: "Get up, Rocky! You can do it!..."
*   **Output**: A list of **768 numbers** (e.g., `[0.023, -0.154, 0.882, ...]`).
*   **Stored In**: `faiss.index` file (optimized binary format).
*   **ID Mapping**: The index stores vector #0. We keep a list: `chunk_ids = ["rocky_chunk_1"]`.

#### 2. SQLite Database (Metadata)
We store the human-readable data here.

**Table: `chunks`**
| chunk_id | movie_id | start_time | end_time | text |
| :--- | :--- | :--- | :--- | :--- |
| `rocky_chunk_1` | `rocky_1976` | `605.0` | `613.0` | "Get up, Rocky!..." |

**Table: `chunks_fts` (Full Text Search)**
| text |
| :--- |
| "get up rocky you can do it i ain't hear no bell" |
*(Optimized for keyword search like "bell" or "rocky")*

---

## Summary of Parameters

| Component | Parameter | Value | Description |
| :--- | :--- | :--- | :--- |
| **Chunking** | `max_chunk_seconds` | `120` | Max duration of a single chunk |
| | `scene_break_threshold` | `3.0` | Gap in seconds = new chunk |
| **Embedding** | `model` | `BAAI/bge-base-en-v1.5` | 768-dim vector model |
| **Retrieval** | `semantic_top_k` | `40` | Vectors retrieved initially |
| | `keyword_top_k` | `30` | Keyword matches retrieved initially |
| | `merge_strategy` | `rrf` | Reciprocal Rank Fusion (k=60) |
| **LLM** | `router/judge` | `llama-3.1-8b` | Fast model for logic |
| | `answerer` | `llama-3.3-70b` | Smart model for writing |
