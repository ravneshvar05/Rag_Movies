# Movie Transcript RAG System

Welcome to the **Movie Transcript RAG (Retrieval-Augmented Generation) System**. This project is designed to accurately answer user questions about a movie based on its transcript (SRT files), while returning specific timestamps as citations for its answers.

This document serves as the primary technical documentation for the system pipeline, the chunking strategies evaluated, and the embedding models implemented.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- API Keys for LLM providers (e.g., Groq API for Llama 3 models) defined in your environment.

### Installation & Startup
1. **Clone the repository and navigate to the project:**
   ```bash
   cd movie_rag
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure Environment:**
   Create a `.env` file and add your required API keys (e.g., `GROQ_API_KEY`). Ensure `config/config.yaml` is set up with your desired model parameters.
4. **Ingest a Movie Transcript:**
   Before querying, you must ingest an SRT file into the vector database.
   ```bash
   python src/main.py ingest path/to/transcript.srt "movie_title_id"
   ```
5. **Run the Interactive QA System:**
   Start the interactive chat to ask questions about the ingested transcripts.
   ```bash
   python src/main.py interactive
   ```
   Or ask a single query:
   ```bash
   python src/main.py query "What happens after the bank robbery?"
   ```

---

## 🏗️ System Architecture & Pipeline Steps

The system operates across a sophisticated multi-stage pipeline designed to parse, index, search, and reason over movie subtitle files.

### 1. Data Ingestion & Parsing (`SRTParser`)
The system takes raw movie transcript `.srt` files and extracts the text alongside exact start and end timestamps. This allows the system to ground every piece of dialogue or text to a specific moment in the video.

### 2. Time-Aware Chunking (`TranscriptChunker`)
Instead of just slicing text arbitrarily, the parsed transcripts are passed through a chunker that relies on time constraints (e.g., min 10s, max 120s, 5s overlap) and scene-detection heuristics (detecting long silences between dialogue). 

### 3. Vectorization & Storage (`EmbeddingStore` & `MetadataStore`)
- **Metadata Store (SQLite):** Stores the raw text, timestamps, and chunk IDs to maintain structural data context.
- **Embedding Store (FAISS):** The textual content of every chunk is passed through an embedding model to create dense vector representations. These vectors are then stored in a local FAISS index for high-speed semantic similarity searches.

### 4. Hybrid Retrieval (`HybridRetriever`)
When a user asks a question, the system queries the databases using two parallel strategies:
- **Semantic Search (FAISS):** Finds chunks conceptually related to the user's question.
- **Keyword Search:** Uses sparse/exact keyword matching to find mentions of specific character names or unique objects.
- **Reciprocal Rank Fusion (RRF):** The results of both searches are merged and re-ranked. Currently, the system merges up to **35 top chunks** to guarantee high recall context retrieval.

### 5. Multi-Agent LLM Reasoning Pipeline
The RAG reasoning phase is handled by Groq-hosted Llama 3 models. The generation is deliberately broken down into steps:

- **Step 1: `QuestionRouter`** 
  *(Note: This step is currently bypassed in code to optimize API costs, defaulting all queries to standard retrieval without needing an initial LLM routing pass).*
- **Step 2: `RelevanceJudge`** (Powered by `llama-3.1-8b-instant`)
  Scrutinizes the retrieved transcript chunks and filters out any that aren't genuinely relevant to the question.
- **Step 3: `ContextDistiller`** (Powered by `llama-3.1-8b-instant`)
  Compresses the selected chunks to remove noise while preserving core facts, dialogue, and timestamps.
- **Step 4: `Answerer`** (Powered by `llama-3.3-70b-versatile` / `llama-3.1-8b-instant` fallback)
  Generates the final, conversational answer based *only* on the distilled facts, returning precise SRT timestamps as citations.

> **⚙️ Toggle LLM Filter Layers for Speed:** 
> The `RelevanceJudge` and `ContextDistiller` act as two consecutive LLM filtering layers before sending the final context to the Answerer. While they ensure high accuracy, they consume API tokens and increase latency. 
> **You can easily toggle these off to drastically speed up processing.** In your `config/config.yaml` file, simply set the pipeline flags to `false`:
> ```yaml
> pipeline:
>   enable_judge: false      
>   enable_distiller: false 
> ```
> *(Note: The system currently ships with these disabled by default for faster inference).*

---

## 🧩 Chunking Strategy Analysis

Breaking down a movie script for an AI brain is fundamentally different from chunking a standard text document (like a PDF or article). We evaluated three standard chunking strategies before settling on our current implementation.

### Current Approach: Time-Stamp / Scene-Aware Chunking 🏆
Our system chunks based on **fixed time intervals** (e.g., 2-minute blocks) and **scene breaks** (detecting long gaps in subtitles). 
**Why it works:** In video, time is the ultimate anchor. A user might ask, *"What happens exactly 10 minutes after John enters the building?"* By chunking dialogue logically by its timestamp constraints, we ensure that the temporal flow of events is never broken, and all facts remain grounded to their timeframes.

### Alternative 1: Recursive Character Chunking
This strategy splits text hierarchically based on character counts and structural separators (like double newlines, single newlines, and spaces).
- **Why it is unreliable for movie transcripts:** Recursive chunking ignores time. Movie subtitles are highly irregular—a 5-second action scene might have heavy dialogue, while a 3-minute suspense sequence might have a single word. Recursive chunking might combine 10 minutes of sparse dialogue into one chunk, or split a fast 5-second conversation across three chunks. This completely destroys the temporal accuracy required for a movie assistant.

### Alternative 2: Semantic Chunking
This strategy dynamically groups text segments until the semantic meaning or "topic" changes, using embeddings to detect shifts in conversation.
- **Why it is unreliable for movie transcripts:** Movie dialogues frequently shift topics erratically, feature rapid interruptions, or bounce between subplots. A purely semantic chunker will separate a single continuous scene into scattered chunks just because the characters changed the subject. This removes the chronological context (what happened *when*), which is fatal for movie story comprehension.

---

## 🧠 Embedding Model Selection

To convert text chunks into vector representations, we evaluated several industry-standard embedding models.

### Chosen Model: `BAAI/bge-base-en-v1.5` 🏆
We utilize the `bge-base-en-v1.5` model (768 dimensions), created by BAAI. 
**Why we chose it:**
1. **Local & Open-Source:** It runs entirely locally on CPU/GPU, meaning zero API costs, no rate limits, and full data privacy. 
2. **State-of-the-Art Performance:** Across the MTEB (Massive Text Embedding Benchmark), BGE models consistently demonstrate best-in-class retrieval accuracy for their weight class.
3. **Efficiency:** The `base` version offers an optimal balance between low memory footprint/fast inference latency and high-quality semantic understanding, making it perfect for rapidly generating vectors across long movie files.

### Considered Alternatives

1. **`BAAI/bge-large-en-v1.5`**
   - *Pros:* Offers even higher accuracy and dimensional depth (1024 dimensions) than the base model.
   - *Cons:* The model weights are significantly larger, making it very hard to load on standard hardware. It requires a substantial RAM and VRAM footprint, which drastically slows down standard local deployments and extraction times. We opted for the `base` version to ensure the system remains snappy and accessible.
2. **SentenceTransformers `all-MiniLM-L6-v2`**
   - *Pros:* Blazing fast, extremely lightweight (384 dimensions), and easily runs on HuggingFace Space free tiers without OOM errors.
   - *Cons:* Lower retrieval accuracy on nuanced or complex domain queries compared to BGE. Given that movie plots often contain subtle narrative themes, MiniLM drifted in reasoning quality for complex "WHY" or "HOW" questions.
