# Financial Document RAG Pipeline

A Retrieval-Augmented Generation (RAG) pipeline for querying
financial documents using LangChain, ChromaDB, and Groq.

Built on the RBI Annual Report 2024-25 (318 pages, 1,194 chunks at chunk_size=1000).as the source document.

## Why I Built This

Financial documents are dense, long, and hard to query manually.
This pipeline chunks, embeds, and retrieves relevant sections to
answer specific questions about financial data — the kind of
problem every fintech AI team is solving at scale.

## Tech Stack

- **LangChain** — document loading and text splitting
- **PyPDF** — PDF ingestion
- **ChromaDB** — vector store for semantic search
- **HuggingFace sentence-transformers** — embeddings
- **Groq** — LLM for answer generation (llama-3.3-70-versatile)
- **Python 3.10+**

## Project Structure

```
financial-rag/
├── embed_store.py   # builds ChromaDB once from PDF
├── rag.py           # queries ChromaDB and generates answers
├── .env             # GROQ_API_KEY (not committed)
└── README.md
```

## How to Run

```bash
git clone https://github.com/rojitharepalle/financial-rag
cd financial-rag
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install langchain langchain-community langchain-text-splitters \
  langchain-huggingface langchain-chroma chromadb \
  sentence-transformers groq pypdf python-dotenv
```

Add your Groq API key:

```bash
echo "GROQ_API_KEY=your_key_here" > .env
```

Build the vector store once:

```bash
python embed_store.py
```

Query the document:

```bash
python rag.py
```

## Chunking Parameters

```python
chunk_size=1000      # characters per chunk
chunk_overlap=200    # overlap between consecutive chunks
```

Chose 1000 over 500 because RBI report paragraphs are densewith financial terminology — smaller chunks lose context.Will benchmark retrieval quality at both sizes in Day 4.

## What I Learned — Day 1

- RBI 2024-25 report loads into **318 pages** and produces **1194 chunks** at chunk_size=1000, overlap=200
- `RecursiveCharacterTextSplitter` splits on paragraph boundaries first before falling back to character splits — this preserves more semantic meaning than fixed-size splits
- Chunk overlap of 200 characters ensures context is not lost at boundaries between chunks

## What I Learned — Day 2:

- ChromaDB stores 1,194 chunks and retrieval works
- Result 1 failure: PDF chart text extracted as garbled axis labels, polluting the vector store with noise
- Result 2 success: Relevant inflation projection content retrieved correctly from page 37
- Fix planned: filter or flag chunks where meaningful word ratio is below a threshold before storing

- Initial retrieval returned chart axis labels as top results
- Root cause: PyPDF extracts graph tick marks as plain text, indistinguishable from paragraphs at chunk level
- Fix: digit ratio filter — chunks where >30% of characters are digits or symbols are dropped before storage
- Result: chart noise eliminated, all top 3 results now return genuine financial prose
- Chunks after filtering: 1028 of 1,194 (166 removed as noise)

## What I Learned — Day 3

- Connected ChromaDB retrieval to Groq LLM (llama-3.3-70b-versatile) for end-to-end question answering
- Tested 3 questions against the RBI Annual Report

| Question                  | Result                                                             |
| ------------------------- | ------------------------------------------------------------------ |
| RBI's stance on inflation | Weak — chunks retrieved from bibliography, not policy sections     |
| GDP growth forecast       | Strong — returned 6.4% for 2024-25 and 6.7% for 2025-26 accurately |
| Interest rate measures    | Partial — retrieved table data but missed narrative policy text    |

- Pattern: larger chunks improve narrative policy retrieval but the noise filter removed chunks containing numerical forecasts, breaking GDP retrieval

## What I Learned — Day 4

### Chunk Size Benchmarking — 500 vs 1000

| Question                  | chunk_size=500                       | chunk_size=1000                              |
| ------------------------- | ------------------------------------ | -------------------------------------------- |
| RBI's stance on inflation | Weak — bibliography chunks retrieved | Weak — inferred from research topics         |
| GDP growth forecast       | Strong — 6.4% and 6.7% returned      | Weak — IMF forecast returned instead         |
| Interest rate measures    | Partial — table data only            | Strong — specific measures with basis points |

### Noise Filter Iterations

- Initial filter removed 166 chunks — but was also removing
  legitimate numerical policy chunks containing GDP forecasts
- Refined filter with 3 conditions:
  - Month sequence pattern (Jan-21, Feb-21 etc.) → filter
  - Lines with standalone numbers ratio > 30% → filter
  - High digit ratio AND short average word length → filter
- Final result: 1,028 clean chunks from 1,194 total

### Critical Finding — Silent Hallucination

- When retrieved chunks don't contain the answer, `llama-3.3-70b-versatile` pulls numbers from training data instead of saying "not found"
- Example: Asked for India GDP forecast, returned 7.1% confidently — number not present in any retrieved chunk
- This is the most dangerous failure mode in production RAG — the system sounds confident while being wrong
- Fix: add a confidence check or force the LLM to cite the specific chunk it used

### Retrieval Failure — Deep Document Chunks

- Page 114 contains the exact GDP forecast (6.4% for 2024-25, 6.7% for 2025-26) but is never retrieved regardless of question wording
- Root cause: `all-MiniLM-L6-v2` embedding model does not capture semantic similarity between forecast questions and forecast prose buried deep in the document
- Fix planned for Day 5: add reranking using a cross-encoder model to reorder top-k results by true relevance

  ## Roadmap

- [x] PDF ingestion and chunking
- [x] Embedding and vector store with ChromaDB
- [x] End-to-end Q&A with Groq LLM
- [x] Noise filter for chart extraction artifacts
- [x] Chunk size benchmarking — 500 vs 1000
- [ ] Reranking with cross-encoder model
- [ ] Hallucination detection — force LLM to cite source chunk
- [ ] Evaluation scoring across 10 standard questions
