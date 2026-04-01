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
- **Groq** — LLM for answer generation (in progress)
- **Python 3.10+**

## Project Status

| Step                       | Status   |
| -------------------------- | -------- |
| PDF ingestion and chunking | ✅ Done  |
| Embedding and vector store | 🔄 Day 2 |
| End-to-end Q&A with Groq   | 🔄 Day 3 |
| Failure analysis and fixes | 🔄 Day 4 |

## What I Learned — Day 1

- RBI 2024-25 report loads into **318 pages** and produces
  **1194 chunks** at chunk_size=1000, overlap=200
- `RecursiveCharacterTextSplitter` splits on paragraph
  boundaries first before falling back to character splits —
  this preserves more semantic meaning than fixed-size splits
- Chunk overlap of 200 characters ensures context is not
  lost at boundaries between chunks

## Chunking Parameters

```python
chunk_size=1000      # characters per chunk
chunk_overlap=200    # overlap between consecutive chunks
```

Chose 1000 over 500 because RBI report paragraphs are dense
with financial terminology — smaller chunks lose context.
Will benchmark retrieval quality at both sizes in Day 4.

## How to Run

```bash
git clone https://github.com/rojitharepalle/Financial-RAG
cd Financial-RAG
python -m venv venv
source venv/bin/activate
pip install langchain langchain-community chromadb \
  sentence-transformers groq pypdf python-dotenv
```

Add your Groq API key:

```bash
echo "GROQ_API_KEY=your_key_here" > .env
```

Run:

```bash
python load_chunks.py
```

## Roadmap

- [ ] Add ChromaDB vector store
- [ ] Add semantic search over chunks
- [ ] Connect Groq LLM for Q&A
- [ ] Evaluate answer quality on 10 financial questions
- [ ] Add reranking to improve retrieval precision

## What I Learned — Day 2:

- ChromaDB stores 1,194 chunks and retrieval works
- Result 1 failure: PDF chart text extracted as garbled axis labels, polluting the vector store with noise
- Result 2 success: Relevant inflation projection content retrieved correctly from page 37
- Fix planned: filter or flag chunks where meaningful word ratio is below a threshold before storing

- Initial retrieval returned chart axis labels as top results
- Root cause: PyPDF extracts graph tick marks as plain text, indistinguishable from paragraphs at chunk level
- Fix: digit ratio filter — chunks where >30% of characters are digits or symbols are dropped before storage
- Result: chart noise eliminated, all top 3 results now return genuine financial prose
- Chunks after filtering: 1021 of 1,194 (173 removed as noise)

## What I Learned — Day 3

- Connected ChromaDB retrieval to Groq LLM (llama-3.3-70b-versatile)
  for end-to-end question answering
- Tested 3 questions against the RBI Annual Report

| Question                  | Result                                                             |
| ------------------------- | ------------------------------------------------------------------ |
| RBI's stance on inflation | Weak — chunks retrieved from bibliography, not policy sections     |
| GDP growth forecast       | Strong — returned 6.4% for 2024-25 and 6.7% for 2025-26 accurately |
| Interest rate measures    | Partial — retrieved table data but missed narrative policy text    |

- Pattern identified: pipeline handles tabular data well, struggles when answers live in narrative prose that spans multiple chunks
- Root cause hypothesis: chunks are splitting mid-paragraph in dense policy sections, breaking the context needed to answer qualitative questions
- Day 4 fix: benchmark chunk_size=500 vs 1000 on the same questions and measure answer quality difference
