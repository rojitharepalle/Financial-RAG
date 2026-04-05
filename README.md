# Financial Document RAG Pipeline

A Retrieval-Augmented Generation (RAG) pipeline for querying
financial documents using LangChain, ChromaDB, and Groq.

Built on the RBI Annual Report 2024-25 (318 pages, 1,194 chunks
at chunk_size=1000) as the source document.

## Why I Built This

Financial documents are dense, long, and hard to query manually.
This pipeline chunks, embeds, and retrieves relevant sections to
answer specific questions about financial data — the kind of
problem every fintech AI team is solving at scale.

##Demo

<img width="1902" height="965" alt="image" src="https://github.com/user-attachments/assets/ac285d53-d3b5-4cc5-8205-3a037bb483a4" />

Ask questions about the RBI Annual Report 2024-25 and get 
answers grounded in the actual document with source citations.


## Tech Stack

- **LangChain** — document loading and text splitting
- **PyPDF** — PDF ingestion
- **ChromaDB** — vector store for semantic search
- **HuggingFace sentence-transformers** — embeddings (all-MiniLM-L6-v2)
- **Cross-Encoder** — reranking (ms-marco-MiniLM-L-6-v2)
- **Groq** — LLM for answer generation (llama-3.3-70b-versatile)
- **Python 3.10+**

## Project Structure

```
financial-rag/
├── embed_store.py    # builds ChromaDB once from PDF
├── rag.py            # queries ChromaDB and generates answers
├── evaluate.py       # runs 10 standard questions, saves CSV
├── .env              # GROQ_API_KEY (not committed)
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

Run evaluation:

```bash
python evaluate.py
```

## Chunking Parameters

```python
chunk_size=1000      # characters per chunk
chunk_overlap=200    # overlap between consecutive chunks
```

Chose 1000 over 500 because RBI report paragraphs are dense
with financial terminology — smaller chunks lose context.

## What I Learned — Day 1

- RBI 2024-25 report loads into **318 pages** and produces
  **1,194 chunks** at chunk_size=1000, overlap=200
- `RecursiveCharacterTextSplitter` splits on paragraph
  boundaries first before falling back to character splits —
  this preserves more semantic meaning than fixed-size splits
- Chunk overlap of 200 characters ensures context is not
  lost at boundaries between chunks

## What I Learned — Day 2

- ChromaDB stores 1,194 chunks and retrieval works
- Initial retrieval returned chart axis labels as top results
- Root cause: PyPDF extracts graph tick marks as plain text,
  indistinguishable from paragraphs at chunk level
- Fix: digit ratio filter — chunks where >30% of characters
  are digits or symbols are dropped before storage
- Result: chart noise eliminated, all top 3 results now
  return genuine financial prose
- Chunks after filtering: 1,028 of 1,194 (166 removed as noise)

## What I Learned — Day 3

- Connected ChromaDB retrieval to Groq LLM end-to-end
- Tested 3 questions against the RBI Annual Report

| Question                  | Result                                                       |
| ------------------------- | ------------------------------------------------------------ |
| RBI's stance on inflation | Weak — bibliography chunks retrieved, not policy sections    |
| GDP growth forecast       | Strong — returned 6.4% for 2024-25 and 6.7% for 2025-26      |
| Interest rate measures    | Partial — table data retrieved, missed narrative policy text |

- Pattern: larger chunks improve narrative policy retrieval
  but noise filter accidentally removed numerical forecast chunks

## What I Learned — Day 4

### Chunk Size Benchmarking — 500 vs 1000

| Question                  | chunk_size=500                  | chunk_size=1000                              |
| ------------------------- | ------------------------------- | -------------------------------------------- |
| RBI's stance on inflation | Weak — bibliography chunks      | Weak — inferred from research topics         |
| GDP growth forecast       | Strong — 6.4% and 6.7% returned | Weak — IMF forecast returned instead         |
| Interest rate measures    | Partial — table data only       | Strong — specific measures with basis points |

### Noise Filter Iterations

- Initial filter removed 166 chunks but also removed
  legitimate numerical policy chunks containing GDP forecasts
- Refined filter with 3 conditions:
  - Month sequence pattern (Jan-21, Feb-21 etc.) → filter
  - Lines with standalone numbers ratio > 30% → filter
  - High digit ratio AND short average word length → filter
- Final result: 1,028 clean chunks from 1,194 total

### Critical Finding — Silent Hallucination

- When retrieved chunks don't contain the answer,
  `llama-3.3-70b-versatile` pulls numbers from training
  data instead of saying "not found"
- Example: Asked for India GDP forecast, returned 7.1%
  confidently — number not present in any retrieved chunk
- This is the most dangerous failure mode in production RAG
  — the system sounds confident while being wrong
- Fix: force LLM to cite the specific chunk and page used

### Retrieval Failure — Deep Document Chunks

- Specific forecast chunks buried deep in document were
  never retrieved regardless of question wording
- Root cause: `all-MiniLM-L6-v2` does not capture semantic
  similarity between forecast questions and forecast prose
  at depth
- Fix: reranking with cross-encoder model

## What I Learned — Day 5

### Reranking with Cross-Encoder

- Added `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker
- Pipeline now retrieves k=10 candidates, reranks by true
  relevance, passes top 3 to LLM
- Reranking improved answer quality for policy questions

### Critical Finding — Semantic Ambiguity

- The number 6.4% appears in 4 different chunks meaning
  completely different things:
  - GVA growth (page 46)
  - Electricity price inflation (page 62)
  - Export growth (page 98)
  - G-sec holdings (page 176)
- Embedding model cannot distinguish between these contexts
- A question about "GDP growth forecast" retrieves whichever
  6.4% chunk has highest cosine similarity — not the correct one
- This is a fundamental limitation of dense retrieval for
  numerical data in financial documents
- Fix: metadata filtering by document section

## What I Learned — Day 6

### Metadata Section Filtering

- Tagged all 1,028 chunks with document section metadata:
  - Pages 1-30 → overview
  - Pages 31-110 → economic_review
  - Pages 111-160 → monetary_policy
  - Pages 161-220 → financial_markets
  - Pages 221+ → other
- GDP questions now filter to economic_review section only
- Eliminates cross-section pollution where same number
  means different things in different chapters

### Hallucination Detection

- Updated system prompt with strict rules:
  - Answer only from provided context
  - Return "Not found in provided context" if answer absent
  - Always cite source page number
- Before: returned 7.1% GDP confidently from training data
- After: returns "Not found in provided context. Source: Page 78"
- Production-grade behaviour — traceable, honest, safe

### Why This Matters In Fintech

- A wrong number stated confidently in a financial system
  is not an inconvenience — it is a compliance risk
- Silent hallucination + semantic ambiguity + no source
  citation = untrusted AI in regulated environments
- Every fix in this pipeline addresses a real production
  failure mode, not a tutorial edge case

## What I Learned — Day 10

### Evaluation Scoring

- Built automated evaluation script across 10 standard
  financial questions with known expected answers
- Final score: 9/10 correct
- Results saved to timestamped CSV for repeatability

| Question                             | Score                                        |
| ------------------------------------ | -------------------------------------------- |
| India real GDP growth rate 2024-25   | Correct — 6.5%                               |
| RBI repo rate March 2025             | Correct — 6.0%                               |
| CPI inflation projection 2025-26     | Correct — 4.0%                               |
| Fiscal deficit target 2025-26        | Correct                                      |
| Foreign exchange reserves March 2025 | Correct — 11 months import cover             |
| Bank credit growth 2024-25           | Correct — 11.8%                              |
| Unemployment rate 2024-25            | Correct — not in document, answered honestly |
| Liquidity measures 2024-25           | Wrong — vague context retrieved              |
| Current account deficit 2024-25      | Correct — not in document, answered honestly |
| RBI inflation target band            | Correct                                      |

- Scoring logic: keyword matching against expected answers
- "Not found in provided context" treated as correct when
  document genuinely lacks the information
- Only failure: liquidity measures — section filter too
  restrictive, relevant chunks in monetary_policy section
  not searched

## Roadmap

- [x] PDF ingestion and chunking
- [x] Embedding and vector store with ChromaDB
- [x] End-to-end Q&A with Groq LLM
- [x] Noise filter for chart extraction artifacts
- [x] Chunk size benchmarking — 500 vs 1000
- [x] Reranking with cross-encoder model
- [x] Metadata section filtering
- [x] Hallucination detection with source citation
- [x] Evaluation scoring — 9/10 on 10 standard questions
- [x] Streamlit demo interface
