from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import re


def get_section(page_num):
    """Map page numbers to RBI Annual Report sections."""
    if page_num <= 30:
        return "overview"
    elif page_num <= 110:
        return "economic_review"
    elif page_num <= 160:
        return "monetary_policy"
    elif page_num <= 220:
        return "financial_markets"
    else:
        return "other"

PDF_PATH = "rbi_report.pdf"
CHROMA_PATH = "chroma_db"

print("Loading PDF...")
loader = PyPDFLoader(PDF_PATH)
pages = loader.load()

print("Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(pages)
print(f"Total chunks: {len(chunks)}")

print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
def is_meaningful_chunk(text, min_word_length=4, threshold=0.5):
    words = text.split()
    if len(words) < 15:
        return False
    

    # Filter chart x-axis patterns — month sequences
    month_pattern = r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{1,2}\b'
    month_matches = re.findall(month_pattern, text)
    if len(month_matches) >= 4:
        return False

    # Check for axis label pattern — many short lines
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    short_lines = sum(1 for l in lines if re.match(r'^-?\d+\.?\d*$', l))
    if len(lines) > 0 and short_lines / len(lines) > 0.3:
        return False

    # Calculate digit/symbol ratio
    digit_chars = sum(1 for c in text if c.isdigit() or c in '-./%')
    total_chars = len(text)
    digit_ratio = digit_chars / total_chars
    
    # Calculate average word length
    avg_word_length = sum(len(w) for w in words) / len(words)
    
    # Only filter if BOTH digit ratio is high AND words are short
    # Short words + high digits = chart axis labels
    # Long words + high digits = policy prose with numbers (keep these)
    if digit_ratio > 0.3 and avg_word_length < 4.5:
        return False
    
    meaningful = [w for w in words if len(w) >= min_word_length 
                  and not re.match(r'^[\d\-\/\.]+$', w)]
    ratio = len(meaningful) / len(words)
    return ratio >= threshold

filtered_chunks = [c for c in chunks if is_meaningful_chunk(c.page_content)]
print(f"Chunks after filtering: {len(filtered_chunks)} "
      f"(removed {len(chunks) - len(filtered_chunks)})")


# Add section metadata to each chunk
for chunk in filtered_chunks:
    page = chunk.metadata.get('page', 0)
    chunk.metadata['section'] = get_section(page)

print("Sample metadata:", filtered_chunks[0].metadata)

print("Storing chunks in ChromaDB...")
vectorstore = Chroma.from_documents(
    documents=filtered_chunks,
    embedding=embeddings,
    persist_directory=CHROMA_PATH
)
print(vectorstore._collection.count())
print(f"Done. {len(chunks)} chunks stored in ChromaDB.")

query = "What is India real GDP growth estimate for 2024-25?"
result = vectorstore.similarity_search(query, k=5)

print("\nRetrieved chunks:")
for i, doc in enumerate(result):
    print(f"\n--- Chunk {i+1} --- Page: {doc.metadata.get('page', 'unknown')}")
    print(doc.page_content[:200])

context = "\n\n".join([doc.page_content for doc in result])
