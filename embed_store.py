from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import re

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
    
    # Check digit/symbol ratio
    digit_chars = sum(1 for c in text if c.isdigit() or c in '-./%')
    total_chars = len(text)
    if digit_chars / total_chars > 0.3:
        return False
    
    meaningful = [w for w in words if len(w) >= min_word_length 
                  and not re.match(r'^[\d\-\/\.]+$', w)]
    ratio = len(meaningful) / len(words)
    return ratio >= threshold

filtered_chunks = [c for c in chunks if is_meaningful_chunk(c.page_content)]
print(f"Chunks after filtering: {len(filtered_chunks)} "
      f"(removed {len(chunks) - len(filtered_chunks)})")


print("Storing chunks in ChromaDB...")
vectorstore = Chroma.from_documents(
    documents=filtered_chunks,
    embedding=embeddings,
    persist_directory=CHROMA_PATH
)

print(f"Done. {len(chunks)} chunks stored in ChromaDB.")

print("\nTesting retrieval...")
query = "What is the RBI inflation target for 2024-25?"
results = vectorstore.similarity_search(query, k=3)

print(f"\nTop 3 chunks for query: '{query}'")
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(f"Page: {doc.metadata.get('page', 'unknown')}")
    print(doc.page_content[:300])