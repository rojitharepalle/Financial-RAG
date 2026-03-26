from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = "rbi_report.pdf"

loader = PyPDFLoader(PDF_PATH)
pages = loader.load()
print(f"Total pages loaded: {len(pages)}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(pages)
print(f"Total chunks created: {len(chunks)}")

for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i+1} ---")
    print(f"Length: {len(chunk.page_content)} characters")
    print(chunk.page_content[:300])