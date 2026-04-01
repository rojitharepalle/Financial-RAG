import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from groq import Groq
from sentence_transformers import CrossEncoder

load_dotenv()

CHROMA_PATH = "chroma_db"

# Load existing vectorstore
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings
)

print(f"Total chunks in store: {vectorstore._collection.count()}")

# Load cross-encoder for reranking
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Query
question = "What is India GDP growth forecast for 2024-25 and 2025-26?"
retrieved = vectorstore.similarity_search(question, k=10)

# Rerank using cross-encoder
pairs = [[question, doc.page_content] for doc in retrieved]
scores = reranker.predict(pairs)

# Sort by score descending
ranked = sorted(zip(scores, retrieved), key=lambda x: x[0], reverse=True)
top3 = [doc for score, doc in ranked[:3]]

print("\nReranked top 3 chunks:")
for i, doc in enumerate(top3):
    print(f"\n--- Chunk {i+1} --- Page: {doc.metadata.get('page', 'unknown')}")
    print(doc.page_content[:200])

context = "\n\n".join([doc.page_content for doc in top3])

# Generate answer
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": "Answer using only the context provided."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]
)

print("\nAnswer:")
print(response.choices[0].message.content)

all_chunks = vectorstore.get()
for i, doc in enumerate(all_chunks['documents']):
    if "6.4 per cent" in doc and "2024-25" in doc:
        metadata = all_chunks['metadatas'][i]
        print(f"\nFound on page: {metadata.get('page')}")
        print(doc[:500])
        print("---")