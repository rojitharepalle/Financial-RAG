import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from groq import Groq

load_dotenv()

CHROMA_PATH = "chroma_db"

# Load existing vectorstore
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings
)

print(f"Total chunks in store: {vectorstore._collection.count()}")


# Query
question = "What is the RBI's GDP growth forecast for 2024-25?"
retrieved = vectorstore.similarity_search(question, k=5)

question = "real GDP growth 6.4 per cent 2024-25 India"
retrieved = vectorstore.similarity_search(question, k=5)

print("\nRetrieved chunks:")
for i, doc in enumerate(retrieved):
    print(f"\n--- Chunk {i+1} --- Page: {doc.metadata.get('page', 'unknown')}")
    print(doc.page_content[:200])

context = "\n\n".join([doc.page_content for doc in retrieved])

# Generate answer
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": "Answer using only the context provided."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]
)

print(response.choices[0].message.content)