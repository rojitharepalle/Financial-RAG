import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from groq import Groq
from sentence_transformers import CrossEncoder

load_dotenv()

CHROMA_PATH = "chroma_db"
print("Starting evaluation...")
# Load existing vectorstore
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings
)

print(f"Total chunks in store: {vectorstore._collection.count()}")

# Load cross-encoder for reranking
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 10 evaluation questions
questions = [
    "What is India's real GDP growth rate for 2024-25?",
    "What is the RBI repo rate as of March 2025?",
    "What is the CPI inflation projection for 2025-26?",
    "What is India's fiscal deficit target for 2025-26?",
    "What is the foreign exchange reserve level as of March 2025?",
    "What is the growth rate of bank credit in 2024-25?",
    "What is the unemployment rate in India for 2024-25?",
    "What measures did RBI take regarding liquidity in 2024-25?",
    "What is India's current account deficit for 2024-25?",
    "What is the RBI's inflation target band?"
]

for i, question in enumerate(questions):
    # Retrieve
    retrieved = vectorstore.similarity_search(
        question,
        k=10,
        filter={"section": "economic_review"}
    )

    # Rerank
    pairs = [[question, doc.page_content] for doc in retrieved]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, retrieved), key=lambda x: x[0], reverse=True)
    top3 = [doc for score, doc in ranked[:3]]

    context = "\n\n".join([doc.page_content for doc in top3])

    # Generate answer
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """Answer using ONLY the context provided.
Rules:
1. If the answer is not explicitly in the context, say exactly: 'Not found in provided context.'
2. After your answer, always add: 'Source: Page [X]' citing which page your answer came from.
3. Never use knowledge outside the provided context.
4. If multiple pages contain relevant info, cite all of them."""
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ]
    )

    print(f"\nQ{i+1}: {question}")
    print(f"A: {response.choices[0].message.content}")
    print("-" * 50)