import os
import csv
from datetime import datetime
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from groq import Groq
from sentence_transformers import CrossEncoder

load_dotenv()

CHROMA_PATH = "chroma_db"

# Load vectorstore
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings
)

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Questions with known expected answers
eval_questions = [
    {"question": "What is India's real GDP growth rate for 2024-25?", "expected": "6.5 per cent"},
    {"question": "What is the RBI repo rate as of March 2025?", "expected": "6.0 per cent"},
    {"question": "What is the CPI inflation projection for 2025-26?", "expected": "4.0 per cent"},
    {"question": "What is India's fiscal deficit target for 2025-26?", "expected": "4.4 per cent of GDP"},
    {"question": "What is the foreign exchange reserve level as of March 2025?", "expected": "11 months import cover"},
    {"question": "What is the growth rate of bank credit in 2024-25?", "expected": "11.8 per cent"},
    {"question": "What is the unemployment rate in India for 2024-25?", "expected": "not found"},
    {"question": "What measures did RBI take regarding liquidity in 2024-25?", "expected": "liquidity management operations"},
    {"question": "What is India's current account deficit for 2024-25?", "expected": "not found"},
    {"question": "What is the RBI's inflation target band?", "expected": "4 per cent with 2-6 per cent band"}
]

results = []

print("Running evaluation...\n")

for i, item in enumerate(eval_questions):
    question = item["question"]
    expected = item["expected"]

    # Retrieve and rerank
    retrieved = vectorstore.similarity_search(
        question,
        k=10,
        filter={"section": "economic_review"}
    )

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
2. After your answer, always add: 'Source: Page [X]'
3. Never use knowledge outside the provided context."""
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ]
    )

    answer = response.choices[0].message.content

    # Simple scoring
    answer_lower = answer.lower()
    expected_lower = expected.lower()

    if "not found in provided context" in answer_lower and expected_lower == "not found":
        score = "correct"
    elif any(word in answer_lower for word in expected_lower.split() if len(word) > 3):
        score = "correct"
    else:
        score = "wrong"

    results.append({
        "question": question,
        "expected": expected,
        "answer": answer,
        "score": score
    })

    print(f"Q{i+1}: {score.upper()} — {question}")

# Save to CSV
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"evaluation_{timestamp}.csv"

with open(filename, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["question", "expected", "answer", "score"])
    writer.writeheader()
    writer.writerows(results)

# Summary
correct = sum(1 for r in results if r["score"] == "correct")
wrong = sum(1 for r in results if r["score"] == "wrong")

print(f"\nEvaluation complete.")
print(f"Correct: {correct}/10")
print(f"Wrong: {wrong}/10")
print(f"Results saved to: {filename}")