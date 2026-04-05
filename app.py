import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from groq import Groq
from sentence_transformers import CrossEncoder

load_dotenv()

CHROMA_PATH = "chroma_db"

st.set_page_config(
    page_title="Financial Document Q&A",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Financial Document Q&A")
st.markdown("Ask questions about the **RBI Annual Report 2024-25**")
st.markdown("---")

@st.cache_resource
def load_pipeline():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return vectorstore, reranker, client

with st.spinner("Loading pipeline..."):
    vectorstore, reranker, client = load_pipeline()

st.success(f"Pipeline ready. {vectorstore._collection.count()} chunks loaded.")

# Section filter option
section = st.selectbox(
    "Search section:",
    ["economic_review", "monetary_policy", "financial_markets", "overview", "other"]
)

# Question input
question = st.text_input(
    "Ask a question:",
    placeholder="What is India's GDP growth rate for 2024-25?"
)

if st.button("Get Answer") and question:
    with st.spinner("Searching and generating answer..."):

        # Retrieve
        retrieved = vectorstore.similarity_search(
            question,
            k=10,
            filter={"section": section}
        )

        if not retrieved:
            st.warning("No chunks found in this section. Try a different section.")
        else:
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

            # Display answer
            st.markdown("### Answer")
            st.write(answer)

            # Show retrieved chunks
            with st.expander("View retrieved chunks"):
                for i, doc in enumerate(top3):
                    st.markdown(f"**Chunk {i+1} — Page {doc.metadata.get('page', 'unknown')}**")
                    st.text(doc.page_content[:300])
                    st.markdown("---")