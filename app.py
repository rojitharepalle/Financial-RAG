import os
import re
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq
from sentence_transformers import CrossEncoder

load_dotenv()

CHROMA_PATH = "chroma_db"
PDF_PATH = "rbi_report.pdf"

st.set_page_config(
    page_title="Financial Document Q&A",
    page_icon="🏦",
    layout="wide"
)


st.markdown("""
    <style>
    .stApp { background-color: #ffffff; color: #1a1a1a; }
    .block-container { 
        padding-top: 2.5rem !important; 
        padding-bottom: 1rem !important; 
    }

    /* Sidebar buttons */
    section[data-testid="stSidebar"] .stButton > button {
        background-color: #1a1a2e !important;
        color: white !important;
        border-radius: 6px !important;
        border: none !important;
        width: 100% !important;
        text-align: left !important;
        padding: 8px 12px !important;
        font-size: 13px !important;
        margin-bottom: 4px !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: #4f8ef7 !important;
    }

    /* Main Get Answer button */
    .main .stButton > button {
        background-color: #1a1a2e !important;
        color: white !important;
        border-radius: 6px !important;
        padding: 10px 24px !important;
        font-size: 15px !important;
        border: none !important;
        width: 100% !important;
    }
    .main .stButton > button:hover {
        background-color: #4f8ef7 !important;
        color: white !important;
    }

    /* Text input */
    .stTextInput > div > div > input {
        background-color: #f5f5f5 !important;
        border: 1px solid #cccccc !important;
        border-radius: 6px !important;
        color: #1a1a1a !important;
        font-size: 15px !important;
        padding: 10px !important;
    }
    .stTextInput label, 
    .stTextInput label p,
    div[data-testid="stTextInput"] label {
        font-size: 14px !important;
        font-weight: 700 !important;
        color: #1a1a1a !important;
        display: block !important;
        visibility: visible !important;
        height: auto !important;
        opacity: 1 !important;
    }

    /* Selectbox */
    div[data-testid="stSelectbox"] > div > div {
        border: 1px solid #cccccc !important;
        border-radius: 6px !important;
        background-color: #f5f5f5 !important;
        color: #1a1a1a !important;
    }
    div[data-testid="stSelectbox"] label,
    div[data-testid="stSelectbox"] label p {
        font-size: 14px !important;
        font-weight: 700 !important;
        color: #1a1a1a !important;
        opacity: 1 !important;
    }

    /* Success message */
    .stSuccess {
        background-color: #f0faf0 !important;
        border: 1px solid #c3e6c3 !important;
        border-radius: 6px !important;
        margin-bottom: 4px !important;
        margin-top: 0px !important;
    }

    /* Answer box */
    h3 { color: #1a1a2e; }
    .answer-box {
        background-color: #f8f9ff;
        border-left: 4px solid #4f8ef7;
        border-radius: 6px;
        padding: 16px 20px;
        margin-top: 8px;
        font-size: 15px;
        line-height: 1.6;
        color: #1a1a1a;
    }

    /* Expander */
    .stExpander {
        border: 1px solid #e0e0e0 !important;
        border-radius: 6px !important;
        background-color: #ffffff !important;
    }
    .stExpander > div {
        background-color: #ffffff !important;
    }
    pre {
        color: #1a1a1a !important;
        background-color: #f5f5f5 !important;
        padding: 8px !important;
        border-radius: 4px !important;
        white-space: pre-wrap !important;
        font-size: 13px !important;
    }

    /* Caption spacing */
    .stCaption { margin-bottom: 2px !important; margin-top: 8px !important; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
        <svg width="48" height="48" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect width="48" height="48" rx="10" fill="#1a1a2e"/>
            <rect x="12" y="8" width="20" height="26" rx="2" fill="white" opacity="0.9"/>
            <rect x="15" y="13" width="14" height="2" rx="1" fill="#1a1a2e"/>
            <rect x="15" y="17" width="14" height="2" rx="1" fill="#1a1a2e"/>
            <rect x="15" y="21" width="10" height="2" rx="1" fill="#1a1a2e"/>
            <circle cx="32" cy="32" r="8" fill="#4f8ef7"/>
            <circle cx="32" cy="32" r="5" fill="#1a1a2e"/>
            <line x1="36" y1="36" x2="40" y2="40" stroke="white" stroke-width="2.5" stroke-linecap="round"/>
        </svg>
        <span style="font-size: 1.8rem; font-weight: 700; color: #1a1a2e;">Financial Document Q&A</span>
    </div>
        <p style="color:#555;font-size:14px;margin-top:0;margin-bottom:4px;">
        Ask questions about the <strong>RBI Annual Report 2024-25</strong>
    </p>
    <hr style="border:none;border-top:1px solid #e0e0e0;margin:8px 0 8px 0;">
""", unsafe_allow_html=True)



if "selected_question" not in st.session_state:
    st.session_state.selected_question = ""
with st.sidebar:
    st.markdown("## Sample Questions")
    st.markdown("Click any question to try it:")
    
    sample_questions = [
        "What is India's real GDP growth rate for 2024-25?",
        "What is the RBI repo rate as of March 2025?",
        "What is the CPI inflation projection for 2025-26?",
        "What measures did RBI take regarding liquidity?",
        "What is the growth rate of bank credit in 2024-25?",
        "What is India's fiscal deficit for 2025-26?"
    ]
    
    
    for q in sample_questions:
        if st.button(q, key=q):
            st.session_state.selected_question = q
    
    st.markdown("---")
    st.markdown("**About**")
    st.markdown("Built on RBI Annual Report 2024-25. 318 pages. 1,028 chunks.")
    st.markdown("[GitHub](https://github.com/rojitharepalle/financial-rag)")

def is_meaningful_chunk(text):
    words = text.split()
    if len(words) < 15:
        return False
    month_pattern = r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{1,2}\b'
    if len(re.findall(month_pattern, text)) >= 4:
        return False
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    number_only_lines = sum(1 for l in lines if re.match(r'^-?\d+\.?\d*$', l))
    if len(lines) > 0 and number_only_lines / len(lines) > 0.3:
        return False
    digit_chars = sum(1 for c in text if c.isdigit() or c in '-./%')
    digit_ratio = digit_chars / len(text)
    avg_word_length = sum(len(w) for w in words) / len(words)
    if digit_ratio > 0.3 and avg_word_length < 4.5:
        return False
    meaningful = [w for w in words if len(w) >= 4
                  and not re.match(r'^[\d\-\/\.]+$', w)]
    return len(meaningful) / len(words) >= 0.5

def get_section(page_num):
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

@st.cache_resource
def load_pipeline():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if not os.path.exists(CHROMA_PATH):
        build_msg = st.info("Building vector store for first time. This takes 2-3 minutes...")

        loader = PyPDFLoader(PDF_PATH)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(pages)

        filtered_chunks = [c for c in chunks if is_meaningful_chunk(c.page_content)]
        for chunk in filtered_chunks:
            chunk.metadata['section'] = get_section(chunk.metadata.get('page', 0))

        vectorstore = Chroma.from_documents(
            documents=filtered_chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PATH
        )
        build_msg.empty()
    else:
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
st.empty()


section = st.selectbox(
    "Search section:",
    options = ["economic_review", "monetary_policy", "financial_markets", "overview", "other"],
    format_func=lambda x:{
        "economic_review": "📈 Economic Review (GDP, inflation, credit, trade)",
        "monetary_policy": "🏦 Monetary Policy (repo rate, MPC decisions, liquidity)",
        "financial_markets": "📊 Financial Markets (bonds, forex, equity)",
        "overview": "📋 Overview (summary, highlights, key indicators)",
        "other": "📁 Other Sections"
    }[x]
)

question = st.text_input(
    "Ask a question:",
    value = st.session_state.selected_question,
    placeholder="What is India's GDP growth rate for 2024-25?"
)

if st.button("Get Answer") and question:
    with st.spinner("Searching and generating answer..."):

        retrieved = vectorstore.similarity_search(
            question,
            k=10,
            filter={"section": section}
        )

        if not retrieved:
            st.warning("No chunks found in this section. Try a different section.")
        else:
            pairs = [[question, doc.page_content] for doc in retrieved]
            scores = reranker.predict(pairs)
            ranked = sorted(zip(scores, retrieved), key=lambda x: x[0], reverse=True)
            top3 = [doc for score, doc in ranked[:3]]
            context = "\n\n".join([doc.page_content for doc in top3])

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": """Answer using ONLY the context provided.
Rules:
1. If the answer is not explicitly in the context, say exactly: 'Not found in provided context.'
2. Do NOT include any source or page number citation in your answer.
3. Never use knowledge outside the provided context."""
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {question}"
                    }
                ]
            )

            answer = response.choices[0].message.content

            st.markdown("### Answer")
            st.markdown(f'<div class="answer-box">{answer}</div>',
                       unsafe_allow_html=True)

            pages_cited = [str(doc.metadata.get('page', 'unknown')) for doc in top3]
            st.markdown(f"<p style='color:#666;font-size:13px;margin-top:8px;'>Sources: PDF pages {', '.join(pages_cited)}</p>",
                        unsafe_allow_html=True)

            with st.expander("View retrieved chunks"):
                for i, doc in enumerate(top3):
                    st.markdown(f"**Chunk {i+1} — Page {doc.metadata.get('page', 'unknown')}**")
                    st.text(doc.page_content[:300])
                    st.markdown("---")