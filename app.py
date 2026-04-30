"""
app.py
Streamlit frontend for the Personalized Study Plan Generator.
RAG Pipeline: PDF → Chunks → FAISS Embeddings → Groq LLM → Study Plan
"""

import os
import streamlit as st
from dotenv import load_dotenv

from utils.pdf_loader import load_document, split_into_chunks
from utils.vectorstore import build_vectorstore
from utils.planner import generate_study_plan

load_dotenv()

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StudyPlan AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 1rem;
        color: #6b7280;
        margin-top: 0.2rem;
        margin-bottom: 2rem;
    }
    .step-badge {
        background: #eef2ff;
        color: #4338ca;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 10px;
        border-radius: 999px;
        display: inline-block;
        margin-bottom: 6px;
    }
    .plan-box {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        line-height: 1.8;
    }
    .info-chip {
        background: #dbeafe;
        color: #1d4ed8;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 500;
        display: inline-block;
        margin: 2px;
    }
    .success-banner {
        background: #dcfce7;
        border: 1px solid #86efac;
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        color: #166534;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">📚 StudyPlan AI</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload your syllabus or notes → Get a personalized day-by-day study plan powered by RAG + LLM</p>', unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1.6], gap="large")

# ─── LEFT SIDEBAR: Inputs ────────────────────────────────────────────────────
with col_left:
    st.markdown('<span class="step-badge">STEP 1 — API KEY</span>', unsafe_allow_html=True)
    groq_api_key = st.text_input(
        "Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Free at console.groq.com — no credit card needed",
        placeholder="gsk_...",
    )
    st.caption("🔑 [Get free key at console.groq.com](https://console.groq.com)")

    st.divider()

    st.markdown('<span class="step-badge">STEP 2 — UPLOAD DOCUMENT</span>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload Syllabus or Notes",
        type=["pdf", "txt"],
        help="PDF or plain text — syllabus, lecture notes, or any study material",
    )

    if uploaded_file:
        st.markdown(f'<div class="success-banner">✅ Loaded: <strong>{uploaded_file.name}</strong> ({uploaded_file.size // 1024} KB)</div>', unsafe_allow_html=True)

    st.divider()

    st.markdown('<span class="step-badge">STEP 3 — YOUR PREFERENCES</span>', unsafe_allow_html=True)

    days = st.slider("📅 Days until exam", min_value=3, max_value=60, value=14, step=1)
    hours = st.slider("⏱️ Study hours per day", min_value=1.0, max_value=12.0, value=4.0, step=0.5)

    difficulty = st.selectbox(
        "📊 Intensity preference",
        options=["Balanced (mix of topics daily)", "Intense (finish faster, study more per day)", "Light (slow and steady, more revision)"],
    )

    weak_topics = st.text_area(
        "🎯 Topics I want extra focus on (optional)",
        placeholder="e.g. Thermodynamics, Sorting Algorithms, Organic Chemistry...",
        height=80,
    )

    exam_note = st.text_input(
        "📝 Any other context (optional)",
        placeholder="e.g. Final semester exam, competitive exam, project deadline",
    )

    generate_btn = st.button("✨ Generate My Study Plan", type="primary", use_container_width=True)

# ─── RIGHT PANEL: Output ─────────────────────────────────────────────────────
with col_right:
    st.markdown('<span class="step-badge">YOUR PERSONALIZED STUDY PLAN</span>', unsafe_allow_html=True)

    if generate_btn:
        # Validation
        if not groq_api_key or not groq_api_key.startswith("gsk_"):
            st.error("⚠️ Please enter a valid Groq API key (starts with gsk_)")
        elif not uploaded_file:
            st.error("⚠️ Please upload a syllabus or notes file first")
        else:
            try:
                # Phase 1: Parse document
                with st.spinner("📄 Reading your document..."):
                    uploaded_file.seek(0)
                    raw_text = load_document(uploaded_file)
                    chunks = split_into_chunks(raw_text)

                st.markdown(f'<div class="info-chip">📄 {len(chunks)} chunks extracted</div>', unsafe_allow_html=True)

                # Phase 2: Build FAISS vector store
                with st.spinner("🔍 Building knowledge index (FAISS embeddings)..."):
                    vectorstore = build_vectorstore(chunks)

                st.markdown('<div class="info-chip">✅ Vector index built</div>', unsafe_allow_html=True)

                # Phase 3: Generate plan via RAG + LLM
                with st.spinner("🧠 Generating your personalized plan with Llama 3.1 70B..."):
                    plan = generate_study_plan(
                        vectorstore=vectorstore,
                        groq_api_key=groq_api_key,
                        days=days,
                        hours_per_day=hours,
                        weak_topics=weak_topics,
                        exam_note=exam_note,
                        difficulty=difficulty,
                    )

                st.success("🎉 Your study plan is ready!")
                st.divider()

                # Display plan
                st.markdown(plan)

                st.divider()

                # Download button
                st.download_button(
                    label="⬇️ Download Study Plan (.txt)",
                    data=plan,
                    file_name="my_study_plan.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.caption("Common issues: invalid API key, empty PDF, or network error.")

    else:
        # Placeholder state
        st.info("👈 Upload your syllabus and fill in preferences, then click **Generate My Study Plan**")

        st.markdown("#### How it works")
        st.markdown("""
**1. 📄 Document Parsing** — PyMuPDF extracts text from your PDF

**2. ✂️ Smart Chunking** — LangChain splits it into overlapping chunks

**3. 🔢 Embedding** — `all-MiniLM-L6-v2` converts chunks to vectors (runs locally, free)

**4. 🗄️ FAISS Index** — Chunks stored in a local vector database for fast retrieval

**5. 🔍 RAG Retrieval** — Most relevant syllabus sections fetched for your query

**6. 🤖 LLM Generation** — Groq's Llama 3.1 70B writes your personalized plan
        """)

        st.markdown("#### Tips for best results")
        st.markdown("""
- Upload a structured syllabus (with topic names) for the most accurate plan
- Mention specific weak topics to get extra days allocated to them
- For competitive exams, mention it in the "context" field
        """)

# ─── Footer ─────────────────────────────────────────────────────────────────
st.divider()
st.caption("Built with LangChain · FAISS · Groq (Llama 3.1 70B) · Sentence Transformers · Streamlit | All free tools 🚀")
