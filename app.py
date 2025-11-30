# app.py

from dotenv import load_dotenv
load_dotenv()   # load FIRST

import streamlit as st
from backend.embeddings import index_paper
from backend.rag import answer_question

import streamlit as st
from dotenv import load_dotenv
import os

from backend.pdf_utils import extract_pdf
from backend.chunking import chunk_text
from backend.embeddings import index_paper
from backend.rag import answer_question
from backend.summarizer import generate_summary
from backend.concept_map import build_concept_map

load_dotenv()  # loads OPENAI_API_KEY from .env

st.set_page_config(page_title="AI Research Paper Q&A", layout="wide")

st.title("📚 AI Research Paper Q&A Platform")

# Session state
if "paper_id" not in st.session_state:
    st.session_state.paper_id = None
if "paper_text" not in st.session_state:
    st.session_state.paper_text = ""
if "paper_meta" not in st.session_state:
    st.session_state.paper_meta = {}

uploaded_file = st.file_uploader("Upload a research paper (PDF)", type=["pdf"])

if uploaded_file is not None:
    if st.button("Process Paper"):
        with st.spinner("Extracting and indexing..."):
            pdf_data = extract_pdf(uploaded_file)
            full_text = pdf_data["full_text"]
            meta = pdf_data["metadata"]

            chunks = chunk_text(full_text)
            # Use file name as paper_id for now
            paper_id = uploaded_file.name

            index_paper(paper_id, chunks)

            st.session_state.paper_id = paper_id
            st.session_state.paper_text = full_text
            st.session_state.paper_meta = meta

        st.success("Paper processed successfully!")

if st.session_state.paper_id is None:
    st.info("Upload and process a PDF to start.")
else:
    paper_id = st.session_state.paper_id
    full_text = st.session_state.paper_text
    meta = st.session_state.paper_meta

    # Layout: columns + tabs
    st.sidebar.subheader("Paper Info")
    st.sidebar.write(f"**Title:** {meta.get('title')}")
    st.sidebar.write(f"**Author:** {meta.get('author')}")
    st.sidebar.write(f"**Pages:** {meta.get('num_pages')}")

    tab1, tab2, tab3 = st.tabs(["💬 Q&A", "📝 Summaries", "📈 Concept Map"])

    # --- Tab 1: Q&A ---
    with tab1:
        st.subheader("Ask questions about the paper")
        question = st.text_input("Your question")

        if st.button("Get Answer"):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Thinking..."):
                    result = answer_question(paper_id, question)

                st.markdown("### Answer")
                st.write(result["answer"])

                if result["chunks"]:
                    with st.expander("Show retrieved chunks (after re-ranking)"):
                        for i, c in enumerate(result["chunks"], start=1):
                            st.markdown(f"**Chunk {i} (ID: {c['chunk_id']})**")
                            st.write(c["text"])
                            st.caption(
                                f"Cosine score: {c.get('cosine_score', 0):.3f}, "
                                f"Lexical score: {c.get('lex_score', 0):.3f}, "
                                f"Hybrid score: {c.get('hybrid_score', 0):.3f}"
                            )
                            st.markdown("---")

    # --- Tab 2: Summaries ---
    with tab2:
        st.subheader("Summaries at different levels")

        level = st.selectbox(
            "Choose explanation level",
            ["beginner", "intermediate", "expert"]
        )

        if st.button("Generate Summary"):
            with st.spinner("Summarizing..."):
                summary = generate_summary(full_text, level=level)
            st.markdown("### Summary")
            st.write(summary)

    # --- Tab 3: Concept Map ---
    with tab3:
        st.subheader("Concept Map")
        if st.button("Generate Concept Map"):
            with st.spinner("Extracting concept map..."):
                cmap = build_concept_map(full_text)

            nodes = cmap.get("nodes", [])
            edges = cmap.get("edges", [])

            st.markdown("**Nodes:**")
            st.write(nodes)
            st.markdown("**Edges (source → target):**")
            st.write(edges)

            # Simple graphviz visualization
            if nodes and edges:
                graphviz_src = "digraph G {\n"
                for n in nodes:
                    graphviz_src += f'  "{n}";\n'
                for src, tgt in edges:
                    graphviz_src += f'  "{src}" -> "{tgt}";\n'
                graphviz_src += "}\n"

                st.graphviz_chart(graphviz_src)
            else:
                st.info("Could not build a detailed concept map. Try another paper.")
