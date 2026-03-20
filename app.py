"""
Streamlit web interface for Relationship Extractor (OpenAI-only, no sidebar).
"""

import os
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from relationship_extractor import RelationshipExtractor

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
load_dotenv(find_dotenv(), override=False)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
MODEL = "gpt-4o-mini"  # default model
INCLUDE_IMPLICIT = False  # default extraction behavior

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Relationship Extractor", layout="wide")

st.markdown("""
    <style>
    .main { max-width: 1400px; }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p { font-size: 1.1em; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
st.title("Relationship Extractor")
st.markdown("Extract subject–predicate–object relationships from text using OpenAI")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tabs = st.tabs(["Extract", "Analyze", "Guide"])

# ---------------------------------------------------------------------------
# TAB 1: Extract
# ---------------------------------------------------------------------------
with tabs[0]:
    col1, col2 = st.columns([3, 1])
    with col1:
        input_method = st.radio("Input Method", ["Paste Text", "Upload File"], horizontal=True)

    text = ""
    if input_method == "Paste Text":
        text = st.text_area("Enter text for relationship extraction", height=300, placeholder="Paste your text here...")
    else:
        uploaded_file = st.file_uploader("Upload a text file", type=["txt", "md", "pdf"], help="Supported: .txt, .md, .pdf")
        if uploaded_file:
            if uploaded_file.type == "application/pdf":
                try:
                    import PyPDF2
                    reader = PyPDF2.PdfReader(uploaded_file)
                    text = "".join([p.extract_text() or "" for p in reader.pages])
                except ImportError:
                    st.error("Install PyPDF2 for PDF support: pip install PyPDF2")
            else:
                text = uploaded_file.read().decode("utf-8", errors="ignore")
            st.success(f"Loaded {len(text)} characters")

    if st.button("Extract Relationships", type="primary", use_container_width=True):
        if not OPENAI_API_KEY:
            st.error("Missing OPENAI_API_KEY in .env file")
        elif not text.strip():
            st.error("Please enter or upload text to analyze")
        else:
            try:
                extractor = RelationshipExtractor(api_key=OPENAI_API_KEY, model=MODEL)
                with st.spinner("Extracting relationships..."):
                    df = extractor.extract(text, include_implicit=INCLUDE_IMPLICIT)
                st.session_state.extracted_df = df
                st.session_state.original_text = text

                if df.empty:
                    st.warning("No relationships found")
                else:
                    st.success(f"Extracted {len(df)} unique relationships")

                    st.subheader(f"Relationships ({len(df)})")
                    df_display = df.copy()
                    df_display.insert(0, "#", range(1, len(df_display) + 1))
                    st.dataframe(df_display, use_container_width=True, height=400)

                    with st.expander("View source quotes"):
                        for _, row in df.iterrows():
                            c1, c2 = st.columns([3, 1])
                            with c1:
                                st.markdown(f"**{row['subject']}** → *{row['predicate']}* → **{row['object']}**")
                                st.caption(f"Source: {row['source_quote']}")
                            with c2:
                                if bool(row.get("negated", False)):
                                    st.warning("Negated")

                    c1, c2, c3 = st.columns(3)
                    with c1: st.metric("Total Relationships", len(df))
                    with c2:
                        negated_count = df[df.get("negated") == True].shape[0] if "negated" in df.columns else 0
                        st.metric("Negated", negated_count)
                    with c3: st.metric("Unique Subjects", df["subject"].nunique())

                    st.divider()
                    st.subheader("Export Results")
                    ec1, ec2 = st.columns(2)
                    with ec1:
                        st.download_button("Download as CSV", df.to_csv(index=False), "relationships.csv", "text/csv", use_container_width=True)
                    with ec2:
                        st.download_button("Download as JSON", df.to_json(orient="records", indent=2),
                                           "relationships.json", "application/json", use_container_width=True)
                    kg = "\n".join([f"{r['subject']} | {r['predicate']} | {r['object']}" for _, r in df.iterrows()])
                    st.download_button("Download as Knowledge Graph (triplets)", kg, "knowledge_graph.txt", "text/plain", use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")

# ---------------------------------------------------------------------------
# TAB 2: Analyze
# ---------------------------------------------------------------------------
with tabs[1]:
    if "extracted_df" in st.session_state and not st.session_state.extracted_df.empty:
        df = st.session_state.extracted_df
        st.subheader("Relationship Analysis")
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Total Relationships", len(df))
        with c2: st.metric("Unique Subjects", df["subject"].nunique())
        with c3: st.metric("Unique Objects", df["object"].nunique())
        with c4: st.metric("Unique Predicates", df["predicate"].nunique())

        st.divider()
        g1, g2 = st.columns(2)
        with g1:
            st.subheader("Top Predicates")
            pred_counts = df["predicate"].value_counts().head(10)
            fig = px.bar(x=pred_counts.values, y=pred_counts.index, orientation="h",
                         labels={"x": "Count", "y": "Predicate"}, height=400)
            st.plotly_chart(fig, use_container_width=True)
        with g2:
            st.subheader("Top Subjects")
            subj_counts = df["subject"].value_counts().head(10)
            fig = px.bar(x=subj_counts.values, y=subj_counts.index, orientation="h",
                         labels={"x": "Count", "y": "Subject"}, height=400)
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Filter Relationships")
        f1, f2, f3 = st.columns(3)
        with f1: search_subject = st.text_input("Search subject", key="search_subj")
        with f2: search_predicate = st.text_input("Search predicate", key="search_pred")
        with f3: search_object = st.text_input("Search object", key="search_obj")

        filtered = df.copy()
        if search_subject:
            filtered = filtered[filtered["subject"].str.contains(search_subject, case=False, na=False)]
        if search_predicate:
            filtered = filtered[filtered["predicate"].str.contains(search_predicate, case=False, na=False)]
        if search_object:
            filtered = filtered[filtered["object"].str.contains(search_object, case=False, na=False)]

        st.metric("Filtered Results", len(filtered))
        st.dataframe(filtered, use_container_width=True)
    else:
        st.info("Extract relationships first to analyze.")

# ---------------------------------------------------------------------------
# TAB 3: Guide
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("How to Use")
    st.markdown("""
    ### What is Relationship Extraction?
    Relationship extraction identifies structured relationships from unstructured text as triplets.

    **Pass 1:** Extract explicit relationships from text chunks  
    **Pass 2:** Cross-chunk reasoning (no pronoun resolution)  
    **Pass 3 (optional):** Identify implicit relationships

    Pronoun resolution applies only to Pass 1.  
    Negation flagging identifies explicit negatives.  
    The model avoids inference beyond the text.
    """)
    st.divider()
    st.markdown("<div style='text-align:center;color:#888;font-size:0.9em;'>Powered by OpenAI | Multi-Pass Relationship Extraction</div>", unsafe_allow_html=True)
