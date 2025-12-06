import os
from pathlib import Path
import requests
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Knowledge Base Search Engine",
    page_icon="🔍",
    layout="wide"
)

# API configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Custom CSS
st.markdown(
    """
    <style>
    .main { padding: 2rem; }
    .stButton>button { width: 100%; }
    .upload-text { font-size: 14px; color: #666; }
    .source-box { background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin: 5px 0; }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.title("🔍 Knowledge Base Search Engine")
st.markdown("Upload documents and ask questions using AI-powered RAG")

# Sidebar
with st.sidebar:
    st.header("📊 System Info")

    # Get stats
    try:
        response = requests.get(f"{API_URL}/stats", timeout=15)
        if response.status_code == 200:
            stats = response.json()
            st.metric("Total Chunks", stats.get("total_chunks", 0))
        else:
            st.warning("Could not fetch stats")
    except Exception:
        st.error("API connection failed")

    st.divider()

    # Clear knowledge base
    st.header("🗑️ Management")
    if st.button("Clear Knowledge Base", type="secondary", key="clear_kb_btn"):
        try:
            response = requests.delete(f"{API_URL}/clear", timeout=30)
            if response.status_code == 200:
                st.success("Knowledge base cleared!")
                st.rerun()
            else:
                st.error("Failed to clear knowledge base")
        except Exception as e:
            st.error(f"Error: {str(e)}")

    st.divider()
    st.markdown(
        """
        ### About
        This app uses:
        - **FastAPI** for backend
        - **ChromaDB** for vector storage
        - **Groq** for LLM inference
        - **Streamlit** for UI
        """
    )

# Main content Tabs
tab1, tab2 = st.tabs(["📤 Upload Documents", "💬 Ask Questions"])

# Tab 1: Upload Documents
with tab1:
    st.header("Upload Documents")
    st.markdown("Upload PDF or TXT files to add them to the knowledge base")

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload PDF or TXT documents",
        key="uploader"
    )

    if uploaded_files and st.button("Process Documents", type="primary", key="process_docs_btn"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")
            try:
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type or "application/octet-stream")}
                response = requests.post(f"{API_URL}/upload", files=files, timeout=120)

                if response.status_code == 200:
                    result = response.json()
                    st.success(f"✅ {uploaded_file.name}: {result.get('chunks_created', 0)} chunks created")
                else:
                    st.error(f"❌ {uploaded_file.name}: Upload failed ({response.status_code})")
            except Exception as e:
                st.error(f"❌ {uploaded_file.name}: {str(e)}")

            progress_bar.progress((idx + 1) / len(uploaded_files))

        status_text.text("Processing complete!")
        st.toast("🎉 Files processed successfully!")

# Tab 2: Query Interface
with tab2:
    st.header("Ask Questions")
    st.markdown("Query your knowledge base using natural language")

    # ---- Initialize state once
    if "query" not in st.session_state:
        st.session_state.query = ""
    if "top_k" not in st.session_state:
        st.session_state.top_k = 5

    
    with st.form(key="query_form", clear_on_submit=False):
        
        st.text_input(
            "Your question",
            placeholder="e.g., What are the main findings in the research papers?",
            label_visibility="collapsed",
            key="query",
        )

        
        st.number_input(
            "Search Depth (Top-K)",
            min_value=1,
            max_value=10,
            step=1,
            key="top_k",
            help=(
                "How many small text sections (chunks) to retrieve from your documents. "
                "Higher = deeper/wider search but may be slower; lower = faster and more focused."
            ),
        )

        submitted = st.form_submit_button("🔍 Search", type="primary")

    # Handling submit
    if submitted:
        q = st.session_state.query.strip()
        if not q:
            st.warning("Please enter a question")
        else:
            top_k = int(st.session_state.get("top_k", 5))
            with st.spinner("Searching knowledge base..."):
                try:
                    response = requests.post(
                        f"{API_URL}/query",
                        json={"question": q, "top_k": top_k},
                        timeout=60,
                    )
                    if response.status_code == 200:
                        result = response.json()
                        if result.get("status") == "success":
                            st.subheader("💡 Answer")
                            st.markdown(result["answer"])

                            st.subheader("📚 Sources")
                            sources = result.get("sources", []) or []
                            if sources:
                                for source in sources:
                                    st.markdown(f"- {source}")
                            else:
                                st.write("_No sources returned_")

                            with st.expander("🔧 Query Details"):
                                st.write(f"**Chunks Retrieved:** {result.get('context_used', 0)}")
                                st.write(f"**Search Depth (Top-K):** {top_k}")
                        else:
                            st.error(result.get("message", "Query failed"))
                    else:
                        st.error(f"API request failed ({response.status_code})")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    # Example queries
    with st.expander("💡 Example Questions"):
        st.markdown(
            """
            - What are the main topics discussed in the documents?
            - Summarize the key findings from the research
            - What recommendations are mentioned?
            - Compare the different approaches discussed
            - What are the limitations mentioned in the papers?
            """
        )

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <small>Built with FastAPI, ChromaDB, Groq, and Streamlit | RAG-powered Knowledge Base</small>
    </div>
    """,
    unsafe_allow_html=True
)


