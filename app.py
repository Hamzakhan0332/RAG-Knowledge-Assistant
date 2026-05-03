import streamlit as st
import os
import tempfile
from ingest import ingest_document
from rag_pipeline import add_documents_to_index, generate_response_stream, get_chroma_collection

# Page config
st.set_page_config(page_title="Local RAG Assistant", layout="wide")

# Styling
st.markdown("""
    <style>
    .stChatFloatingInputContainer { padding-bottom: 20px; }
    .stStatus { border-radius: 10px; }
    .sidebar-section { padding: 10px; background-color: #f8f9fa; border-radius: 5px; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 Local RAG Knowledge Assistant")

# Cache the database check
@st.cache_resource
def check_db_count():
    collection = get_chroma_collection()
    return collection.count()

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    engine_type = st.radio(
        "Select LLM Engine",
        ["Local (Ollama)", "Cloud (Groq)"],
        help="Local uses your hardware (private but slow). Cloud uses Groq (fast but needs API key)."
    )
    
    api_key = None
    if engine_type == "Cloud (Groq)":
        api_key = st.text_input("Groq API Key", type="password")
        st.caption("[Get your free key here](https://console.groq.com/keys)")

    st.divider()

    st.header("📁 Document Management")
    uploaded_files = st.file_uploader(
        "Upload Documents", 
        type=["pdf", "docx", "txt", "md"], 
        accept_multiple_files=True
    )
    
    if st.button("Process Documents"):
        if uploaded_files:
            all_parsed_docs = []
            with st.status("Reading files...", expanded=True) as status:
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    try:
                        parsed = ingest_document(tmp_path)
                        for p in parsed:
                            p["metadata"]["source"] = uploaded_file.name
                        all_parsed_docs.extend(parsed)
                    except Exception as e:
                        st.error(f"Error: {e}")
                    finally:
                        os.unlink(tmp_path)
                
                if all_parsed_docs:
                    st.write("Creating vector index...")
                    add_documents_to_index(all_parsed_docs, engine_type, api_key)
                    st.cache_resource.clear() 
                    status.update(label="Indexing complete!", state="complete", expanded=False)
                    st.success(f"Indexed {len(uploaded_files)} documents.")
        else:
            st.warning("Please upload files first.")

    if st.button("Clear Database"):
        collection = get_chroma_collection()
        ids = collection.get()["ids"]
        if ids:
            collection.delete(ids=ids)
        st.cache_resource.clear()
        st.success("Database cleared!")
        st.rerun()

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.success("Chat history cleared!")
        st.rerun()

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "citations" in message and message["citations"]:
            with st.expander("Sources"):
                for cite in message["citations"]:
                    st.markdown(f"**Source:** {cite['source']}\n\n{cite['content']}")

if prompt := st.chat_input("Ask a question..."):
    if engine_type == "Cloud (Groq)" and not api_key:
        st.error("Please provide a Groq API Key to use the Cloud engine.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            doc_count = check_db_count()
            if doc_count == 0:
                st.warning("Please upload and process documents first.")
            else:
                try:
                    response_gen, citations = generate_response_stream(prompt, engine_type, api_key)
                    full_response = st.write_stream(response_gen)
                    
                    if citations:
                        with st.expander("Sources"):
                            for cite in citations:
                                st.markdown(f"**Source:** {cite['source']}\n\n{cite['content']}")
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": full_response,
                        "citations": citations
                    })
                except Exception as e:
                    st.error(f"Error: {e}")
                    if "NoneType" in str(e) and engine_type == "Cloud (Groq)":
                        st.info("Tip: This error often means the API key is incorrect or restricted.")
