import os
from llama_index.core import VectorStoreIndex, StorageContext, Document, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.groq import Groq
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import PromptTemplate
import chromadb

def init_settings(engine_type="Local (Ollama)", api_key=None):
    """Initialize settings. Switching between Local and Cloud for speed."""
    
    # Always Local Embedding for Privacy
    Settings.embed_model = HuggingFaceEmbedding(model_name="./all-MiniLM-L6-v2.pt")
    Settings.text_splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

    if engine_type == "Cloud (Groq)" and api_key:
        # Use Groq (Super Fast)
        Settings.llm = Groq(model="llama-3.3-70b-versatile", api_key=api_key)
    else:
        # Local Ollama
        Settings.llm = Ollama(model="qwen2.5:1.5b-instruct", request_timeout=600.0)

def get_chroma_collection():
    """Get persistent ChromaDB collection."""
    db_path = "./chroma_db"
    chroma_client = chromadb.PersistentClient(path=db_path)
    return chroma_client.get_or_create_collection("company_docs")

def add_documents_to_index(parsed_docs, engine_type="Local (Ollama)", api_key=None):
    """Convert parsed text to LlamaIndex Documents and add to ChromaDB."""
    init_settings(engine_type, api_key)
    collection = get_chroma_collection()
    
    documents = [Document(text=doc["text"], metadata=doc["metadata"]) for doc in parsed_docs]
    
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context,
        show_progress=True
    )
    return index

def get_query_engine(engine_type="Local (Ollama)", api_key=None):
    """Get a streaming query engine."""
    init_settings(engine_type, api_key)
    collection = get_chroma_collection()
    
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(vector_store)
    
    template = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, "
        "answer the query.\n"
        "If the answer is not in the context, say 'I don't know.'\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    qa_template = PromptTemplate(template)
    
    return index.as_query_engine(
        text_qa_template=qa_template,
        similarity_top_k=3,
        streaming=True
    )

def generate_response_stream(query_text, engine_type="Local (Ollama)", api_key=None):
    """Optimized RAG Pipeline with Cloud support."""
    query_engine = get_query_engine(engine_type, api_key)
    response = query_engine.query(query_text)
    
    citations = []
    for node in response.source_nodes:
        source = node.metadata.get("source", "Unknown")
        text_snippet = node.get_content()[:200] + "..."
        citations.append({"source": source, "content": text_snippet})
        
    return response.response_gen, citations
