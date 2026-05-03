# 🧠 Local RAG Knowledge Assistant

Hey there! This is a simple but powerful RAG (Retrieval-Augmented Generation) assistant that I built to help chat with documents locally. Whether you're a privacy enthusiast or just tired of slow local inference, this tool gives you the best of both worlds.

### Why use this?
*   **Privacy First:** Keep your sensitive PDFs and docs on your machine. No data leaks.
*   **Hybrid Speed:** Choose between your local hardware (Ollama) or lightning-fast cloud inference (Groq).
*   **Smart Retrieval:** Uses ChromaDB and LlamaIndex to find exactly what you're looking for with source citations.
*   **Zero Setup Pain:** Fast document parsing with `pypdf`—no heavy OCR models required.

---

## 🛠️ How to get it running

### 1. The Basics
Make sure you have **Python 3.10+** and **Ollama** installed on your system.

### 2. Setup your Environment
```bash
# Clone the repo (or just download the files)
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Get the Brains
*   **For Local Mode:** Pull the model: `ollama pull qwen2.5:1.5b-instruct`
*   **For Cloud Mode:** Grab a free API key from [Groq Console](https://console.groq.com/keys).

---

## 🚀 Launching the App

```bash
streamlit run app.py
```

### Pro Tips:
1.  **First Run:** The first time you ask a question, it might take a few seconds to load the embedding model. Be patient—it's only for the first time!
2.  **Streaming:** I've enabled streaming so you can see the answer being typed out in real-time.
3.  **Clear Options:** Use the "Clear Chat" to reset the vibe or "Clear Database" to swap projects entirely.

---

## 🤝 Contributing
Feel free to fork this, open issues, or submit PRs. This was built to be a clean, hackable starting point for anyone exploring local AI.

*Happy Chatting!*
