# RAG-AI-Tutor-Alex-the-Tutor-

# 🧠 Python Tutor (RAG + Streamlit)

An AI-powered Python tutor that answers your questions with real examples using **Retrieval-Augmented Generation (RAG)**.


## 🚀 Features

- 💬 Chat-style interface with memory
- 📚 Context-aware answers from your own dataset
- 💡 Real Python code examples
- ⚙️ Built with Streamlit, LangChain, ChromaDB, and Ollama


## 🧰 Tech Stack

- **LLM:** `llama3.2` via Ollama
- **Embeddings:** `mxbai-embed-large`
- **Vector DB:** Chroma
- **Framework:** LangChain
- **UI:** Streamlit


## 📦 Setup

```bash
pip install -r requirements.txt
ollama pull llama3
ollama pull mxbai-embed-large
streamlit run app.py
