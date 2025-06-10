# 📄 Document Research & Theme Identification Chatbot

An interactive chatbot to research across PDFs, extract answers using a local LLM (Flan-T5), and identify common themes using semantic embeddings.

## 🔧 Features
- Upload one or more PDF documents
- Extract text, chunk it, and embed using `Sentence-BERT`
- Store embeddings in FAISS for semantic search
- Ask questions in natural language
- Answer generated using local HuggingFace `flan-t5-base` (no OpenAI API needed)
- Extract top themes using KeyBERT
- Simple frontend (HTML + JS) served via FastAPI

## 🚀 How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
