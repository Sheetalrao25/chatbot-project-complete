
import os
import shutil
from typing import List

from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings

from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

# === SETUP ===

app = FastAPI()
UPLOAD_FOLDER = "uploaded_pdfs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Serve frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")

# === EMBEDDING SETUP ===

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()

embedding_model = SentenceTransformerEmbeddings()
vectorstore = None  # Global FAISS store
docs_for_all = []   # Holds all chunks for themes


# === UPLOAD ROUTE ===

@app.post("/upload/")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    global vectorstore, docs_for_all
    docs_for_all = []

    try:
        for file in files:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            loader = PyPDFLoader(file_path)
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)
            docs_for_all.extend(texts)

        vectorstore = FAISS.from_documents(docs_for_all, embedding_model)

        # Extract themes from first few chunks
        text_content = " ".join([doc.page_content for doc in docs_for_all[:10]])
        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(text_content, top_n=10)
        themes = [kw[0] for kw in keywords]

        return {"filenames": [file.filename for file in files], "themes": themes}

    except Exception as e:
        return {"error": str(e)}


# === QUERY ROUTE ===

class QueryRequest(BaseModel):
    query: str

# === LOCAL LLM INTEGRATION ===

from transformers import pipeline

# Load a public LLM (flan-t5-base is smaller & free to use)
generator = pipeline("text2text-generation", model="google/flan-t5-base", max_length=256)

def generate_answer_with_local_llm(question: str, context: str) -> str:
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    result = generator(prompt)[0]['generated_text']
    return result.strip()

@app.post("/ask/")
def ask_question(request: QueryRequest):
    global vectorstore

    try:
        if vectorstore is None:
            return {"error": "No documents uploaded yet. Please upload files first."}

        docs = vectorstore.similarity_search(request.query, k=3)
        combined_text = " ".join([doc.page_content for doc in docs])

        # Use local model
        answer = generate_answer_with_local_llm(request.query, combined_text)

        return {
            "query": request.query,
            "answer": answer
        }

    except Exception as e:
        return {"error": str(e)}
