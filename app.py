import os
import shutil
from typing import List
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

from services import generate_answer_with_local_llm

# === Setup ===
app = FastAPI()
UPLOAD_FOLDER = "uploaded_pdfs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")

# === Embedding ===
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()

embedding_model = SentenceTransformerEmbeddings()
vectorstore = None
docs_for_all = []

# === Upload Route ===
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

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = splitter.split_documents(documents)
            docs_for_all.extend(texts)

        vectorstore = FAISS.from_documents(docs_for_all, embedding_model)

        # Extract themes
        text_content = " ".join([doc.page_content for doc in docs_for_all[:10]])
        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(text_content, top_n=10)
        themes = [kw[0] for kw in keywords]

        return {"filenames": [file.filename for file in files], "themes": themes}

    except Exception as e:
        return {"error": str(e)}

# === Ask Route ===
class QueryRequest(BaseModel):
    query: str

@app.post("/ask/")
def ask_question(request: QueryRequest):
    global vectorstore
    try:
        if vectorstore is None:
            return {"error": "No documents uploaded yet."}

        docs = vectorstore.similarity_search(request.query, k=3)
        combined_text = " ".join([doc.page_content for doc in docs])

        answer = generate_answer_with_local_llm(request.query, combined_text)

        return {"query": request.query, "answer": answer}

    except Exception as e:
        return {"error": str(e)}
