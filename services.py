import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from keybert import KeyBERT

def load_and_split_documents(file_paths, chunk_size=1000, chunk_overlap=200):
    all_texts = []
    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = splitter.split_documents(documents)
        all_texts.extend(texts)
    return all_texts

def build_vectorstore(texts, embedding_model):
    return FAISS.from_documents(texts, embedding_model)

def extract_themes(text_chunks, top_n=10):
    content = " ".join([doc.page_content for doc in text_chunks])
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(content, top_n=top_n)
    return [kw[0] for kw in keywords]

from paddleocr import PaddleOCR
import fitz  # PyMuPDF
import os

ocr_model = PaddleOCR(use_angle_cls=True, lang='en')

def extract_text_with_ocr(pdf_path):
    text = ""
    pdf_doc = fitz.open(pdf_path)
    for page_index in range(len(pdf_doc)):
        page = pdf_doc[page_index]
        pix = page.get_pixmap()
        image_path = f"temp_page_{page_index}.png"
        pix.save(image_path)

        result = ocr_model.ocr(image_path, cls=True)
        for line in result[0]:
            text += line[1][0] + " "
        os.remove(image_path)
    return text
from transformers import pipeline

# Load once at startup
qa_pipeline = pipeline("text2text-generation", model="mistralai/Mistral-7B-Instruct-v0.1")

def generate_answer_with_local_llm(query, context):
    prompt = f"""Answer the following question based on the context below. Cite relevant paragraphs if useful.

Query: {query}

Context:
{context}

Answer:"""

    result = qa_pipeline(prompt, max_new_tokens=300)[0]['generated_text']
    return result

