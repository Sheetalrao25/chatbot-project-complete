from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from keybert import KeyBERT
from transformers import pipeline

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

# Load model only when called to reduce startup memory
_generator = None

def generate_answer_with_local_llm(question: str, context: str) -> str:
    global _generator
    if _generator is None:
        _generator = pipeline("text2text-generation", model="google/flan-t5-base")
    
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    result = _generator(prompt, max_new_tokens=256)[0]['generated_text']
    return result.strip()
