# chatbot_app/llm.py
from transformers import pipeline

# Load a public LLM (flan-t5-base is smaller & free to use)
generator = pipeline("text2text-generation", model="google/flan-t5-base", max_length=256)

def generate_answer_with_local_llm(question: str, context: str) -> str:
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    result = generator(prompt)[0]['generated_text']
    return result.strip()
