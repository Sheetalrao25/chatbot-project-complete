from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
from app import vectorstore  # Import the shared vectorstore
from services import extract_themes

router = APIRouter()

class ThemeRequest(BaseModel):
    top_n: Optional[int] = 10

@router.post("/theme/")
def get_themes(request: ThemeRequest):
    if vectorstore is None:
        return {"error": "No documents uploaded yet."}

    try:
        # Get the documents stored in the vectorstore
        docs = vectorstore.similarity_search("", k=50)  # Empty query returns top docs by relevance (usually in order)
        themes = extract_themes(docs, top_n=request.top_n)
        return {"themes": themes}

    except Exception as e:
        return {"error": str(e)}
