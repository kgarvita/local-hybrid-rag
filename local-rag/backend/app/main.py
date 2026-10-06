from fastapi import FastAPI, HTTPException, Query

from .config import TOP_K
from .generate import generate_answer

app = FastAPI(title="Local Multimodal RAG", version="2.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/ask")
def ask(
    question: str = Query(..., min_length=1),
    top_k: int = Query(TOP_K, ge=1, le=20),
):
    try:
        answer, results = generate_answer(question, top_k=top_k)
        return {"question": question, "answer": answer, "sources": results}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG request failed: {type(exc).__name__}: {exc}") from exc
