from fastapi import FastAPI
from pydantic import BaseModel
import torch
from sentence_transformers import CrossEncoder

MODEL_ID = "BAAI/bge-reranker-v2-m3"  # 日本語OKの多言語CE
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = CrossEncoder(MODEL_ID, device=device)

app = FastAPI()

class RerankRequest(BaseModel):
    query: str
    documents: list[str]
    top_k: int = 6

@app.post("/rerank")
def rerank(req: RerankRequest):
    pairs = [(req.query, d) for d in req.documents]
    scores = model.predict(pairs).tolist()
    ranked = sorted(zip(req.documents, scores), key=lambda x: x[1], reverse=True)[:req.top_k]
    return {"results": [{"text": d, "score": s} for d, s in ranked]}
