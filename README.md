# simple-reranker
Simple reranker FastAPI + HuggingFace（M4 Mac対応）

## 1) セットアップ

```bash
# 任意の作業フォルダで
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# PyTorchはmacOSならMPS対応版がpipで入ります
pip install "torch>=2.2" fastapi uvicorn[standard] transformers sentence-transformers
```

## 2) 最小サーバ（`main.py`）

```python
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
```

## 3) 起動

```bash
uvicorn main:app --host 127.0.0.1 --port 8080
```

## 4) 動作確認

```bash
curl -X POST http://127.0.0.1:8080/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query":"就業規則の看護休暇の条件",
    "documents":["看護休暇は子の看護を目的として取得できる","年次有給休暇の計算方法","テレワーク規程の対象者"],
    "top_k":2
  }'
```

これで**LM StudioはLLM専用**（OpenAI互換エンドポイント）として使い、**再ランクは上のHTTPサーバ**に投げれば、DifyのWorkflowで「BM25/ベクトル→/rerank→LLM」の流れが組めます。

---

# 速度/精度のコツ（Apple Silicon向け）

* **MPSが有効**なら自動でGPU加速されます（上のコードで`device="mps"`）。初回だけモデルDLで少し時間がかかります。
* もう少し軽くしたい場合は `BAAI/bge-reranker-base` などの**小さめモデル**に切替。
* バッチ処理したい場合は `model.predict(pairs, convert_to_tensor=True, batch_size=32)` などで微調整。

---



