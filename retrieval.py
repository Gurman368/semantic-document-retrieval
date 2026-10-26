"""Semantic document retrieval over Simple English Wikipedia.

Usage:
    python retrieval.py "artificial intelligence" --top_k 5
"""
import argparse

import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents(n=2000):
    ds = load_dataset("wikimedia/wikipedia", "20231101.simple", split=f"train[:{n}]")
    return [d["text"] for d in ds]


def cosine_similarity(query_emb, doc_embs):
    """cos(q, d) = q.d / (||q|| ||d||); after L2 normalisation this is a dot product."""
    q = torch.nn.functional.normalize(query_emb, p=2, dim=1)
    d = torch.nn.functional.normalize(doc_embs, p=2, dim=1)
    return torch.mm(q, d.T).squeeze(0)


def retrieve(query, documents, doc_embs, model, top_k=5):
    q = model.encode(query, convert_to_tensor=True).unsqueeze(0)
    scores = cosine_similarity(q, doc_embs)
    top = torch.topk(scores, k=top_k)
    return [
        {"document": documents[i], "score": s.item()}
        for s, i in zip(top.values, top.indices)
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("query")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--n_docs", type=int, default=2000)
    args = p.parse_args()

    documents = load_documents(args.n_docs)
    model = SentenceTransformer(MODEL_NAME)
    doc_embs = model.encode(documents, convert_to_tensor=True, show_progress_bar=True)

    for rank, r in enumerate(retrieve(args.query, documents, doc_embs, model, args.top_k), 1):
        print(f"\n{rank}. score={r['score']:.4f}")
        print(r["document"][:200].replace("\n", " "))


if __name__ == "__main__":
    main()
