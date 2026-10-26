# Semantic Document Retrieval

A small semantic search engine over Simple English Wikipedia. Documents and queries are embedded with `all-MiniLM-L6-v2` and ranked by cosine similarity, so results match on meaning rather than exact keywords.

## How it works

1. **Data:** the first 2,000 articles of `wikimedia/wikipedia` (`20231101.simple`).
2. **Embeddings:** each article is encoded into a 384-dimensional vector with `sentence-transformers/all-MiniLM-L6-v2`.
3. **Similarity:** for a query vector `q` and document vector `d`,

   cos(q, d) = (q · d) / (‖q‖ ‖d‖)

   Both sides are L2-normalised, so the score is a single matrix product `Q Dᵀ`. A score near 1 means the vectors point the same way.
4. **Retrieval:** `retrieve()` encodes the query, scores all 2,000 documents, and returns the top `k` (default 5) with their scores.

## Run it

```bash
pip install -r requirements.txt
python retrieval.py "artificial intelligence" --top_k 5
```

Or open `document_retrieval.ipynb`, which has the step-by-step version with outputs.

## Example results (from the notebook run)

| Query | Top result | Score |
|---|---|---|
| artificial intelligence | Artificial intelligence (AI) article | 0.50 |
| world war history | "A World War is a war in which most of the countries..." | 0.54 |
| machine learning | A search engine article (no ML article in the subset) | 0.23 |

Scores fall steadily from rank 1 to rank 5, as expected. The "machine learning" query shows a limitation: when the 2,000-article subset has no matching article, the best scores stay low (about 0.23) and the results are only loosely related. A minimum-score cutoff would let the system say "no good match" instead.

## Limitations

- The 2,000-article subset is tiny compared to real search corpora, so some topics have no relevant article.
- `all-MiniLM-L6-v2` truncates its input at 256 word pieces, so only the start of each article is embedded.
- The model is not fine-tuned for this corpus, and there is no re-ranking step.
- Pure semantic search: no keyword matching and no deeper understanding of user intent.

## Possible extensions

- Hybrid retrieval (BM25 + dense) and a cross-encoder re-ranker.
- Chunk long articles so the whole text is searchable.
- Evaluate with labelled queries (recall@k, MRR).
