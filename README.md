# RAG Search Engine

A learning project implementing keyword-based and semantic search capabilities to understand information retrieval fundamentals.

## 🎯 Current Status (10% Complete)

**Implemented:**

- Inverted index with BM25 ranking
- Text preprocessing (tokenization, stemming, stopwords)
- Semantic search foundation with sentence transformers
- Vector similarity operations (cosine similarity)

## 🏗️ Project Structure

```
rag-search-engine/
├── cli/
│   ├── keyword_search_cli.py        # BM25 inverted index
│   └── lib/semantic_search_cli.py   # Semantic search utilities
├── data/                             # Datasets and stopwords
└── cache/                            # Serialized indices
```

## 🚀 Quick Start

```bash
git clone https://github.com/SudeeptaGiri/SearchEngine-RAG.git
cd rag-search-engine
uv sync  # or pip install -r requirements.txt
```

**Dependencies:** `nltk==3.9.1`, `sentence-transformers>=5.2.2`

## 📚 Key Concepts

**Inverted Index**: Maps tokens to documents for fast retrieval  
**BM25**: Probabilistic ranking (k1=1.5, b=0.75) using TF-IDF  
**Text Pipeline**: Tokenize → Lowercase → Remove stopwords → Stem  
**Semantic Search**: Vector embeddings (384-dim) with cosine similarity

## 🎓 Next Steps

- [ ] Hybrid search combining keyword + semantic
- [ ] Query expansion and relevance feedback
- [ ] Web interface
- [ ] Performance benchmarking

---

_Educational project - Last updated: February 18, 2026_
