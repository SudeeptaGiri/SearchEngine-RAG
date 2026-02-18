#!/usr/bin/env python3

import argparse
from email.mime import text
from pydoc import doc
from sentence_transformers import SentenceTransformer
import numpy as np
import os

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}
    
    def generate_embedding(self, text:str):
        if(text =="" or text.strip() == ""):
            raise ValueError("Input text cannot be empty")
        embedding = self.model.encode(list(text)).tolist()
        return embedding[0]
    def build_embeddings(self, documents):
        self.documents = documents
        documents_stringify = []
        for doc in documents:
            self.document_map[doc['id']] = doc
            documents_stringify.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(documents_stringify, show_progress_bar=True).tolist()
        np.save("cache/movie_embeddings.npy", self.embeddings)
        return self.embeddings
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        if(os.path.exists("cache/movie_embeddings.npy")):
            self.embeddings = np.load("cache/movie_embeddings.npy", allow_pickle=True).tolist()
            if(len(self.embeddings) == len(documents)):
                # Rebuild document_map when loading from cache
                for doc in documents:
                    self.document_map[doc['id']] = doc
                return self.embeddings
        return self.build_embeddings(documents)
    
    def search(self, query, limit):
        if(self.embeddings is None):
            raise ValueError("Embeddings not built. Call load_or_create_embeddings() first.")
        query_embedding = self.generate_embedding(query)
        cosine_similarity_scores = [cosine_similarity(query_embedding, doc_embedding) for doc_embedding in self.embeddings]
        similarity_list = list(zip(self.documents, cosine_similarity_scores))
        sorted_similarity = sorted(similarity_list, key=lambda x: x[1], reverse=True)
        return sorted_similarity[:limit]
#Models (General Purpose Models) ->  all-MiniLM-L6-v2, all-mpnet-base-v2 
#Models (Domain-Specific Models) ->  llenai-specter, microsoft/BiomedNLP-PubMedBERT 
#Models (Multilingual Models) ->  paraphrase-multilingual-MiniLM-L12-v2 

def verify_model():
    MODEL = SemanticSearch().model
    MAX_LENGTH = MODEL.max_seq_length
    print(f"Model loaded: {MODEL}, where {MODEL} is the .model")
    print(f"Max sequence length: {MAX_LENGTH}, where {MAX_LENGTH} is the .max_seq_length property of the model")

def embed_text(text):
    if(text =="" or text.strip() == ""):
        raise ValueError("Input text cannot be empty")
    semanticSearch = SemanticSearch()
    embedding = semanticSearch.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {len(embedding)}")

def load_movies():
    import json
    with open("data/movies.json", "r") as f:
        data = json.load(f)
    return data["movies"]
def verify_embeddings():
    movies = load_movies()
    semanticSearch = SemanticSearch()
    embeddings = semanticSearch.load_or_create_embeddings(movies)
    if(len(embeddings) != len(movies)):
        raise ValueError("Number of embeddings must match number of documents")
    print(f"Number of docs:   {len(movies)}")
    print(f"Embeddings shape: {len(embeddings)} vectors in {len(embeddings[0])} dimensions")


def embed_query_text(query):
    semanticSearch = SemanticSearch()
    embedding = semanticSearch.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {len(embedding)}")

def chunk_text(text, chunk_size, overlap=0):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks
    
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)



def search(query, limit):
    movies = load_movies()
    semanticSearch = SemanticSearch()
    semanticSearch.load_or_create_embeddings(movies)
    results = semanticSearch.search(query, limit)
    for idx, (movie, score) in enumerate(results):
        print(f"{idx+1}. {movie['title']} (score: {score:.4f})")
        print(f"   {movie['description'][:100]}...\n")
def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_model_parser = subparsers.add_parser("verify_model", help="Verify the semantic search model")

    embed_text_parser = subparsers.add_parser("embed_text", help="Generate embedding for a given text")
    embed_text_parser.add_argument("text", type=str, help="Text to generate embedding for")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verify the semantic search embeddings")
   
    embed_query_text_parser = subparsers.add_parser("embedquery", help="Generate embedding for a given query text")
    embed_query_text_parser.add_argument("query", type=str, help="Query text to generate embedding for")

    search_parser = subparsers.add_parser("search", help="Search for movies based on a query")
    search_parser.add_argument("query", type=str, help="Query text to search for")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk a given text into smaller pieces")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size",default=200, type=int, help="Chunk size")
    chunk_parser.add_argument("--overlap", default=0, type=int, help="Overlap size between chunks")

    args = parser.parse_args()
    match args.command:
        case "verify_model":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            search(args.query, args.limit)
        case "chunk":
            chunks=chunk_text(args.text, args.chunk_size,args.overlap)
            for idx, chunk in enumerate(chunks):
                print(f"Chunk {idx+1}: {chunk}\n")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()