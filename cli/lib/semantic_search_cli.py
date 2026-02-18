#!/usr/bin/env python3

import argparse
from email.mime import text
from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        

#Models (General Purpose Models) ->  all-MiniLM-L6-v2, all-mpnet-base-v2 
#Models (Domain-Specific Models) ->  llenai-specter, microsoft/BiomedNLP-PubMedBERT 
#Models (Multilingual Models) ->  paraphrase-multilingual-MiniLM-L12-v2 

def verify_model():
    MODEL = SemanticSearch().model
    MAX_LENGTH = MODEL.max_seq_length
    print(f"Model loaded: {MODEL}, where {MODEL} is the .model")
    print(f"Max sequence length: {MAX_LENGTH}, where {MAX_LENGTH} is the .max_seq_length property of the model")

def add_vector(v1,v2):
    if(len(v1) != len(v2)):
        raise ValueError("Vectors must be of the same length")
    return [a + b for a, b in zip(v1, v2)]
def substract_vector(v1,v2):
    if(len(v1) != len(v2)):
        raise ValueError("Vectors must be of the same length")
    return [a - b for a, b in zip(v1, v2)]
def dot_product(v1,v2):
    if(len(v1) != len(v2)):
        raise ValueError("Vectors must be of the same length")
    return sum(a * b for a, b in zip(v1, v2))

def euclidean_norm(vec):
    return sum(x ** 2 for x in vec) ** 0.5
def cosine_similarity(vec1, vec2):
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must be of the same length")
    if len(vec1) == 0 or len(vec2) == 0:
        return 0
    
    dot_prod = dot_product(vec1, vec2)
    magnitude_vec1 = euclidean_norm(vec1)
    magnitude_vec2 = euclidean_norm(vec2)
    
    if magnitude_vec1 == 0 or magnitude_vec2 == 0:
        return 0.0
    
    return dot_prod / (magnitude_vec1 * magnitude_vec2)

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_model_parser = subparsers.add_parser("verify_model", help="Verify the semantic search model")
    
    args = parser.parse_args()
    match args.command:
        case "verify_model":
            verify_model()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()