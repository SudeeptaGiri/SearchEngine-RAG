#!/usr/bin/env python3

import argparse
import json
import math
import os
import pickle
import token
from nltk.stem import PorterStemmer

from collections import Counter, defaultdict

BM25_K1 = 1.5
BM25_B = 0.75

class Inverted_index:
    def __init__(self):
        self.index = defaultdict(set) # token : [docid1, docid2)
        self.docmap = {} # docid : docname
        self.term_freq = defaultdict(Counter) # token : frequency across all documents
        self.doc_lengths = defaultdict(int) # docid : length of the document (in terms of number of tokens)
        self.index_path = "cache/index.pkl"
        self.docmap_path = "cache/docmap.pkl"
        self.term_freq_path = "cache/term_freq.pkl"
        self.doc_lengths_path = "cache/doc_lengths.pkl"
    def __add_document(self, doc_id, text):
        '''
        Tokenize the input text, then add each token to the index with the document ID.
        '''
        tokenized_text = stemming(text)
        for token in set(tokenized_text):
            self.index[token].add(doc_id)
            # For each token, increment its count in the Counter for that document ID.
            self.term_freq[token][doc_id] += tokenized_text.count(token)
        # Update the document length for the given document ID
        self.doc_lengths[doc_id] = len(tokenized_text)
    def __get_avg_doc_length(self) -> float:
        total_length = sum(self.doc_lengths.values())
        num_docs = len(self.doc_lengths)
        return total_length / num_docs if num_docs > 0 else 0.0
    def get_documents(self, term):
        '''
        It should get the set of document IDs for a given token, and return them as a list, sorted in ascending order. For our purposes, you can assume that the input term is a single word/token – though you may still want to lowercase it for good measure.
        '''
        tokens = stemming(term)
        if len(tokens) != 1:
            raise ValueError("Input term should be a single token")
        return sorted(list(self.index[tokens[0]]))
    def get_tf(self, doc_id, term):
        tokens = stemming(term)
        if len(tokens) != 1:
            raise ValueError("Input term should be a single token")
        return self.term_freq[tokens[0]][doc_id]
    def get_bm25_idf(self, term: str) -> float:
        term = stemming(term)  # tokenize the term
        if len(term)!=1:
            raise ValueError("Input term should be a single token")
        N = len(self.docmap)  # Total number of documents
        df = len(self.get_documents(term[0]))  # Document frequency of the term
        return math.log((N - df + 0.5) / (df + 0.5) + 1)
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        tf = self.get_tf(doc_id, term)
        avg_length = self.__get_avg_doc_length()
        length_norm = (1 - b) + b * (self.doc_lengths[doc_id] / avg_length)
        saturated_tf = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return saturated_tf
    def bm25(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        idf = self.get_bm25_idf(term)
        tf = self.get_bm25_tf(doc_id, term, k1, b)
        return idf * tf
    def bm25_search(self, query, limit=5):
        tokenizationed_query = stemming(query)
        doc_scores = defaultdict(float)
        for token in tokenizationed_query:
            for doc_id in self.get_documents(token):
                doc_scores[doc_id] += self.bm25(doc_id, token)
        return sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:limit]
    def build(self):
        '''
        It should iterate over all the movies and add them to both the index and the docmap.
            When adding the movie data to the index with __add_document(), concatenate the title and the description and use that as the input text. For example:
                f"{m['title']} {m['description']}"
        '''
        movies = load_movies()
        for movie in movies:
            movie_id = movie['id']
            movie_title = movie['title']
            movie_description = movie['description']

            self.__add_document(movie_id, f"{movie_title} {movie_description}")
            self.docmap[movie_id] = movie
    def save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        with open(self.index_path, 'wb') as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, 'wb') as f:
            pickle.dump(self.docmap, f)
        with open(self.term_freq_path, 'wb') as f:
            pickle.dump(self.term_freq, f)
        with open(self.doc_lengths_path, 'wb') as f:
            pickle.dump(self.doc_lengths, f)
    def load(self):
        '''
        It should load the index, docmap, and term_freq from their respective pickle files. You can assume that these files already exist and are properly formatted.
        '''
        with open(self.index_path, 'rb') as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, 'rb') as f:
            self.docmap = pickle.load(f)
        with open(self.term_freq_path, 'rb') as f:
            self.term_freq = pickle.load(f)
        with open(self.doc_lengths_path, 'rb') as f:
            self.doc_lengths = pickle.load(f)

def load_index()->Inverted_index:
    idx = Inverted_index()
    idx.load()
    return idx
def build_command():
    idx = Inverted_index()
    idx.build()
    idx.save()
    # docs = idx.get_documents("merida")
    # print(f"First document for token 'merida' = {docs[0]}")

def tf_command(doc_id, term):
    idx = load_index()
    return idx.get_tf(doc_id, term)

def idf_command(term):
    idx = load_index()
    total_docs = len(idx.docmap)
    doc_freq = len(idx.get_documents(term))
    idf = math.log((total_docs + 1) / (doc_freq + 1))
    return idf

def tfidf_command(doc_id, term):
    tf = tf_command(doc_id, term)
    idf = idf_command(term)
    tf_idf = tf * idf
    return tf_idf

def bm25_idf_command(term):
    idx = load_index()
    return idx.get_bm25_idf(term)

#optional k1
def bm25_tf_command(doc_id, term, k1=BM25_K1, b=BM25_B):
    idx = load_index()
    return idx.get_bm25_tf(doc_id, term, k1, b)
def load_movies()->list[dict]:
    data =  json.load(open("data/movies.json"))
    return data['movies']

def load_stopwords()->list[str]:
    stopwords=[]
    with open("data/stopwords.txt") as f:
        data = f.read()
        stopwords.extend(data.splitlines())
    return stopwords

def clean_text(text:str)->str:
    text = text.lower()
    # Added underscore to punctuation
    text = text.translate(str.maketrans('', '', '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'))
    return text

def tokenization(text:str)->list[str]:
    text = clean_text(text)
    return text.split()  # Better handling of whitespace

def stemming(text:str)->list[str]:
    stemmer = PorterStemmer()
    tokenized_text = tokenization(text)
    stopwords = load_stopwords()  # Fixed: was "laod_stopwords()"
    without_stopwords = [token for token in tokenized_text if token not in stopwords]
    stemmed = [stemmer.stem(token) for token in without_stopwords]
    return stemmed

def keyword_search(query)->list:
    res = []
    movies = load_movies()
    final_query = stemming(query)
    for movie in movies:
        final_title = stemming(movie['title'])
        for token in final_query:
            if(token in final_title):
                res.append(movie['title'])
                break
        if(len(res) == 5):
            break
    return res

def inverted_index_search(query:str)->dict:
    idx = load_index()
    tokenized_query = tokenization(query)
    doc_ids = set()
    
    for token in tokenized_query:
        doc_ids.update(idx.get_documents(token))
    
    final_res = dict()
    for movie_id in list(doc_ids)[:5]:  # Limit to 5 unique docs
        final_res[movie_id] = idx.docmap[movie_id]["title"]
    
    return final_res

#1. (15) The Adventures of Mowgli - Score: 7.79
# 2. (11342) Gakuen Alice - Score: 7.42
# 3. (30) Day of the Animals - Score: 7.21
# 4. (5542) Candy - Score: 7.07
# 5. (3395) Life of Pi - Score: 7.05
def bm25_search(query:str)->dict:
    idx = load_index()
    results = idx.bm25_search(query)
    final_res = defaultdict(dict)
    for movie_id, score in results:
        final_res[movie_id] = {"score": score, "title": idx.docmap[movie_id]["title"]}
    return final_res


def main() -> None:
    
    
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build the inverted index")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term")

    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency")
    idf_parser.add_argument("term", type=str, help="Term")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF score")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, 
    help="Tunable BM25 B parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print ("Search Query :",args.query)
            # results = keyword_search(args.query)
            results = inverted_index_search(args.query)
            for id,movie in results.items():
                print(f"{id}. {movie}")
        case "build":
            build_command()
        case "tf":
            tf = tf_command(args.doc_id, args.term)
            print(f"Term Frequency for doc_id {args.doc_id} and term '{args.term}': {tf:.2f}")
        case "idf":
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            tf_idf = tfidf_command(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case "bm25idf":
            bm25_idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25_idf:.2f}")
        case "bm25tf":
            bm25_tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}' with k1={args.k1} and b={args.b}: {bm25_tf:.2f}")
        case "bm25search":
            print ("Search Query :",args.query)
            results = bm25_search(args.query)
            sl=1
            for id,data in results.items():
                print(f"{sl}. ({id}) {data['title']} - Score: {data['score']:.2f}")
                sl += 1
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

