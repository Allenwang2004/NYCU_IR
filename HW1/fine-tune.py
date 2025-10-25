# Use pretrained models (CodeBERT) to find the top 10 similar code snippet for each query
import os
import torch
from transformers import RobertaTokenizer, RobertaModel
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import argparse

def load_model(model_name):
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name)
    return tokenizer, model

def get_embeddings(tokenizer, model, texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

def find_similar_snippets(query_embeddings, code_embeddings, top_k=10):
    similarities = cosine_similarity(query_embeddings, code_embeddings)
    top_k_indices = np.argsort(-similarities, axis=1)[:, :top_k]
    return top_k_indices, similarities

def clean_code(code):
    code = code.strip()
    lines = [line.rstrip() for line in code.splitlines() if line.strip() != ""]
    return "\n".join(lines)

def main(args):

    # Load code snippets
    code_snippets_df = pd.read_csv(args.code_snippets_file,engine='python')
    code_snippets = code_snippets_df['code'].tolist()
    code_snippets = [clean_code(code) for code in code_snippets]
    print(f"Loaded {len(code_snippets)} code snippets.")

    # Load queries
    queries_df = pd.read_csv(args.queries_file)
    queries = queries_df['query'].tolist()
    print(f"Loaded {len(queries)} queries.")

    # Load model and tokenizer
    tokenizer, model = load_model(args.model_name)

    # Get embeddings
    code_embeddings = get_embeddings(tokenizer, model, code_snippets).numpy()
    query_embeddings = get_embeddings(tokenizer, model, queries).numpy()

    # Find similar snippets
    top_k_indices, similarities = find_similar_snippets(query_embeddings, code_embeddings, top_k=args.top_k)

    # Output results
    for i, query in enumerate(queries):
        print(f"Query: {query}")
        print("Top similar code snippets:")
        for idx in enumerate(top_k_indices[i]):
            print(f"Code Snippet: {code_snippets[idx]}")
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find similar code snippets using CodeBERT")
    parser.add_argument("--model_name", type=str, default="microsoft/codebert-base", help="Pretrained model name")
    parser.add_argument("--code_snippets_file", type=str, default="data/code_snippets.csv", help="Path to the file containing code snippets")
    parser.add_argument("--queries_file", type=str, default="data/test_queries.csv", help="Path to the file containing queries")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top similar snippets to retrieve")
    args = parser.parse_args()

    main(args)