from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from tqdm import tqdm
import ast
import numpy as np
import torch.nn.functional as F

model_name = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


out_queries_csv = 'results/pre_trained_queries.csv'
out_codes_csv = 'results/pre_trained_codes.csv'

def output_submission_file(results, output_file):
    with open(output_file, 'w') as f:
        f.write("query_id,code_id\n")
        for query_id, code_ids in enumerate(results):
            code_ids_str = ' '.join(map(str, code_ids))
            f.write(f"{query_id+1},{code_ids_str}\n")

def format_str(string):
    for char in ['\r\n', '\r', '\n']:
        string = string.replace(char, ' ')
    return string.strip()


code_snippets = pd.read_csv(out_codes_csv, engine="python")['code']
queries = pd.read_csv(out_queries_csv, engine="python")['query']

codes = [format_str(c)[:512] for c in code_snippets]
queries = [format_str(q)[:512] for q in queries]

def get_embeddings(text_list, batch_size=16):
    all_embeddings = []
    for i in tqdm(range(0, len(text_list), batch_size)):
        batch_texts = text_list[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]   # 取 [CLS] 向量
            norm_embeddings = F.normalize(cls_embeddings, p=2, dim=1)
            all_embeddings.append(norm_embeddings)
    return torch.cat(all_embeddings, dim=0)

query_embs = get_embeddings(queries)
code_embs  = get_embeddings(codes)

sim_matrix = torch.matmul(query_embs, code_embs.T)

top_k = 10
results = []

for i in range(len(queries)):
    top_indices = torch.topk(sim_matrix[i], k=top_k).indices
    results.append(top_indices.cpu().numpy().tolist())
    
output_submission_file(results, 'results/pre_trained_submission.csv')