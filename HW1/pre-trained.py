from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from tqdm import tqdm
import ast
import numpy as np

model_name = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

out_codes_csv = "data/code_snippets_proc.csv"
out_queries_csv = "data/test_queries_proc.csv"

def tokens_to_ids_batch(batch_tokens, max_len, pad_token_id):
    input_ids = []
    attention_masks = []

    for tokens in batch_tokens:
        ids = tokenizer.convert_tokens_to_ids(tokens)

        ids = [tokenizer.cls_token_id] + ids + [tokenizer.sep_token_id]

        ids = ids[:max_len]
        attn_mask = [1] * len(ids)
        while len(ids) < max_len:
            ids.append(pad_token_id)
            attn_mask.append(0)

        input_ids.append(ids)
        attention_masks.append(attn_mask)

    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_masks)
    }

def encode_tokens_list(tokens_list, max_len=256, batch_size=32):
    embeddings = []
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id 
    for i in tqdm(range(0, len(tokens_list), batch_size)):
        batch_tokens = tokens_list[i:i+batch_size]
        encoded_input = tokens_to_ids_batch(batch_tokens, max_len, pad_token_id)
        print(encoded_input)
        with torch.no_grad():
            model_output = model(**encoded_input)
        batch_embeddings = model_output.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

code_snippets = pd.read_csv(out_codes_csv, engine="python")['code_tokens'].apply(ast.literal_eval).tolist()
queries = pd.read_csv(out_queries_csv, engine="python")['query_tokens'].apply(ast.literal_eval).tolist()

code_embeds = encode_tokens_list(code_snippets, max_len=256)
query_embeds = encode_tokens_list(queries, max_len=128)

code_embeds = normalize(code_embeds)
query_embeds = normalize(query_embeds)

similarity_matrix = cosine_similarity(query_embeds, code_embeds)