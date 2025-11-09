import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def mean_pooling(token_embeddings, attention_mask):
    """
    Mean Pooling - Take attention mask into account for correct averaging
    """
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def output_submission_file(results, output_file):
    with open(output_file, 'w') as f:
        f.write("query_id,code_id\n")
        for query_id, code_ids in enumerate(results):
            code_ids_str = ' '.join(map(str, code_ids))
            f.write(f"{query_id+1},{code_ids_str}\n")

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModel.from_pretrained(model_name)

class CodeSearchModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.encoder = base_model

    def forward(self, code_input_ids, code_attention_mask, query_input_ids, query_attention_mask):
        batch_size = code_input_ids.size(0)
        combined_input_ids = torch.cat([code_input_ids, query_input_ids], dim=0)
        combined_attention_mask = torch.cat([code_attention_mask, query_attention_mask], dim=0)
        outputs = self.encoder(input_ids=combined_input_ids, attention_mask=combined_attention_mask)
        
        # Use mean pooling instead of CLS token
        token_embeddings = outputs.last_hidden_state
        pooled_emb = mean_pooling(token_embeddings, combined_attention_mask)
        
        code_emb = F.normalize(pooled_emb[:batch_size], p=2, dim=1)
        query_emb = F.normalize(pooled_emb[batch_size:], p=2, dim=1)
        return code_emb, query_emb
    
class CodeSearchDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len=512):
        df = pd.read_csv(csv_path)
        self.queries = df["query"].tolist()
        self.codes = df["code"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = str(self.queries[idx])
        code = str(self.codes[idx])
        code_enc = self.tokenizer(code, padding='max_length', truncation=True,
                                  max_length=self.max_len, return_tensors="pt")
        query_enc = self.tokenizer(query, padding='max_length', truncation=True,
                                   max_length=self.max_len, return_tensors="pt")
        return {
            "code_input_ids": code_enc["input_ids"].squeeze(0),
            "code_attention_mask": code_enc["attention_mask"].squeeze(0),
            "query_input_ids": query_enc["input_ids"].squeeze(0),
            "query_attention_mask": query_enc["attention_mask"].squeeze(0),
        }
    
def contrastive_loss(code_emb, query_emb, temperature=0.05):
    sim_matrix = torch.matmul(query_emb, code_emb.T) / temperature  # (B, B)
    labels = torch.arange(sim_matrix.size(0)).to(sim_matrix.device)
    return nn.CrossEntropyLoss()(sim_matrix, labels)

def recall_at_k(sim_matrix, k=10):
    N = sim_matrix.size(0)
    _, topk_idx = torch.topk(sim_matrix, k, dim=1)
    correct = sum(i in topk_idx[i] for i in range(N))
    return correct / N

def get_embeddings(text_list, model, tokenizer, batch_size=16, max_len=512):
    model.eval()
    embs = []
    for i in tqdm(range(0, len(text_list), batch_size), desc="Embedding"):
        batch_text = text_list[i:i+batch_size]
        inputs = tokenizer(batch_text, padding=True, truncation=True,
                           max_length=max_len, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.encoder(**inputs)
            # Use mean pooling instead of CLS token
            token_embeddings = outputs.last_hidden_state
            pooled_emb = mean_pooling(token_embeddings, inputs['attention_mask'])
            embs.append(F.normalize(pooled_emb, p=2, dim=1))
    return torch.cat(embs, dim=0)

def train_model(model, tokenizer, train_csv, epochs=10, batch_size=8, lr=2e-5, max_len=512):
    dataset = CodeSearchDataset(train_csv, tokenizer, max_len=max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    best_train_loss = float("inf")
    best_model_path = "best_model.pt"
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            code_input_ids = batch["code_input_ids"].to(device)
            code_attention_mask = batch["code_attention_mask"].to(device)
            query_input_ids = batch["query_input_ids"].to(device)
            query_attention_mask = batch["query_attention_mask"].to(device)

            optimizer.zero_grad()
            code_emb, query_emb = model(
                code_input_ids, code_attention_mask,
                query_input_ids, query_attention_mask
            )
            loss = contrastive_loss(code_emb, query_emb, temperature=0.05)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Avg Train Loss: {avg_loss:.4f}")

        if avg_loss < best_train_loss:
            best_train_loss = avg_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model (train_loss={avg_loss:.4f})")

    print(f"Training complete. Best Train Loss: {best_train_loss:.4f}")
    return best_model_path

train_csv = "data/train_queries.csv"
model = CodeSearchModel(base_model)
best_model_path = train_model(model, tokenizer, train_csv, epochs=20, batch_size=16, lr=2e-5)

print("\nEvaluating Recall@10 for Best Model ...")
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.to(device)
model.eval()

def format_str(string):
    for char in ['\r\n', '\r', '\n']:
        string = string.replace(char, ' ')
    return string.strip()

df = pd.read_csv(train_csv)
codes = [format_str(c)[:512] for c in df["code"]]
queries = [format_str(q)[:512] for q in df["query"]]

query_embs = get_embeddings(queries, model, tokenizer)
code_embs = get_embeddings(codes, model, tokenizer)

sim_matrix = torch.matmul(query_embs, code_embs.T)
recall10 = recall_at_k(sim_matrix, k=10)
print(f"Best Model Recall@10 = {recall10:.4f}")


import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


model_name = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModel.from_pretrained(model_name)

class CodeSearchModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.encoder = base_model

    def forward(self, code_input_ids, code_attention_mask, query_input_ids, query_attention_mask):
        batch_size = code_input_ids.size(0)
        combined_input_ids = torch.cat([code_input_ids, query_input_ids], dim=0)
        combined_attention_mask = torch.cat([code_attention_mask, query_attention_mask], dim=0)
        outputs = self.encoder(input_ids=combined_input_ids, attention_mask=combined_attention_mask)
        
        # Use mean pooling instead of CLS token
        token_embeddings = outputs.last_hidden_state
        pooled_emb = mean_pooling(token_embeddings, combined_attention_mask)
        
        code_emb = F.normalize(pooled_emb[:batch_size], p=2, dim=1)
        query_emb = F.normalize(pooled_emb[batch_size:], p=2, dim=1)
        return code_emb, query_emb

def get_embeddings(text_list, model, tokenizer, batch_size=16, max_len=512):
    model.eval()
    all_embs = []
    for i in tqdm(range(0, len(text_list), batch_size), desc="Embedding"):
        batch_text = text_list[i:i+batch_size]
        inputs = tokenizer(batch_text, padding=True, truncation=True,
                           max_length=max_len, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.encoder(**inputs)
            # Use mean pooling instead of CLS token
            token_embeddings = outputs.last_hidden_state
            pooled_emb = mean_pooling(token_embeddings, inputs['attention_mask'])
            emb = F.normalize(pooled_emb, p=2, dim=1)
            all_embs.append(emb)
    return torch.cat(all_embs, dim=0)

def run_inference(model_path, query_csv, code_csv, output_csv="submission.csv", top_k=10):
    # 載入 fine-tuned 模型
    model = CodeSearchModel(base_model)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    df = pd.read_csv(query_csv)
    queries = [format_str(q)[:512] for q in df["query"]]
    df = pd.read_csv(code_csv)
    codes = [format_str(c)[:512] for c in df["code"]]

    print(f"Loaded {len(queries)} queries and {len(codes)} code snippets")

    query_embs = get_embeddings(queries, model, tokenizer)
    code_embs = get_embeddings(codes, model, tokenizer)

    print("Calculating similarity matrix ...")
    sim_matrix = torch.matmul(query_embs, code_embs.T)

    _, topk_indices = torch.topk(sim_matrix, k=top_k, dim=1)
    results = topk_indices.cpu().numpy().tolist()

    results = [[idx + 1 for idx in code_ids] for code_ids in results]

    output_submission_file(results, output_csv)
    print(f"Saved results to {output_csv}")

model_path = "best_model.pt"       
query_csv = "data/test_queries.csv"          
code_csv = "data/code_snippets.csv"
output_csv = "results/fine_tuned_submission.csv"
run_inference(model_path, query_csv, code_csv, output_csv, top_k=10)