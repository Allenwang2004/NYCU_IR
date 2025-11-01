# Code Search Report: Sparse vs. Dense Retrieval

## 1. Sparse Retrieval

### 1.1 Preprocessing

We first preprocess all comments and code snippets by splitting them into lexical tokens using a self-defined tokenizer.

**Example tokenization:**
```
"def initialize_bagit(self):" 
→ ["def", "initialize", "bagit", "(", "self", ")", ":", ...]
```

Each code snippet and comment is represented as a **bag-of-words (BoW)** vector, where each token corresponds to a dimension in the vocabulary.

---

### 1.2 TF-IDF

The **TF-IDF** (Term Frequency–Inverse Document Frequency) model weights tokens by their importance within the corpus.

#### Term Frequency (TF)
$$\mathrm{TF}(t, d) = 1 + \log f_{t,d}$$

where $f_{t,d}$ is the frequency of term $t$ in document $d$.

#### Inverse Document Frequency (IDF)
$$\mathrm{IDF}(t) = \log \frac{N}{n_t}$$

where $N$ is the total number of documents, and $n_t$ is the number of documents containing term $t$.

#### TF-IDF Weight
$$w_{t,d} = \mathrm{TF}(t, d) \times \mathrm{IDF}(t)$$

Each document $d$ (or query $q$) is represented by a vector:
$$\mathbf{v}_d = [w_{1,d}, w_{2,d}, \dots, w_{T,d}]$$

#### Cosine Similarity
$$\mathrm{Sim}(q, d) = \frac{\mathbf{v}_q \cdot \mathbf{v}_d}{\|\mathbf{v}_q\| \, \|\mathbf{v}_d\|}$$

---

### 1.3 BM25

**BM25** (Best Match 25) refines TF-IDF by adding term frequency saturation and document length normalization.

$$\mathrm{BM25}(q, d) = \sum_{t \in q} \mathrm{IDF}(t) \cdot \frac{f_{t,d} \cdot (k_1 + 1)}{f_{t,d} + k_1 \cdot (1 - b + b \cdot \frac{|d|}{\text{avgdl}})}$$

**Parameters:**
- $f_{t,d}$: term frequency of token $t$ in document $d$
- $|d|$: document length
- $\text{avgdl}$: average document length
- $k_1 \in [1.2, 2.0]$: saturation parameter
- $b \in [0, 1]$: length normalization weight
- $\mathrm{IDF}(t) = \log\frac{N - n_t + 0.5}{n_t + 0.5}$

---

### 1.4 Similarity Computation & Evaluation

For each query, we compute cosine similarities (TF-IDF) or BM25 scores against all code snippets to form a score matrix $S \in \mathbb{R}^{|Q| \times |C|}$.

We select the top-k results per query and evaluate using **Recall@k**:

$$\mathrm{Recall@k} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \mathbb{I}\big(\text{GT}(q_i) \in \text{Top-k}(q_i)\big)$$

---

### 1.5 Comparison and Analysis

| Method | Recall@10 | Observation |
|--------|-----------|-------------|
| TF-IDF | Lower | Lacks normalization; penalized by long code |
| BM25 | Higher | Handles document length & TF saturation |

**Why BM25 performs better:**
1. Down-weights very frequent tokens (e.g., `def`, `return`)
2. Normalizes document length through parameter $b$
3. Adjusts TF nonlinearly — avoids over-weighting frequent terms

---

## 2. Dense Retrieval

### 2.1 Pre-Training with CodeBERT

We use **Microsoft CodeBERT** (base), a Transformer model trained on paired natural language and code corpora. It learns joint embeddings through:
- **Masked Language Modeling (MLM)**
- **Replaced Token Detection (RTD)**

This creates a shared semantic space between queries and code snippets.

---

### 2.2 Tokenization

The CodeBERT Tokenizer maps tokens to unique IDs with **50,265 entries**.

**Special tokens:**

| Token | Meaning |
|-------|---------|
| `<s>` | Start of sequence |
| `</s>` | End of sequence |
| `<unk>` | Unknown token |
| `<pad>` | Padding |
| `<mask>` | Used in MLM |

Each sequence becomes:
$$\mathbf{x} = [\texttt{<s>}, t_1, t_2, \ldots, t_n, \texttt{</s>}]$$

---

### 2.3 Fine-Tuning for Code Search

We fine-tune CodeBERT using **contrastive learning**, so that matching (query, code) pairs are close, and non-matching ones are far apart.

#### Contrastive (InfoNCE) Loss

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\mathrm{sim}(\mathbf{q}_i, \mathbf{c}_i)/\tau)}{\sum_{j=1}^{N} \exp(\mathrm{sim}(\mathbf{q}_i, \mathbf{c}_j)/\tau)}$$

**Where:**
- $\mathrm{sim}(\mathbf{q}, \mathbf{c}) = \frac{\mathbf{q}\cdot\mathbf{c}}{\|\mathbf{q}\|\|\mathbf{c}\|}$
- $\tau$: temperature parameter (e.g., 0.05)
- $N$: batch size

This loss encourages correct pairs to have the highest similarity in each batch.

---

### 2.4 Dense Retrieval Process

**After fine-tuning:**
1. Encode all code snippets → vectors $\mathbf{C}_1, \dots, \mathbf{C}_n \in \mathbb{R}^D$
2. Encode each query → vector $\mathbf{q} \in \mathbb{R}^D$

**Compute similarity:**
$$\mathrm{Sim}(\mathbf{q}, M) = \mathbf{q} \cdot M^\top$$

where $M \in \mathbb{R}^{n \times D}$ is the matrix of all code embeddings.

Select **Top-k** highest scores for retrieval.

---

### 2.5 Evaluation

$$\mathrm{Recall@10} = \frac{1}{|Q|}\sum_{i=1}^{|Q|} \mathbb{I}\big(\text{GT}(q_i) \in \text{Top-10}(q_i)\big)$$

| Model | Recall@10 | Observation |
|-------|-----------|-------------|
| Pre-trained CodeBERT | Moderate | Captures syntax but lacks task-specific alignment |
| Fine-tuned CodeBERT | **Highest** | Learns semantic alignment via supervision |

**Conclusion:**
Fine-tuning adjusts embedding spaces specifically for the text-to-code retrieval task, bridging the semantic gap between natural language and programming language.

---

## 3. Sparse vs. Dense Retrieval Comparison

| Aspect | Sparse Retrieval | Dense Retrieval |
|--------|------------------|-----------------|
| **Representation** | Term frequency vectors | Contextual embeddings |
| **Similarity** | Lexical overlap (cosine/BM25) | Semantic similarity |
| **Tokenization** | Custom split | Transformer-based |
| **Strength** | Simple, interpretable | Captures deep semantics |
| **Weakness** | Word mismatch, no context | Requires GPU, data |
| **Example** | BM25 | Fine-tuned CodeBERT |

**Key Insight:** Dense retrieval outperforms sparse methods because it captures semantic equivalence (e.g., "return maximum value" ↔ `def get_max(a,b)`).

---

## 4. Retrieve-and-Re-Rank (Hybrid Approach)

To further boost accuracy, we can use a **two-stage pipeline**:

### Stage 1: Retrieve
Use BM25 or dense encoder to recall **top-100** candidates.

### Stage 2: Re-Rank
Use a **cross-encoder** (e.g., CodeBERT-Cross) that jointly encodes (query, code):
$$s(q, c) = \text{Softmax}(W[\text{CLS}])$$

The `[CLS]` embedding represents the joint contextual relevance.

### Stage 3: Final Selection
Sort candidates by $s(q, c)$ → obtain final **Top-k**.

**Benefits:** This Retrieve-then-Re-Rank setup combines the speed of bi-encoder and the precision of cross-encoder, often improving Recall@10 by **3–8%**.

---

## 5. Key Findings

| Comparison | Winner | Reason |
|------------|--------|--------|
| TF-IDF vs. BM25 | **BM25** | Document length normalization & smoother TF scaling |
| Pre-trained vs. Fine-tuned | **Fine-tuned** | Learns semantic alignment |
| Sparse vs. Dense | **Dense** | Contextual understanding beyond lexical overlap |

---

## Summary

**Fine-tuned CodeBERT** achieves the best retrieval performance in the text-to-code task, demonstrating that **contrastive learning + transformer-based embeddings** effectively bridge the gap between human language and code semantics.

The key to success lies in learning task-specific representations that capture both syntactic patterns and semantic intent, rather than relying solely on lexical matching.

---

## Results Overview

| Method | Type | Recall@10 | Computational Cost |
|--------|------|-----------|-------------------|
| TF-IDF | Sparse | Low | Very Low |
| BM25 | Sparse | Medium | Low |
| Pre-trained CodeBERT | Dense | Medium-High | High |
| **Fine-tuned CodeBERT** | **Dense** | **Highest** | **High** |
| Hybrid (BM25 + Re-rank) | Mixed | Very High | Medium |

---

*Would you like me to add LaTeX equation numbering or figure placeholders for academic PDF export?*