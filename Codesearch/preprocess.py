#!/usr/bin/env python3
"""
Preprocessing pipeline for code search like the figure (split comments vs. code):
- Extract comments (including header/NatSpec/docstrings and inline comments)
- Extract code with comments removed
- Tokenize comments as natural language words
- Tokenize code while preserving punctuation/operators as separate tokens
- Optional: split identifiers (snake_case, camelCase)
- Optional: lowercase, normalize numbers/strings
- Produce columns: comment_tokens, code_tokens, all_tokens

Usage
-----
python preprocess.py

Input CSVs must contain columns:
- codes file: `code` (str)
- queries file: `query` (str)
"""
from __future__ import annotations
import html
import json
import re
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd

# -----------------------------
# Utilities
# -----------------------------
_whitespace_re = re.compile(r"\s+", re.MULTILINE)
_word_re = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Punctuation/delimiters typical for C/Java/JS/Solidity/Python
# We'll split code tokens by placing spaces around these, then split.
_separators = r"[\(\)\[\]\{\};:,\.\+\-\*/%&\|\^!=<>\?~]"
_sep_re = re.compile(f"({_separators})")

# String literal patterns ("...", '...', `...`)
_str_re = re.compile(r'(""".*?"""|\'\'\'.*?\'\'\'|".*?"|\'.*?\'|`.*?`)', re.DOTALL)
_num_re = re.compile(r"\b\d+(?:_\d+)*(?:\.[0-9_]+)?\b")

# Comments (C-like): // line, /* block */
_c_line_cmt = re.compile(r"//.*?(?=\n|$)")
_c_block_cmt = re.compile(r"/\*.*?\*/", re.DOTALL)
# Python: # line, triple-quoted docstrings (we'll treat as comments)
_py_line_cmt = re.compile(r"#.*?(?=\n|$)")
_py_doc_cmt = re.compile(r'(""".*?"""|\'\'\'.*?\'\'\')', re.DOTALL)

_c_like_file_ext = {".c", ".h", ".cpp", ".hpp", ".cc", ".java", ".js", ".ts", ".sol", ".cs", ".swift", ".go"}
_py_like_file_ext = {".py"}

@dataclass
class PreprocessConfig:
    lowercase: bool = True
    normalize_numbers: bool = True   # replace numbers with <NUM>
    normalize_strings: bool = True   # replace strings with <STR>
    split_identifiers: bool = True
    keep_empty: bool = False


def normalize_whitespace(text: str) -> str:
    return _whitespace_re.sub(" ", text).strip()


def split_identifiers(token: str) -> List[str]:
    """Split camelCase/mixedCase and snake_case identifiers.
    e.g. `allowedAmount_total` -> ["allowed", "Amount", "total"] -> lowercased later
    """
    # First split snake_case
    parts = re.split(r"_+", token)
    out: List[str] = []
    for p in parts:
        # Split on camelCase transitions (including digits)
        # e.g. HTTPServerError -> HTTP, Server, Error
        for m in re.finditer(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", p):
            out.append(m.group(0))
    return [t for t in out if t]


def extract_comments_and_code(text: str, language_hint: str = "auto") -> Tuple[str, str]:
    """Return (comments_text, code_without_comments).
    Supports a broad regex-based pass for C-like and Python-like languages.
    """
    s = text
    comments: List[str] = []

    # Python-like docstrings first to avoid gobbling by string regex
    for m in _py_doc_cmt.finditer(s):
        comments.append(m.group(0))
    s = _py_doc_cmt.sub("\n", s)

    # C-like block comments
    for m in _c_block_cmt.finditer(s):
        comments.append(m.group(0))
    s = _c_block_cmt.sub("\n", s)

    # Line comments
    for m in _c_line_cmt.finditer(s):
        comments.append(m.group(0))
    s = _c_line_cmt.sub("\n", s)

    for m in _py_line_cmt.finditer(s):
        comments.append(m.group(0))
    s = _py_line_cmt.sub("\n", s)

    comments_text = "\n".join(comments)
    code_wo_comments = s
    return comments_text, code_wo_comments


def tokenize_comment_text(text: str, cfg: PreprocessConfig) -> List[str]:
    text = html.unescape(text)
    if cfg.normalize_strings:
        text = _str_re.sub(" <STR> ", text)
    if cfg.normalize_numbers:
        text = _num_re.sub(" <NUM> ", text)
    # Words only for comments; punctuation is less useful
    toks = _word_re.findall(text)
    if cfg.split_identifiers:
        expanded: List[str] = []
        for tok in toks:
            expanded.extend(split_identifiers(tok))
        toks = expanded
    if cfg.lowercase:
        toks = [t.lower() for t in toks]
    return toks if toks or cfg.keep_empty else (toks or [])


def tokenize_code(text: str, cfg: PreprocessConfig) -> List[str]:
    text = html.unescape(text)
    # Replace string and numbers with placeholders to reduce noise
    if cfg.normalize_strings:
        text = _str_re.sub(" <STR> ", text)
    if cfg.normalize_numbers:
        text = _num_re.sub(" <NUM> ", text)
    # Put spaces around separators so we can split and KEEP them
    text = _sep_re.sub(r" \1 ", text)
    # Collapse whitespace
    text = normalize_whitespace(text)
    raw_toks = text.split(" ") if text else []

    toks: List[str] = []
    for tok in raw_toks:
        if not tok:
            continue
        if tok == "<STR>" or tok == "<NUM>" or _sep_re.fullmatch(tok):
            toks.append(tok)
        else:
            if cfg.split_identifiers:
                toks.extend(split_identifiers(tok))
            else:
                toks.append(tok)
    if cfg.lowercase:
        toks = [t.lower() for t in toks]
    return toks if toks or cfg.keep_empty else (toks or [])


def preprocess_code_snippet(snippet: str, cfg: PreprocessConfig) -> Tuple[List[str], List[str]]:
    """Given raw code snippet, return (comment_tokens, code_tokens)."""
    comments_text, code_wo = extract_comments_and_code(snippet)
    cmt_toks = tokenize_comment_text(comments_text, cfg)
    code_toks = tokenize_code(code_wo, cfg)
    return cmt_toks, code_toks


def preprocess_query(q: str, cfg: PreprocessConfig) -> List[str]:
    # For queries, treat like comment text (natural language)
    return tokenize_comment_text(q, cfg)


# -----------------------------
# CLI helpers
# -----------------------------

def run_codes(codes_csv: str, out_csv: str, cfg: PreprocessConfig) -> None:
    df = pd.read_csv(codes_csv, engine="python")
    if "code" not in df.columns:
        raise ValueError("codes CSV must contain a 'code' column")
    comment_tokens_list: List[List[str]] = []
    code_tokens_list: List[List[str]] = []
    all_tokens_list: List[List[str]] = []

    for s in df["code"].astype(str).tolist():
        cmt_toks, code_toks = preprocess_code_snippet(s, cfg)
        comment_tokens_list.append(cmt_toks)
        code_tokens_list.append(code_toks)
        all_tokens_list.append(cmt_toks + code_toks)

    df["comment_tokens"] = [json.dumps(x, ensure_ascii=False) for x in comment_tokens_list]
    df["code_tokens"] = [json.dumps(x, ensure_ascii=False) for x in code_tokens_list]
    df["all_tokens"] = [json.dumps(x, ensure_ascii=False) for x in all_tokens_list]
    df.to_csv(out_csv, index=False)


def run_queries(queries_csv: str, out_csv: str, cfg: PreprocessConfig) -> None:
    df = pd.read_csv(queries_csv, engine="python")
    if "query" not in df.columns:
        raise ValueError("queries CSV must contain a 'query' column")
    toks_list: List[List[str]] = []
    for q in df["query"].astype(str).tolist():
        toks_list.append(preprocess_query(q, cfg))
    df["query_tokens"] = [json.dumps(x, ensure_ascii=False) for x in toks_list]
    df.to_csv(out_csv, index=False)


def main():
    # 直接設定檔案路徑和配置
    codes_csv = "data/code_snippets.csv"
    queries_csv = "data/test_queries.csv"
    out_codes = "data/code_snippets.proc.csv"
    out_queries = "data/test_queries.proc.csv"
    
    # 預設配置
    cfg = PreprocessConfig(
        lowercase=True,
        split_identifiers=True,
        normalize_numbers=True,
        normalize_strings=True,
    )
    
    # 處理程式碼檔案
    print("Processing code snippets...")
    run_codes(codes_csv, out_codes, cfg)
    print(f"Code snippets processed and saved to {out_codes}")
    
    # 處理查詢檔案
    print("Processing queries...")
    run_queries(queries_csv, out_queries, cfg)
    print(f"Queries processed and saved to {out_queries}")


if __name__ == "__main__":
    main()
