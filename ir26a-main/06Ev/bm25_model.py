import math
import re
from collections import Counter

import pandas as pd


def tokenize(text):
    """Tokenize text into lowercase word tokens."""
    return re.findall(r"\w+", text.lower())


def build_bm25_index(documents, k1=1.5, b=0.75):
    """Build BM25 index information from a list of documents."""
    tokenized_docs = [tokenize(doc) for doc in documents]
    doc_lengths = [len(tokens) for tokens in tokenized_docs]
    avg_doc_length = sum(doc_lengths) / len(doc_lengths) if documents else 0.0
    N = len(documents)

    term_freqs = [Counter(tokens) for tokens in tokenized_docs]
    df_counts = Counter()
    for tokens in tokenized_docs:
        df_counts.update(set(tokens))

    idf_bm25 = {
        term: math.log((N - df + 0.5) / (df + 0.5) + 1)
        for term, df in df_counts.items()
    }

    return {
        'tokenized_docs': tokenized_docs,
        'doc_lengths': doc_lengths,
        'avg_doc_length': avg_doc_length,
        'term_freqs': term_freqs,
        'idf_bm25': idf_bm25,
        'k1': k1,
        'b': b,
        'N': N,
    }


def bm25_score_doc(query_terms, doc_idx, index):
    """Score a single document for a tokenized query using BM25."""
    score = 0.0
    freqs = index['term_freqs'][doc_idx]
    doc_length = index['doc_lengths'][doc_idx]
    denom_factor = index['k1'] * (1 - index['b'] + index['b'] * doc_length / index['avg_doc_length'])

    for term in query_terms:
        if term not in index['idf_bm25']:
            continue
        tf = freqs.get(term, 0)
        if tf == 0:
            continue
        score += index['idf_bm25'][term] * ((tf * (index['k1'] + 1)) / (tf + denom_factor))

    return score


def bm25_rank(query, documents, index, top_k=None):
    """Rank documents for a single query using BM25."""
    query_terms = tokenize(query)
    scores = [bm25_score_doc(query_terms, doc_idx, index) for doc_idx in range(index['N'])]

    ranking_df = pd.DataFrame({
        'Document_ID': list(range(index['N'])),
        'BM25_Score': scores,
        'Document_Preview': [doc[:100] + '...' if len(doc) > 100 else doc for doc in documents],
    })
    ranking_df = ranking_df.sort_values('BM25_Score', ascending=False).reset_index(drop=True)
    ranking_df['Rank'] = range(1, len(ranking_df) + 1)

    return ranking_df.head(top_k) if top_k else ranking_df


def score_queries_bm25(queries, documents, index=None, k1=1.5, b=0.75):
    """Score a list of queries and return a DataFrame with BM25 results."""
    if index is None:
        index = build_bm25_index(documents, k1=k1, b=b)

    result_rows = []
    for query_idx, query in enumerate(queries):
        query_terms = tokenize(query)
        for doc_idx in range(index['N']):
            result_rows.append({
                'query': query,
                'doc_index': doc_idx,
                'score': bm25_score_doc(query_terms, doc_idx, index),
            })

    results_df = pd.DataFrame(result_rows)
    return results_df.sort_values(by=['query', 'score'], ascending=[True, False]).reset_index(drop=True)
