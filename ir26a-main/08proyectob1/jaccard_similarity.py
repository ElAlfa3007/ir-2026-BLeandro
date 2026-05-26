"""
Módulo de Similitud Jaccard para Recuperación de Información
Implementa similitud Jaccard binaria usando representación de vectores de presencia/ausencia
"""

import re
from collections import Counter
import pandas as pd


def tokenize(text):
    """Tokenizar texto simplemente por palabras."""
    return re.findall(r"\w+", text.lower())


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """
    Calcula la similitud de Jaccard entre dos conjuntos.
    
    Similitud Jaccard = |A ∩ B| / |A ∪ B|
    
    Args:
        set_a: Conjunto A (términos únicos del documento 1)
        set_b: Conjunto B (términos únicos del documento/query 2)
    
    Returns:
        float: Valor de similitud entre 0 y 1
    """
    if not set_a or not set_b:
        return 0.0
    
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    
    return intersection / union if union > 0 else 0.0


def binary_vector_jaccard(doc_tokens, query_tokens) -> float:
    """
    Calcula similitud Jaccard usando representación binaria (presencia/ausencia).
    
    No considera frecuencias, solo presencia de términos únicos.
    
    Args:
        doc_tokens: Lista de tokens del documento
        query_tokens: Lista de tokens de la query
    
    Returns:
        float: Valor de similitud Jaccard entre 0 y 1
    """
    doc_set = set(doc_tokens)
    query_set = set(query_tokens)
    
    return jaccard_similarity(doc_set, query_set)


def jaccard_rank(query, documents, tokenize_func=tokenize, top_k=None):
    """
    Ranking de documentos usando similitud Jaccard.
    
    Args:
        query: String de la query a buscar
        documents: Lista de strings con los documentos
        tokenize_func: Función de tokenización a usar (default: tokenize local)
        top_k: Número de documentos a retornar (None = todos)
    
    Returns:
        pd.DataFrame: Ranking ordenado con columnas:
                     - Document_ID: índice del documento
                     - Jaccard_Score: similitud Jaccard
                     - Document_Preview: primeros 100 caracteres
                     - Rank: posición en el ranking
    """
    query_tokens = tokenize_func(query)
    scores = []
    
    for doc_idx, doc in enumerate(documents):
        doc_tokens = tokenize_func(doc)
        score = binary_vector_jaccard(doc_tokens, query_tokens)
        scores.append({
            'Document_ID': doc_idx,
            'Jaccard_Score': score,
            'Document_Preview': doc[:100] + '...' if len(doc) > 100 else doc,
        })
    
    # Crear DataFrame y ordenar por similitud (descendente)
    ranking_df = pd.DataFrame(scores)
    ranking_df = ranking_df.sort_values('Jaccard_Score', ascending=False).reset_index(drop=True)
    ranking_df['Rank'] = range(1, len(ranking_df) + 1)
    
    # Retornar top_k si se especifica
    return ranking_df.head(top_k) if top_k else ranking_df


def score_queries_jaccard(queries, documents, tokenize_func=tokenize):
    """
    Score múltiples queries contra documentos usando Jaccard.
    
    Retorna un DataFrame con las puntuaciones de similitud para cada pareja query-documento.
    
    Args:
        queries: Lista de strings con las queries
        documents: Lista de strings con los documentos
        tokenize_func: Función de tokenización a usar
    
    Returns:
        pd.DataFrame: Resultados con columnas:
                     - query: texto de la query
                     - doc_index: índice del documento
                     - score: similitud Jaccard
                     Ordenado por query y score (descendente)
    """
    result_rows = []
    
    for query_idx, query in enumerate(queries):
        query_tokens = tokenize_func(query)
        
        for doc_idx, doc in enumerate(documents):
            doc_tokens = tokenize_func(doc)
            score = binary_vector_jaccard(doc_tokens, query_tokens)
            
            result_rows.append({
                'query': query,
                'doc_index': doc_idx,
                'score': score,
            })
    
    # Crear DataFrame y ordenar
    results_df = pd.DataFrame(result_rows)
    results_df = results_df.sort_values(
        by=['query', 'score'], 
        ascending=[True, False]
    ).reset_index(drop=True)
    
    return results_df
