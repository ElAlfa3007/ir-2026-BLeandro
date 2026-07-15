import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import google.generativeai as genai
import gradio as gr

# 1. CONFIGURACIÓN DE SEGURIDAD
# Hugging Face Spaces inyecta el secreto directamente en las variables de entorno
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# 2. INICIALIZACIÓN DE DATOS Y MODELOS
print("Iniciando servidor RAG y cargando modelos...")
df = pd.read_csv('arxiv_sample.csv').head(300)
corpus = df['text_to_embed'].tolist()

retriever_model = SentenceTransformer('all-MiniLM-L6-v2')
reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Creación del índice FAISS
corpus_embeddings = retriever_model.encode(corpus, convert_to_numpy=True)
faiss.normalize_L2(corpus_embeddings)
dimension = corpus_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(corpus_embeddings)

instrucciones_sistema = """Eres un asistente académico experto. Tu tarea es responder la pregunta del usuario utilizando ÚNICAMENTE los Contextos. 
Si la información no es suficiente, debes responder EXACTAMENTE: "El corpus no contiene información suficiente para responder a esta consulta."
No inventes información y siempre cita tus fuentes (ej. [Documento 1])."""

modelo_generativo = genai.GenerativeModel(
    model_name="gemini-1.5-flash-latest",
    system_instruction=instrucciones_sistema
)

# 3. FUNCIONES DEL PIPELINE RAG
def recuperar_documentos(query, top_k_inicial=50, top_k_final=3):
    query_embedding = retriever_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    
    distancias, indices = index.search(query_embedding, top_k_inicial)
    candidatos_indices = indices[0]
    candidatos_textos = [corpus[i] for i in candidatos_indices]
    
    pares_evaluacion = [[query, texto] for texto in candidatos_textos]
    scores_rerank = reranker_model.predict(pares_evaluacion)
    
    resultados_ordenados = sorted(zip(candidatos_indices, scores_rerank), key=lambda x: x[1], reverse=True)
    top_resultados = resultados_ordenados[:top_k_final]
    
    documentos_recuperados = []
    for idx, score in top_resultados:
        documentos_recuperados.append({
            "titulo": df.iloc[idx]['titles'],
            "abstract": df.iloc[idx]['summaries'],
            "score": float(score)
        })
    return documentos_recuperados

def generar_respuesta_rag(query, documentos_recuperados):
    if not documentos_recuperados:
         return "El corpus no contiene información suficiente para responder a esta consulta."
    
    texto_contextos = ""
    for i, doc in enumerate(documentos_recuperados):
        texto_contextos += f"--- Documento {i+1} ---\nTítulo: {doc['titulo']}\nResumen: {doc['abstract']}\n\n"
        
    prompt_usuario = f"Contextos:\n{texto_contextos}\n\nPregunta: {query}"
    
    try:
        respuesta = modelo_generativo.generate_content(prompt_usuario, generation_config=genai.GenerationConfig(temperature=0.1))
        return respuesta.text
    except Exception as e:
        return f"Error en la generación: {str(e)}"

def funcion_interfaz_rag(mensaje, historial):
    documentos = recuperar_documentos(mensaje)
    respuesta_llm = generar_respuesta_rag(mensaje, documentos)
    
    origen_evidencias = ""
    if documentos and "El corpus no contiene información suficiente" not in respuesta_llm:
        origen_evidencias += "\n\n---\n### Evidencias utilizadas para construir la respuesta:\n"
        for i, doc in enumerate(documentos):
            origen_evidencias += f"**[{i+1}] {doc['titulo']}** *(Score de relevancia: {doc['score']:.2f})*\n> {doc['abstract']}\n\n"
    else:
        origen_evidencias += "\n\n---\n### Evidencias:\n*Ninguna evidencia del corpus fue suficientemente relevante para esta consulta.*"
    
    return f"{respuesta_llm}{origen_evidencias}"

# 4. INTERFAZ GRÁFICA
demo = gr.ChatInterface(
    fn=funcion_interfaz_rag,
    title=" Sistema RAG - arXiv (Examen Final)",
    description="Asistente conversacional para consultas sobre resúmenes de artículos científicos de arXiv utilizando búsquedas vectoriales y Gemini.",
    examples=[
        "What are the main applications of Graph Neural Networks?",
        "How is reinforcement learning used in robotics?",
        "Recent advances in diffusion models for image generation.",
        "Techniques for improving retrieval-augmented generation systems."
    ]
)

if __name__ == "__main__":
    import os
    puerto = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=puerto)