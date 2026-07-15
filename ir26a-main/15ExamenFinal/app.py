import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import google.generativeai as genai
import streamlit as st

# Configuración de la página
st.set_page_config(page_title="RAG arXiv - Examen Final", page_icon="🧬")
st.title(" Sistema RAG - arXiv (Examen Final)")

# 1. SEGURIDAD Y API KEY
gemini_api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

if not gemini_api_key:
    st.error("⚠️ Falta la API Key de Gemini. Configúrala en los Secrets de Streamlit.")
    st.stop()
else:
    genai.configure(api_key=gemini_api_key)

# 2. CACHÉ DE MODELOS Y DATOS (VITAL PARA STREAMLIT)
@st.cache_resource(show_spinner="Cargando modelos e índice FAISS... (Esto toma un minuto en el primer arranque)")
def cargar_sistema_rag():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ruta_csv = os.path.join(BASE_DIR, 'arxiv_sample.csv')
    df = pd.read_csv(ruta_csv)

    corpus = df['text_to_embed'].tolist()
    
    retriever_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    corpus_embeddings = retriever_model.encode(corpus, convert_to_numpy=True)
    faiss.normalize_L2(corpus_embeddings)
    dimension = corpus_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(corpus_embeddings)
    
    instrucciones = """Eres un asistente académico experto. Tu tarea es responder la pregunta utilizando ÚNICAMENTE los Contextos. 
    Si la información no es suficiente, debes responder EXACTAMENTE: "El corpus no contiene información suficiente para responder a esta consulta.
    Responde adecuadamente si a cada consulta con un: Se ha obtenido esta información de los artículos y de ser negativo: No se ha encontrado dentro del corpus, sin embargo esta es la aproximación más cercana. No inventes información y siempre cita tus fuentes (ej. [Documento 1]). Adicional si la consulta la hacen en cualquier otra idioma, tradúcela al inglés."""
    
    modelo = genai.GenerativeModel(model_name="gemini-2.5-flash", system_instruction=instrucciones)
    
    return df, corpus, retriever_model, reranker_model, index, modelo

# Ejecutamos la carga en caché
df, corpus, retriever_model, reranker_model, index, modelo_generativo = cargar_sistema_rag()

# 3. FUNCIONES CORE
def recuperar_documentos(query, top_k_inicial=50, top_k_final=3):
    query_embedding = retriever_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    
    distancias, indices = index.search(query_embedding, top_k_inicial)
    candidatos_indices = indices[0]
    candidatos_textos = [corpus[i] for i in candidatos_indices]
    
    pares_evaluacion = [[query, texto] for texto in candidatos_textos]
    scores_rerank = reranker_model.predict(pares_evaluacion)
    
    resultados_ordenados = sorted(zip(candidatos_indices, scores_rerank), key=lambda x: x[1], reverse=True)
    
    docs = []
    for idx, score in resultados_ordenados[:top_k_final]:
        docs.append({
            "titulo": df.iloc[idx]['titles'],
            "abstract": df.iloc[idx]['summaries'],
            "score": float(score)
        })
    return docs

def generar_respuesta(query, documentos):
    if not documentos:
         return "El corpus no contiene información suficiente para responder a esta consulta."
    
    contextos = "".join([f"--- Documento {i+1} ---\nTítulo: {d['titulo']}\nResumen: {d['abstract']}\n\n" for i, d in enumerate(documentos)])
    prompt = f"Contextos:\n{contextos}\n\nPregunta: {query}"
    
    try:
        resp = modelo_generativo.generate_content(prompt, generation_config=genai.GenerationConfig(temperature=0.1))
        return resp.text
    except Exception as e:
        return f"Error: {str(e)}"

# 4. INTERFAZ DE CHAT DE STREAMLIT
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

# Mostrar historial
for mensaje in st.session_state.mensajes:
    with st.chat_message(mensaje["role"]):
        st.markdown(mensaje["content"])

# Entrada del usuario
if prompt := st.chat_input("Escribe tu consulta (ej. What are the main applications of Graph Neural Networks?)"):
    # Mostrar mensaje del usuario
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.mensajes.append({"role": "user", "content": prompt})
    
    # Procesar respuesta
    with st.chat_message("assistant"):
        with st.spinner("Buscando en arXiv y generando respuesta..."):
            docs = recuperar_documentos(prompt)
            respuesta_llm = generar_respuesta(prompt, docs)
            
            # Construir evidencias
            evidencias = ""
            if docs and "El corpus no contiene información suficiente" not in respuesta_llm:
                evidencias = "\n\n---\n### Evidencias utilizadas:\n"
                for i, doc in enumerate(docs):
                    evidencias += f"**[{i+1}] {doc['titulo']}** *(Score: {doc['score']:.2f})*\n> {doc['abstract']}\n\n"
            else:
                evidencias = "\n\n---\n### Evidencias:\n*Ninguna evidencia superó el umbral de relevancia.*"
            
            respuesta_final = f"{respuesta_llm}{evidencias}"
            st.markdown(respuesta_final)
            
    st.session_state.mensajes.append({"role": "assistant", "content": respuesta_final})