import os
import glob
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import textwrap

# Optional Gemini (Google) client
try:
    from google import genai
except Exception:
    genai = None
try:
    from google.cloud import aiplatform
except Exception:
    aiplatform = None

DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"

def build_embeddings(texts, model_name=DEFAULT_EMBED_MODEL, batch_size=32):
    model = SentenceTransformer(model_name)
    embeds = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    return embeds, model

def save_index(df, embeddings, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # Try parquet first, fall back to CSV if pyarrow/fastparquet not installed
    try:
        df.to_parquet(os.path.join(out_dir, "recipe_df.parquet"), index=False)
    except Exception:
        df.to_csv(os.path.join(out_dir, "recipe_df.csv"), index=False)
    np.save(os.path.join(out_dir, "embeddings.npy"), embeddings)

def load_index(out_dir):
    parquet_path = os.path.join(out_dir, "recipe_df.parquet")
    csv_path = os.path.join(out_dir, "recipe_df.csv")
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
    else:
        df = pd.read_csv(csv_path)
    embeddings = np.load(os.path.join(out_dir, "embeddings.npy"))
    return df, embeddings

def retrieve(query, df, embeddings, model, top_k=3):
    q_emb = model.encode([query], convert_to_numpy=True)
    sims = cosine_similarity(q_emb, embeddings)[0]
    idx = np.argsort(-sims)[:top_k]
    results = []
    for i in idx:
        results.append({"id": df.iloc[i]["id"], "title": df.iloc[i]["title"], "path": df.iloc[i]["path"], "score": float(sims[i]), "text": df.iloc[i]["text"]})
    return results

def rag_answer(query, docs, use_openai=False, openai_client=None, llm_model="gpt-3.5-turbo"):
    # Simple prompt that concatenates retrieved docs and asks the LLM to answer.
    context = "\n\n".join([f"Source {i+1} (score={d['score']:.3f}):\n{d['text']}" for i, d in enumerate(docs)])
    prompt = textwrap.dedent(f"""
    You are a helpful assistant. Use the following retrieved recipe documents to answer the user query.

    Retrieved documents:\n{context}

    Query: {query}\n
    Provide a concise answer and cite which Source numbers you used.
    """)

    # Try OpenAI first, then Gemini if requested via environment.
    use_gemini = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))

    if use_openai and openai_client is not None:
        # openai_client is expected to be the `openai` module with api key set in env
        resp = openai_client.ChatCompletion.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.0,
        )
        return resp["choices"][0]["message"]["content"].strip()
    elif use_gemini:
        gemini_model = os.getenv("GEMINI_MODEL", "models/chat-bison-001")
        try:
            # Prefer google.generativeai if available
            if genai is not None:
                api_key = "AQ.Ab*************"
                if api_key:
                    genai.configure(api_key=api_key)
                resp = genai.chat.create(model=gemini_model, messages=[{"role": "user", "content": prompt}])
                # attempt to extract text
                if hasattr(resp, "content"):
                    return resp.content[0].text
                elif isinstance(resp, dict):
                    # fallback dict parsing
                    choices = resp.get("candidates") or resp.get("output")
                    if choices:
                        return choices[0].get("content", "")
                return str(resp)
            # Fallback: Vertex AI chat model
            if aiplatform is not None:
                chat_model = aiplatform.ChatModel.from_pretrained(gemini_model)
                resp = chat_model.predict([{"role": "user", "content": prompt}])
                return str(resp)
        except Exception as e:
            return f"Gemini call failed: {e}\n(Ensure GOOGLE_API_KEY or GOOGLE_APPLICATION_CREDENTIALS are set and the google-generative-ai/client libraries are installed.)"
    else:
        # fallback
        # fallback: return the concatenated context and query as a simple answer
        fallback = "\n\n".join([f"- {d['title']} (score={d['score']:.3f})" for d in docs])
        return f"Retrieved {len(docs)} documents:\n{fallback}\n\n(Enable OpenAI and set OPENAI_API_KEY to get a generated answer.)"
