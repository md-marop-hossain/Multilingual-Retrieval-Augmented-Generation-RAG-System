import os
import numpy as np
import pickle
import faiss
import tiktoken
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import nest_asyncio

# Setup
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"  
client = OpenAI()
nest_asyncio.apply()

# Load vector index and chunks
index = faiss.read_index("index/faiss_index.idx")
with open("index/valid_chunks.pkl", "rb") as f:
    valid_chunks = pickle.load(f)


# Embedding utility
def get_embeddings_safe(texts):
    texts = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
    if not texts:
        return []

    try:
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=texts
        )
        return [d.embedding for d in response.data]
    except Exception as e:
        print("Embedding error:", e)
        return []


# Search
def search_similar_chunks(query, k=15):
    query_embedding = get_embeddings_safe([query])[0]
    D, I = index.search(np.array([query_embedding]), k)
    return [valid_chunks[i] for i in I[0]]

# Answer generator (Short & Relevant)
def generate_answer(query):
    context = "\n\n".join(search_similar_chunks(query))
    prompt = f"""You are a factual assistant that answers Bangla and English questions using the provided context only.

Your goal is to extract the **shortest and most accurate answer** based only on the context.

- Do not explain.
- Do not generate long sentences.
- Only return the **precise name, number, phrase, or sentence fragment** that directly answers the question.

Context:
{context}

Question: {query}
Answer:"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()

# Evaluation
def evaluate_cosine_similarity(query):
    query_emb = np.array([get_embeddings_safe([query])[0]])
    chunks = search_similar_chunks(query)
    chunk_embs = [get_embeddings_safe([c])[0] for c in chunks]
    scores = cosine_similarity(query_emb, chunk_embs)[0]
    return {
        "average": round(np.mean(scores), 4),
        "scores": [round(s, 4) for s in scores]
    }


def check_groundedness(query):
    answer = generate_answer(query)
    context = "\n\n".join(search_similar_chunks(query))
    prompt = f"""Check if the answer is grounded in the context.

Context:
{context}

Question: {query}
Answer: {answer}

Is the answer strictly grounded in the context? Reply YES or NO and explain briefly."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()

# FastAPI app
app = FastAPI()

class Query(BaseModel):
    question: str

@app.get("/")
def root():
    return {"status": "RAG API Running", "endpoints": ["/ask", "/evaluate"]}

@app.post("/ask")
def ask_rag(data: Query):
    answer = generate_answer(data.question)
    return {
        "question": data.question,
        "answer": answer,
        "top_chunks": search_similar_chunks(data.question)
    }

@app.post("/evaluate")
def evaluate_rag(data: Query):
    return {
        "question": data.question,
        "answer": generate_answer(data.question),
        "cosine_similarity": evaluate_cosine_similarity(data.question),
        "groundedness_check": check_groundedness(data.question)
    }
