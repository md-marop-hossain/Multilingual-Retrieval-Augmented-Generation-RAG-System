import streamlit as st
import requests
from app import search_similar_chunks

API_URL = "http://127.0.0.1:8000"  

st.set_page_config(page_title="Bangla-English RAG System", layout="centered")

st.title("📚 Bangla-English RAG QA System")
st.caption("Ask questions based on your HSC26_Bangla.pdf knowledge base.")

# Chat mode toggle
mode = st.radio("Choose Mode:", ["Ask (Short Answer)", "Evaluate Answer"])

# Input
question = st.text_input("🔎 Your Question (Bangla or English)", placeholder="উদাহরণ: কাকে অনুপমের ভাগ্য দেবতা বলা হয়েছে?")

if st.button("Submit") and question.strip():
    with st.spinner("Thinking... 💭"):
        try:
            if mode == "Ask (Short Answer)":
                res = requests.post(f"{API_URL}/ask", json={"question": question})
                if res.status_code == 200:
                    data = res.json()
                    st.success(f"🧠 Answer: {data['answer']}")
                    with st.expander("📄 Retrieved Chunks"):
                        for i, chunk in enumerate(data['top_chunks']):
                            st.markdown(f"**Chunk {i+1}:**\n{chunk}")
                else:
                    st.error("❌ Failed to get response.")
            else:
                res = requests.post(f"{API_URL}/evaluate", json={"question": question})
                if res.status_code == 200:
                    data = res.json()
                    st.success(f"🧠 Answer: {data['answer']}")
                    st.markdown(f"🔍 **Groundedness Check**: {data['groundedness_check']}")
                    st.markdown(f"📊 **Cosine Similarity Avg**: {data['cosine_similarity']['average']}")
                    with st.expander("📄 Top Chunks"):
                        for i, chunk in enumerate(search_similar_chunks(question)):
                            st.markdown(f"**Chunk {i+1}:**\n{chunk}")
                else:
                    st.error("❌ Failed to get evaluation.")
        except Exception as e:
            st.error(f"⚠️ Exception: {e}")
