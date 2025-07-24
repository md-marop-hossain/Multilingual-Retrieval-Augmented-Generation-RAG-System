# Developed a Simple Multilingual Retrieval-Augmented Generation (RAG) System
Design and implement a basic RAG pipeline capable of understanding and responding to both English and Bengali queries. The system should fetch relevant information from a pdf document corpus and generate a meaningful answer grounded in retrieved content.

## 📦 Directory Structure
```
├── app.py # FastAPI backend
├── build_vector_index.py # Embeds and stores vector data using FAISS
├── pdf_extractor.py # Extracts and preprocesses text from PDF
├── rag_ui.py # Extracts and preprocesses text from PDF
├── index/
│ ├── faiss_index.idx # Vector index
│ └── valid_chunks.pkl # Serialized document chunks
├── myenv #virtual environment
├── .env # OpenAI API Key
├── data/
│ ├── cleaned_text.txt 
│ └── HSC26_Bangla.pdf 
└──.gitignore
```

## 🚀 Setup Instructions
```
git clone https://github.com/md-marop-hossain/Multilingual-Retrieval-Augmented-Generation-RAG-System.git
cd Multilingual-Retrieval-Augmented-Generation-RAG-System
python -m venv myenv
source myenv/bin/activate  # or myenv\Scripts\activate on Windows
pip install -r requirements.txt
```
