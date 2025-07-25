# Developed a Simple Multilingual Retrieval-Augmented Generation (RAG) System
Built a lightweight RAG system capable of answering Bangla and English queries from Bangla PDF documents. The system uses Tesseract OCR for Bangla text extraction, OpenAI’s ```text-embedding-3-large``` for multilingual embedding, FAISS for semantic retrieval, and ```GPT-4o``` for context-aware responses. A Streamlit interface communicates with a FastAPI backend for smooth and interactive user experience.

## 🔄 SYSTEM SUMMARY FLOW

<p align="center">
  <img src="images/Flowchart of User Query Process.png" width="600"/>
</p>


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

**Step 1: Clone the Repository and Set Up Environment**
```
git clone https://github.com/md-marop-hossain/Multilingual-Retrieval-Augmented-Generation-RAG-System.git
cd Multilingual-Retrieval-Augmented-Generation-RAG-System
python -m venv myenv
source myenv/bin/activate  # or myenv\Scripts\activate on Windows
pip install -r requirements.txt
```
**Step 2: Extract Text from PDF**

Run the following command to extract text from the PDF using ```pdf_extractor.py```:
   
```python pdf_extractor.py```

**Step 3: Build Vector Index**

This step includes chunking, embedding, and creating the vector store:

```python build_vector_index.py```

**Step 4: Launch the FastAPI Server**

Start the FastAPI app using:

```uvicorn app:app --reload```

 **Available API Endpoints:**
   - Ask a question : ```POST http://127.0.0.1:8000/ask```
   - Evaluate the system : ```POST http://127.0.0.1:8000/evaluate```
   
**Step 5: Start the Streamlit UI**

```streamlit run rag_ui.py```

## 🛠️ Tools & Libraries Used

| Category              | Tool/Library           | Purpose                                                                 |
|-----------------------|------------------------|-------------------------------------------------------------------------|
| LLM & Embeddings      | OpenAI GPT-4o          | Generates grounded answers from retrieved context                       |
|                       | text-embedding-3-large | Multilingual, high-quality embeddings for semantic chunk comparison     |
| Vector Store          | FAISS                  | Fast Approximate Nearest Neighbor search over dense vector space        |
| Backend Framework     | FastAPI                | Backend API for handling queries, answers, and evaluations              |
| Frontend Interface    | Streamlit              | Provides a user-friendly web UI for interacting with the RAG system     |
| PDF Text Extraction   | PyMuPDF (fitz)         | Extracts formatted Bangla and English text from PDFs                    |
| OCR (Image PDF)       | pytesseract            | Extracts text from scanned or image-based PDFs using Tesseract OCR      |
| Image Processing      | Pillow (PIL)           | Handles image preprocessing for OCR pipeline                            |
| Evaluation            | scikit-learn           | Computes cosine similarity for answer relevance scoring                 |
| API Key Management    | python-dotenv          | Loads sensitive keys like `OPENAI_API_KEY` securely from `.env` file    |
| Async Handling        | nest_asyncio           | Enables async FastAPI in notebook-style or blocking environments        |
| Tokenization Utility  | tiktoken               | Used to count tokens (e.g., for chunk size management)        |
| Data Serialization    | pickle                 | Loads/stores vector index and text chunks                               |
| Math/Numeric          | numpy                  | Vector and matrix operations                                            |
| Typing & Validation   | pydantic               | Defines and validates API input/output models                           |
| Server                | uvicorn                | ASGI server for running FastAPI app                                     |

## 💬 Sample Queries and Outputs

<table>
  <tr>
    <td><img src="images/api.png" width="400"/></td>
    <td><img src="images/question.png" width="400"/></td>
  </tr>
  <tr>
    <td><img src="images/answerOne.png" width="400"/></td>
    <td><img src="images/answerTwo.png" width="400"/></td>
  </tr>
  <tr>
    <td><img src="images/cosn.png" width="400"/></td>
    <td><img src="images/steamlit_interface.png" width="400"/></td>
  </tr>
</table>

## 🧾 Questions & Answers

📝 **What method or library did you use to extract the text, and why? Did you face any formatting challenges with the PDF content?**

I used PyMuPDF (```fitz```) to render each page of the PDF as an image and Tesseract OCR (via ```pytesseract```) to extract text from those images. This approach was necessary because the PDF contains scanned Bangla text (not selectable or searchable), which made traditional text extraction tools like ```PyPDF2``` or ```pdfminer``` ineffective.

To support Bangla language OCR, I configured Tesseract with the ```ben``` language model and ensured proper setup with ```TESSDATA_PREFIX``` on Windows.

Yes, there were formatting challenges:
- Many pages had inconsistent spacing, broken characters, and misaligned text blocks.
- OCR sometimes misinterpreted compound Bangla characters or punctuation.
- To address this, I used Unicode normalization (```unicodedata```) and regular expressions to clean and standardize the output.

This OCR-based method allowed me to extract readable and indexable Bangla text from the scanned PDF for downstream RAG tasks.

📝 **What chunking strategy did you choose (e.g. paragraph-based, sentence-based, character limit)? Why do you think it works well for semantic retrieval?**

I used a sentence-based chunking strategy with a 500-token limit per chunk.

- This ensures each chunk is semantically complete and doesn't cut off important context.
- It works well for semantic retrieval because meaningful, token-aware chunks lead to better embedding quality and more accurate search results in FAISS.
- It’s also language-friendly, adapting well to both Bangla and English sentence structures.

📝 **What embedding model did you use? Why did you choose it? How does it capture the meaning of the text?**

I used OpenAI's ```text-embedding-3-large``` model.

- Why? It offers high-quality multilingual support, making it ideal for both Bangla and English queries.
- How? It converts text into dense vectors that capture semantic meaning, allowing similar ideas to be close in vector space—even across languages.
- Result: More accurate chunk retrieval and better answers in our RAG system.

📝 **How are you comparing the query with your stored chunks? Why did you choose this similarity method and storage setup?**

I compare the query with stored chunks using FAISS and L2 (Euclidean) distance on the embedding vectors.

- Why FAISS? It's fast, scalable, and optimized for high-dimensional vector search.
- Why L2 distance? It works well with OpenAI embeddings, providing reliable similarity scoring.
- Storage Setup: We use FAISS for the index and store the corresponding text chunks in a pickle file for quick retrieval.

This setup ensures efficient, accurate semantic search in both Bangla and English.

📝 **How do you ensure that the question and the document chunks are compared meaningfully? What would happen if the query is vague or missing context?**

Ensuring Meaningful Comparison:
- I use OpenAI’s ```text-embedding-3-large``` to embed both the query and the document chunks in the same semantic space, ensuring meaningful comparisons.
- FAISS then retrieves chunks closest in meaning, not just based on keywords.

If the query is vague or lacks context:
- The system may retrieve less relevant chunks, leading to generic or incomplete answers.
- However, the embedding model still tries to infer intent from available cues, often retrieving semantically related content.

To improve performance, we recommend clear and specific queries.

📝 **Do the results seem relevant? If not, what might improve them (e.g. better chunking, better embedding model, larger document)?**

Yes, results are generally relevant thanks to:
- Sentence-based chunking
- High-quality embeddings (```text-embedding-3-large```)
- Semantic search via FAISS

When Results May Seem Irrelevant:
- If chunks are too short or too long → context may be lost or diluted
- If query is vague or ambiguous → embeddings may not capture intent
- If the document is too small → not enough info to match

How to Improve:
- Use dynamic chunking based on meaning and topic
- Fine-tune chunk length (e.g., 300–600 tokens)
- Add metadata filtering or reranking
- Use long-context models like GPT-4o with better prompt design

