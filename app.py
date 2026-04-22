import streamlit as st
from sentence_transformers import SentenceTransformer
from pathlib import Path
import chromadb
import ollama

st.set_page_config(
    page_title="Payslip Assistant",
    page_icon="📄",
    layout="centered"
)

CHROMA_DB_FOLDER = Path(r"D:\Personal\Projects\rag-document-assistant\chromadb")
COLLECTION_NAME = "payslips"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "llama3.2"
NUM_RESULTS = 5

@st.cache_resource
def load_models():
    """Loads the embedding model and ChromaDB collection once and caches them."""
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_DB_FOLDER))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return embedding_model, collection

embedding_model, collection = load_models()

st.title("📄 Payslip Assistant")
st.markdown("Ask questions about your payslips in English or German.")
st.divider()

st.sidebar.header("🔍 Filter Documents")

employer = st.sidebar.selectbox(
    "Employer",
    options=["All", "Merck", "Aioneers", "EFESO"]
)

month = st.sidebar.selectbox(
    "Month",
    options=["All", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
)

year = st.sidebar.selectbox(
    "Year",
    options=["All", "2023", "2024", "2025", "2026"]
)

employer = None if employer == "All" else employer
month = None if month == "All" else month
year = None if year == "All" else year

def search(query, employer=None, month=None, year=None):
    """Searches ChromaDB for the most relevant chunks."""
    query_embedding = embedding_model.encode(query).tolist()
    
    conditions = []
    if employer:
        conditions.append({"employer": {"$eq": employer}})
    if month:
        conditions.append({"month": {"$eq": month}})
    if year:
        conditions.append({"year": {"$eq": year}})
    
    if len(conditions) == 1:
        where_filter = conditions[0]
    elif len(conditions) > 1:
        where_filter = {"$and": conditions}
    else:
        where_filter = None
    
    if where_filter:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=NUM_RESULTS,
            where=where_filter
        )
    else:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=NUM_RESULTS
        )
    
    return results

def generate_answer(query, results):
    """Sends the question and relevant chunks to Llama 3.2 and returns an answer."""
    chunks_text = "\n\n".join(results["documents"][0])
    
    prompt = f"""You are a helpful multilingual assistant that answers questions about payslips and salary documents.
You can understand questions in both English and German, and should answer in the same language the question was asked in.

Important rules:
- Use ONLY the information provided in the context below to answer the question
- If the question is in German, answer in German
- If the question is in English, answer in English
- Be precise with numbers and currency amounts
- If you see repeated information in the context, use it only once
- If the answer is not clearly present in the context, say "I could not find that information in the documents"
- Never make up or estimate numbers that are not explicitly in the context

Context:
{chunks_text}

Question: {query}

Answer:"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response["message"]["content"]

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask a question about your payslips..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            results = search(query, employer=employer, month=month, year=year)
            answer = generate_answer(query, results)
        
        st.markdown(answer)
        
        with st.expander("📚 Source chunks used"):
            for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
                st.caption(f"Source {i+1}: {meta['source_file']} — Chunk {meta['chunk_number']}")
                st.text(doc[:200] + "...")
        
        st.session_state.messages.append({"role": "assistant", "content": answer})