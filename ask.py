from sentence_transformers import SentenceTransformer
from pathlib import Path
import chromadb
import ollama

CHROMA_DB_FOLDER = Path(r"D:\Personal\Projects\rag-document-assistant\chromadb")
COLLECTION_NAME = "payslips"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "llama3.2"
NUM_RESULTS = 5

print(f"Loading embedding model: {EMBEDDING_MODEL}")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

client = chromadb.PersistentClient(path=str(CHROMA_DB_FOLDER))
collection = client.get_or_create_collection(name=COLLECTION_NAME)

print(f"Connected to ChromaDB collection: {COLLECTION_NAME}")
print(f"Total chunks in database: {collection.count()}")
print(f"LLM: {LLM_MODEL} running via Ollama")
print("\nReady to answer questions!\n")

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
    
    prompt = f"""You are a helpful assistant that answers questions about payslips and salary documents.
Use only the information provided in the context below to answer the question.
If the answer is not in the context, say "I could not find that information in the documents."

Context:
{chunks_text}

Question: {query}

Answer:"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response["message"]["content"]

while True:
    query = input("\nAsk a question about your payslips (or type 'exit' to quit): ")
    if query.lower() == "exit":
        break
    
    employer = input("Filter by employer? (Merck / Aioneers / EFESO or press Enter to skip): ").strip() or None
    month = input("Filter by month? (e.g. Feb / Jan / Sep or press Enter to skip): ").strip() or None
    year = input("Filter by year? (e.g. 2023 / 2024 or press Enter to skip): ").strip() or None
    
    print("\nSearching documents...")
    results = search(query, employer=employer, month=month, year=year)
    
    print("Generating answer...\n")
    answer = generate_answer(query, results)
    
    print("=" * 60)
    print(f"Answer:\n{answer}")
    print("=" * 60)