from sentence_transformers import SentenceTransformer
from pathlib import Path
import chromadb

CHROMA_DB_FOLDER = Path(r"D:\Personal\Projects\rag-document-assistant\chromadb")
COLLECTION_NAME = "payslips"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
NUM_RESULTS = 5

print(f"Loading embedding model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

client = chromadb.PersistentClient(path=str(CHROMA_DB_FOLDER))
collection = client.get_or_create_collection(name=COLLECTION_NAME)

print(f"Connected to ChromaDB collection: {COLLECTION_NAME}")
print(f"Total chunks in database: {collection.count()}\n")

def search(query, employer=None, month=None, year=None):
    """Takes a natural language question and returns the most relevant chunks."""
    query_embedding = model.encode(query).tolist()
    
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

def display_results(query, results):
    """Displays search results in a readable format."""
    print(f"\nQuery: {query}")
    print("=" * 60)
    
    for i, (document, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        print(f"\nResult {i + 1}:")
        print(f"Source: {metadata['source_file']}")
        print(f"Chunk: {metadata['chunk_number']}")
        print(f"Content:\n{document}")
        print("-" * 60)

while True:
    query = input("\nAsk a question about your payslips (or type 'exit' to quit): ")
    if query.lower() == "exit":
        break
    
    employer = input("Filter by employer? (Merck / Aioneers / EFESO or press Enter to skip): ").strip() or None
    month = input("Filter by month? (e.g. Feb / Jan / Sep or press Enter to skip): ").strip() or None
    year = input("Filter by year? (e.g. 2023 / 2024 or press Enter to skip): ").strip() or None
    
    results = search(query, employer=employer, month=month, year=year)
    display_results(query, results)