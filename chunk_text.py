from pathlib import Path
import json

OUTPUT_FOLDER = Path(r"D:\Personal\Projects\rag-document-assistant\output")
CHUNKS_FOLDER = Path(r"D:\Personal\Projects\rag-document-assistant\chunks")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def parse_filename(filename):
    """Extracts employer, month and year from a payslip filename."""
    stem = Path(filename).stem
    parts = stem.split("_")
    
    employer = parts[0]
    month = parts[1]
    year = parts[2]
    
    return employer, month, year

def chunk_text(text, source_file):
    """Breaks a large text into overlapping chunks with metadata."""
    employer, month, year = parse_filename(source_file)
    chunks = []
    start = 0
    chunk_number = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk_content = text[start:end]

        chunk = {
            "chunk_number": chunk_number,
            "source_file": source_file,
            "employer": employer,
            "month": month,
            "year": year,
            "content": chunk_content
        }

        chunks.append(chunk)
        chunk_number += 1
        start = end - CHUNK_OVERLAP

    return chunks

CHUNKS_FOLDER.mkdir(exist_ok=True)

for txt_file in OUTPUT_FOLDER.glob("*.txt"):
    text = txt_file.read_text(encoding="utf-8")
    chunks = chunk_text(text, source_file=txt_file.name)

    output_file = CHUNKS_FOLDER / (txt_file.stem + ".json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Chunked: {txt_file.name} → {len(chunks)} chunks")