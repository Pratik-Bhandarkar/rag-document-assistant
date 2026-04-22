from pathlib import Path
import json

OUTPUT_FOLDER = Path(r"D:\Personal\Projects\rag-document-assistant\output")
CHUNKS_FOLDER = Path(r"D:\Personal\Projects\rag-document-assistant\chunks")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
ABBREVIATIONS = {
    "HI": "Health Insurance (Krankenversicherung)",
    "PI": "Pension Insurance (Rentenversicherung)",
    "UI": "Unemployment Insurance (Arbeitslosenversicherung)",
    "CI": "Care Insurance (Pflegeversicherung)",
    "SI": "Social Insurance (Sozialversicherung)",
    "LSG": "Lohnsteuergruppe (Tax Group)",
    "EBeschV": "Entgeltbescheinigungsverordnung (Payslip Regulation)",
    "DEUEV": "Datenerfassungs- und Übermittlungsverordnung (Data Collection Regulation)",
    "KV-Beitrag": "Health Insurance Contribution (Krankenversicherung-Beitrag)",
    "RV-Beitrag": "Pension Insurance Contribution (Rentenversicherung-Beitrag)",
    "AV-Beitrag": "Unemployment Insurance Contribution (Arbeitslosenversicherung-Beitrag)",
    "PV-Beitrag": "Care Insurance Contribution (Pflegeversicherung-Beitrag)",
}

def expand_abbreviations(text):
    """Expands known abbreviations in text to improve embedding quality."""
    for abbr, full_form in ABBREVIATIONS.items():
        text = text.replace(abbr, full_form)
    return text

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
    text = expand_abbreviations(text)
    chunks = chunk_text(text, source_file=txt_file.name)

    output_file = CHUNKS_FOLDER / (txt_file.stem + ".json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Chunked: {txt_file.name} → {len(chunks)} chunks")