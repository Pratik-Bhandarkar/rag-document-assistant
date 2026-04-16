import fitz
from pathlib import Path

OUTPUT_FOLDER = Path(r"D:\Personal\Projects\rag-document-assistant\output")
PAYSLIPS_FOLDER = Path(r"D:\Personal\Projects\rag-document-assistant\documents\payslips")

def extract_text_from_pdf(pdf_path):
    """Opens a PDF file and extracts all text, page by page."""
    document = fitz.open(pdf_path)
    all_text = []

    for page_number, page in enumerate(document):
        text = page.get_text()
        all_text.append(f"--- Page {page_number + 1} ---\n{text}")

    document.close()
    return "\n".join(all_text)

for pdf_file in PAYSLIPS_FOLDER.glob("*.pdf"):
    print(f"\nExtracting: {pdf_file.name}")
    extracted_text = extract_text_from_pdf(pdf_file)
    output_file = OUTPUT_FOLDER / (pdf_file.stem + ".txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(extracted_text)
    print(f"Saved: {output_file.name}")