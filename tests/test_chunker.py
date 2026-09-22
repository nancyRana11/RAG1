from src.document_processing.pdf_extractor import extract_text
from src.document_processing.ocr_processor import apply_ocr_fallback
from src.document_processing.chunker import chunk_document


PDF_PATH = r"C:\Users\nancy\Downloads\Nancy_Resume_Updated.pdf"


# Extract text from PDF
extraction = extract_text(PDF_PATH)

# Apply OCR to scanned pages if needed
extraction = apply_ocr_fallback(PDF_PATH, extraction)

# Create chunks
chunks = chunk_document(
    extraction,
    chunk_size=500,
    chunk_overlap=100
)

print("Total chunks:", len(chunks))

for chunk in chunks:
    print("\n-----------------------------")
    print("Chunk ID:", chunk.chunk_id)
    print("Page:", chunk.page_number)
    print("Text:", chunk.text)
    print("Metadata:", chunk.metadata)