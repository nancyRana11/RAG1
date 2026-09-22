from dataclasses import dataclass, field
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.document_processing.pdf_extractor import PDFExtractionResult
from src.utils.logger import logger


DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    page_number: int
    metadata: dict = field(default_factory=dict)


def chunk_document(
    extraction: PDFExtractionResult,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> list[DocumentChunk]:

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = []

    file_path = Path(extraction.file_path)
    file_name = file_path.name

    for page in extraction.pages:

        page_text = page.text.strip()

        if not page_text:
            continue

        page_chunks = splitter.split_text(page_text)

        for i, text in enumerate(page_chunks):

            chunk_id = (
                f"{file_name}_p{page.page_number}_c{i}"
            )

            metadata = {
                "source": extraction.file_path,
                "file_name": file_name,
                "page_number": page.page_number,
                "chunk_id": chunk_id,
                "chunk_index_in_page": i,
                "total_chunks_in_page": len(page_chunks),
            }

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    text=text,
                    page_number=page.page_number,
                    metadata=metadata
                )
            )

    logger.info(
        f"Chunked '{file_name}': "
        f"{extraction.num_pages} pages -> "
        f"{len(chunks)} chunks"
    )

    return chunks