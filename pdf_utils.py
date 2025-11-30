# backend/pdf_utils.py

from pypdf import PdfReader

def extract_pdf(file) -> dict:
    """
    Extracts text per page + basic metadata from uploaded PDF.
    """
    reader = PdfReader(file)
    pages_text = []
    for page in reader.pages:
        pages_text.append(page.extract_text() or "")

    full_text = "\n".join(pages_text)

    metadata = {
        "title": reader.metadata.title if reader.metadata and reader.metadata.title else "Unknown Title",
        "author": reader.metadata.author if reader.metadata and reader.metadata.author else "Unknown Author",
        "num_pages": len(reader.pages),
    }

    return {
        "full_text": full_text,
        "pages_text": pages_text,
        "metadata": metadata,
    }
