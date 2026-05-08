import base64
import os

import fitz  # PyMuPDF
import requests


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "AIzaSyBQhoJhC7JpDdiKGHhjz3HP45uoeGDmIHE")
OPENROUTER_URL = os.getenv(
    "OPENROUTER_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
)
OPENROUTER_OCR_MODEL = os.getenv("OPENROUTER_OCR_MODEL", "gemini-1.5-flash")
APP_BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:8501")
APP_TITLE = os.getenv("APP_TITLE", "Smart Document Analyser")

def extract_text(file_bytes: bytes) -> str:
    """
    Extract text from a PDF file provided as bytes.
    Uses PyMuPDF (fitz) to open the PDF from memory and concatenate text from all pages.
    Falls back to OCR for pages without a text layer when OpenRouter is configured.
    Cleans extra whitespace and returns a single string.
    """
    pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
    full_text = []

    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text = _extract_page_text(page)
        if not text and OPENROUTER_API_KEY:
            text = _ocr_page_text(page)
        if text:
            full_text.append(text)

    pdf_document.close()

    combined = " ".join(full_text)
    cleaned = " ".join(combined.split())
    return cleaned


def _extract_page_text(page: fitz.Page) -> str:
    """Try multiple extraction modes before falling back to OCR."""
    direct_text = page.get_text("text")
    if direct_text and direct_text.strip():
        return direct_text

    block_text = " ".join(
        block[4].strip()
        for block in page.get_text("blocks")
        if len(block) > 4 and block[4].strip()
    )
    if block_text.strip():
        return block_text

    word_entries = page.get_text("words")
    if word_entries:
        words = " ".join(entry[4] for entry in word_entries if len(entry) > 4)
        if words.strip():
            return words

    return ""


def _ocr_page_text(page: fitz.Page) -> str:
    """OCR a single PDF page with OpenRouter image input."""
    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    image_bytes = pixmap.tobytes("png")
    image_base64 = base64.b64encode(image_bytes).decode("ascii")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": APP_BASE_URL,
        "X-Title": APP_TITLE,
    }
    payload = {
        "model": OPENROUTER_OCR_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Extract all readable text from this document page. "
                            "Return plain text only. If the page has no readable text, return an empty string."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                        },
                    },
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 1200,
    }

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()
        text = result["choices"][0]["message"]["content"]
        return text.strip() if isinstance(text, str) else ""
    except Exception:
        return ""
