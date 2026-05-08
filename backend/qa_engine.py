import os
from typing import List, Dict, Any
import requests
import re

# OpenRouter configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "AIzaSyBQhoJhC7JpDdiKGHhjz3HP45uoeGDmIHE")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "gemini-1.5-flash")
OPENROUTER_URL = os.getenv(
    "OPENROUTER_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
)
APP_BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:8501")
APP_TITLE = os.getenv("APP_TITLE", "Smart Document Analyser")

def load_model(model_name: str = OPENROUTER_MODEL):
    """
    Initialize OpenRouter client.
    This is kept for compatibility but doesn't load a local model anymore.
    """
    print(f"Using OpenRouter model: {model_name}")
    return None, None

def get_model_and_tokenizer():
    """Get the OpenRouter configuration (for compatibility)."""
    return OPENROUTER_URL, OPENROUTER_MODEL

def get_best_answer(question: str, chunks: List[str]) -> Dict[str, Any]:
    """
    Find the best answer to a question from the given text chunks.
    Uses OpenRouter API to extract answers with intelligent context selection.

    Returns a dictionary with:
    - answer: The extracted answer text
    - score: Confidence score (0-1)
    - chunk_index: Index of the chunk containing the answer
    - source_text: The full chunk text where the answer was found
    """
    if not chunks:
        return {
            "answer": None,
            "score": 0.0,
            "chunk_index": -1,
            "source_text": ""
        }

    url, model_name = get_model_and_tokenizer()

    # Find the most relevant chunks based on keyword matching
    relevant_chunks = _get_relevant_chunks(question, chunks, top_k=3)
    
    # Combine relevant chunks with proper spacing
    context = "\n\n".join(relevant_chunks['chunks'])
    best_chunk_index = relevant_chunks['indices'][0] if relevant_chunks['indices'] else -1
    best_source_text = relevant_chunks['chunks'][0] if relevant_chunks['chunks'] else ""

    # Create a refined prompt for better answers
    system_prompt = """You are an expert document analyst and question answerer. Your task is to:
1. Answer questions accurately based ONLY on the provided context
2. Provide clear, concise, and well-structured answers
3. If the answer is not explicitly in the context, say "This information is not available in the provided documents."
4. Always cite which part of the document supports your answer when possible
5. Format your answer in a clear and easy-to-understand manner"""

    user_prompt = f"""Please answer the following question based on the provided context.

Context:
{context}

Question: {question}

Instructions:
- Provide a direct, clear answer
- If the answer is in the context, cite it
- If not found, explicitly state that
- Keep the answer concise but complete
- Use bullet points if listing multiple items"""

    fallback_answer = _extract_answer_locally(
        question,
        relevant_chunks["chunks"],
        relevant_chunks["indices"],
    )

    try:
        if not OPENROUTER_API_KEY:
            return fallback_answer

        # Make API call to OpenRouter using requests
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": APP_BASE_URL,
            "X-Title": APP_TITLE,
        }

        data = {
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 500,
            "temperature": 0.1
        }

        response = requests.post(url, headers=headers, json=data, timeout=60)
        response.raise_for_status()  # Raise an exception for bad status codes

        result = response.json()
        answer_text = result["choices"][0]["message"]["content"].strip()

        # Check if the model couldn't find an answer
        if (
            "cannot find" in answer_text.lower()
            or "not available" in answer_text.lower()
            or "not found" in answer_text.lower()
        ):
            return fallback_answer

        # Calculate confidence score based on answer length and specificity
        # Longer, more detailed answers typically have higher confidence
        score = min(0.95, 0.6 + (len(answer_text.split()) / 100.0))
        score = max(0.5, score)  # Minimum confidence of 0.5

        # Clean up the answer: remove markdown formatting if present
        answer_text = _clean_answer_text(answer_text)

        return {
            "answer": answer_text,
            "score": score,
            "chunk_index": best_chunk_index,
            "source_text": best_source_text
        }

    except Exception as e:
        print(f"Error calling OpenRouter API: {e}")
        return fallback_answer


def _get_relevant_chunks(question: str, chunks: List[str], top_k: int = 3) -> Dict[str, Any]:
    """
    Find the most relevant chunks based on keyword matching with the question.
    Returns the top_k most relevant chunks with their indices.
    """
    if not chunks:
        return {"chunks": [], "indices": []}
    
    # Extract keywords from the question (excluding common words)
    question_lower = question.lower()
    stop_words = {'what', 'how', 'why', 'when', 'where', 'who', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'in', 'at', 'to', 'for', 'of', 'by', 'on'}
    question_words = set(word for word in question_lower.split() if word not in stop_words and len(word) > 2)
    
    # Score each chunk based on keyword matches
    chunk_scores = []
    for idx, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        # Count how many question words appear in the chunk
        score = sum(1 for word in question_words if word in chunk_lower)
        # Boost score if chunk contains exact phrase from question
        if any(phrase in chunk_lower for phrase in question_lower.split() if len(phrase) > 3):
            score += 2
        chunk_scores.append((idx, chunk, score))
    
    # Sort by score (descending) and get top_k
    sorted_chunks = sorted(chunk_scores, key=lambda x: x[2], reverse=True)
    top_chunks = sorted_chunks[:min(top_k, len(sorted_chunks))]
    
    # If no chunks matched well, just take the first few chunks
    if not top_chunks or top_chunks[0][2] == 0:
        top_chunks = [(i, chunk, 0) for i, chunk in enumerate(chunks[:top_k])]
    
    # Extract indices and chunks, preserving original order
    indices = [idx for idx, _, _ in top_chunks]
    chunk_texts = [chunk for _, chunk, _ in top_chunks]
    
    return {
        "chunks": chunk_texts,
        "indices": indices
    }


def _extract_answer_locally(
    question: str,
    relevant_chunks: List[str],
    relevant_indices: List[int],
) -> Dict[str, Any]:
    """
    Lightweight local fallback when the LLM is unavailable.
    Picks the most relevant sentence from the top candidate chunks.
    """
    if not relevant_chunks:
        return {
            "answer": None,
            "score": 0.0,
            "chunk_index": -1,
            "source_text": "",
        }

    question_terms = _extract_query_terms(question)
    best_match = None

    for position, chunk in enumerate(relevant_chunks):
        sentences = re.split(r"(?<=[.!?])\s+|\n+", chunk)
        candidates = [sentence.strip() for sentence in sentences if sentence.strip()]
        if not candidates:
            candidates = [chunk.strip()]

        for sentence in candidates:
            score = _score_text_match(question_terms, sentence)
            if not best_match or score > best_match["score"]:
                best_match = {
                    "answer": sentence,
                    "score": score,
                    "chunk_index": (
                        relevant_indices[position]
                        if position < len(relevant_indices)
                        else -1
                    ),
                    "source_text": chunk,
                }

    if not best_match or best_match["score"] <= 0:
        best_chunk = relevant_chunks[0].strip()
        if len(best_chunk) > 280:
            best_chunk = best_chunk[:277].rstrip() + "..."
        return {
            "answer": best_chunk if best_chunk else None,
            "score": 0.15 if best_chunk else 0.0,
            "chunk_index": relevant_indices[0] if relevant_indices else -1,
            "source_text": relevant_chunks[0] if relevant_chunks else "",
        }

    normalized_score = min(0.89, 0.25 + (best_match["score"] / 6))
    best_match["score"] = normalized_score
    return best_match


def _extract_query_terms(question: str) -> List[str]:
    """Extract meaningful search terms from the question."""
    stop_words = {
        "what", "how", "why", "when", "where", "who", "is", "are", "the", "a",
        "an", "and", "or", "in", "at", "to", "for", "of", "by", "on", "does",
        "do", "did", "was", "were", "can", "could", "should", "would", "about",
        "from", "with", "tell", "me", "explain",
    }
    return [
        word
        for word in re.findall(r"[A-Za-z0-9]+", question.lower())
        if word not in stop_words and len(word) > 2
    ]


def _score_text_match(question_terms: List[str], text: str) -> float:
    """Simple relevance score based on keyword overlap and phrase presence."""
    if not text.strip():
        return 0.0

    lowered = text.lower()
    if not question_terms:
        return 0.1 if lowered else 0.0

    unique_terms = set(question_terms)
    overlap = sum(1 for term in unique_terms if term in lowered)
    density_bonus = overlap / max(1, len(unique_terms))
    exact_bonus = 1.5 if " ".join(question_terms[:3]) in lowered and len(question_terms) >= 3 else 0
    return overlap + density_bonus + exact_bonus


def _clean_answer_text(text: str) -> str:
    """
    Clean up answer text by removing markdown formatting and extra whitespace.
    """
    # Remove markdown bold/italic markers
    text = text.replace('**', '').replace('__', '').replace('*', '').replace('_', '')
    # Remove markdown headers
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    # Clean up excessive whitespace
    text = ' '.join(text.split())
    return text
