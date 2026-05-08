import os
from typing import List, Dict, Any
import requests
import re
import json

# OpenRouter configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "AIzaSyC5AjzUwUAkIYqjVtUoLA_tumSHK1nJ-Ps")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "gemini-2.5-flash")
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
    Uses LLM API to extract answers with accurate context processing.

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

    # Pass a large number of chunks as context for Gemini 1.5 Flash
    max_chunks = 200
    used_chunks = chunks[:max_chunks]
    context = "\n\n".join([f"[Chunk {i}]\n{chunk}" for i, chunk in enumerate(used_chunks)])

    # Create a refined prompt for JSON structured output
    system_prompt = """You are an expert document analyst and question answerer. Your task is to:
1. Answer questions accurately based ONLY on the provided context chunks.
2. Provide a well-written, summarized answer in your own words. Do NOT just copy-paste verbatim sentences from the document. Synthesize and summarize the relevant information clearly.
3. If the answer is not available in the context, explicitly state "This information is not available in the provided documents."
4. You MUST return your response as a valid JSON object with the exact following structure:
{
  "answer": "Your detailed answer here...",
  "confidence_score": 0.95,
  "source_chunk_index": 5
}
Notes on JSON fields:
- "answer": String. The actual answer text.
- "confidence_score": Float between 0.0 and 1.0 representing how confident you are that the answer is correct and supported by the text.
- "source_chunk_index": Integer. The index of the chunk (e.g., from [Chunk 5]) that provided the answer. Pick the most relevant one. Use -1 if not found."""

    user_prompt = f"""Please summarize the answer to the following question based on the provided context.

Context:
{context}

Question: {question}

Instructions:
- Provide a brief, synthesized summary of the answer.
- Do NOT simply extract or quote passages directly from the text.
- Use your own words to explain the answer clearly and concisely."""

    # We need to keep the fallback answer in case API fails
    relevant_chunks_data = _get_relevant_chunks(question, chunks, top_k=3)
    fallback_answer = _extract_answer_locally(
        question,
        relevant_chunks_data["chunks"],
        relevant_chunks_data["indices"],
    )

    try:
        if not OPENROUTER_API_KEY:
            return fallback_answer

        # Make API call to OpenRouter/Gemini using requests
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
            "max_tokens": 1000,
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }

        response = requests.post(url, headers=headers, json=data, timeout=60)
        response.raise_for_status()

        result = response.json()
        answer_text = result["choices"][0]["message"]["content"].strip()

        # Clean markdown code block formatting if LLM includes it
        if answer_text.startswith("```json"):
            answer_text = answer_text[7:]
            if answer_text.endswith("```"):
                answer_text = answer_text[:-3]
        elif answer_text.startswith("```"):
            answer_text = answer_text[3:]
            if answer_text.endswith("```"):
                answer_text = answer_text[:-3]
        answer_text = answer_text.strip()

        try:
            json_response = json.loads(answer_text)
            extracted_answer = json_response.get("answer", "")
            score = float(json_response.get("confidence_score", 0.0))
            chunk_index = int(json_response.get("source_chunk_index", -1))
        except json.JSONDecodeError:
            # Fallback if model didn't return valid JSON
            extracted_answer = answer_text
            score = 0.5
            chunk_index = -1

        # Check if the model couldn't find an answer
        if (
            not extracted_answer
            or "cannot find" in extracted_answer.lower()
            or "not available" in extracted_answer.lower()
            or "not found" in extracted_answer.lower()
            or score < 0.1
        ):
            return fallback_answer

        # Clean up the answer: remove markdown formatting if present
        extracted_answer = _clean_answer_text(extracted_answer)
        
        # Safely get the source text
        source_text = ""
        if 0 <= chunk_index < len(chunks):
            source_text = chunks[chunk_index]

        return {
            "answer": extracted_answer,
            "score": score,
            "chunk_index": chunk_index,
            "source_text": source_text
        }

    except requests.exceptions.RequestException as e:
        error_msg = f"API Connection Error: {e}"
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_data = e.response.json()
                if "error" in error_data and "message" in error_data["error"]:
                    error_msg = f"API Error: {error_data['error']['message']}"
            except Exception:
                error_msg = f"API Error: {e.response.text}"
        
        print(error_msg)
        return {
            "answer": f"**System Error**: {error_msg}\n\nFalling back to basic local extraction:\n\n{fallback_answer['answer']}",
            "score": fallback_answer["score"],
            "chunk_index": fallback_answer["chunk_index"],
            "source_text": fallback_answer["source_text"]
        }
    except Exception as e:
        print(f"Error calling LLM API: {e}")
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
    Clean up answer text but preserve newlines and basic formatting
    so that summaries and bullet points are readable.
    """
    # Just strip leading/trailing whitespace
    return text.strip()
