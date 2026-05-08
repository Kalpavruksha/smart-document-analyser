import re

def simplify_answer(answer: str) -> str:
    """
    Simplify and refine an answer by:
    - Removing unnecessary citations and markdown
    - Shortening long answers while preserving key information
    - Making it more concise and user-friendly
    - Formatting lists properly
    
    Args:
        answer: The answer text to simplify
        
    Returns:
        Simplified and refined answer text
    """
    if not answer:
        return ""
    
    # Remove citation markers like [1], [2], etc.
    simplified = re.sub(r'\[\d+\]', '', answer)
    
    # Remove markdown formatting
    simplified = simplified.replace('**', '').replace('__', '').replace('*', '').replace('_', '')
    
    # Remove markdown headers
    simplified = re.sub(r'^#+\s+', '', simplified, flags=re.MULTILINE)
    
    # Convert markdown lists to simple text lists
    simplified = re.sub(r'^\s*[-*+]\s+', '• ', simplified, flags=re.MULTILINE)
    
    # Remove extra whitespace
    simplified = ' '.join(simplified.split())
    
    # If answer contains bullet points, preserve them
    if '•' in simplified:
        # Just ensure clean formatting
        lines = simplified.split('•')
        simplified = '\n'.join(['• ' + line.strip() for line in lines if line.strip()])
        if simplified.startswith('• '):
            simplified = simplified[2:]  # Remove leading bullet from first line
    else:
        # For non-list answers, try to extract key information
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', simplified)
        
        # Take the first 2-3 sentences as they usually contain the main point
        if len(sentences) > 1:
            key_sentences = sentences[:min(3, len(sentences))]
            simplified = ' '.join(key_sentences)
    
    # Ensure the answer doesn't exceed a reasonable length (about 500 chars)
    if len(simplified) > 500:
        # Try to cut at a sentence boundary
        sentences = re.split(r'(?<=[.!?])\s+', simplified)
        truncated = ""
        for sent in sentences:
            if len(truncated) + len(sent) + 1 <= 450:
                truncated = (truncated + " " + sent).strip()
            else:
                break
        simplified = truncated + "..." if len(truncated) > 0 else simplified[:497] + "..."
    
    return simplified.strip()
