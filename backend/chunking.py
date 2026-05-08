import re

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list:
    """
    Split text into overlapping chunks of roughly `chunk_size` characters, respecting sentence boundaries.
    Overlapping chunks help ensure context is preserved across chunk boundaries.
    
    Args:
        text: The text to chunk
        chunk_size: Target size for each chunk in characters
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    # Split into sentences using a robust regex pattern
    # Handles cases like "Mr. Smith", "Dr. Johnson", etc.
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Calculate length of this sentence
        sentence_len = len(sentence)
        
        # If adding this sentence exceeds chunk_size and we already have some text, start a new chunk
        if current_length + sentence_len > chunk_size and current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)
            
            # Start a new chunk with overlap
            # Add the last few sentences to the next chunk for context
            if len(current_chunk) > 1:
                overlap_sentences = []
                overlap_length = 0
                for i in range(len(current_chunk) - 1, -1, -1):
                    sent = current_chunk[i]
                    if overlap_length + len(sent) <= overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_length += len(sent) + 1  # +1 for space
                    else:
                        break
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_chunk) + len(current_chunk) - 1
            else:
                current_chunk = [sentence]
                current_length = sentence_len
        else:
            current_chunk.append(sentence)
            current_length += sentence_len + 1  # +1 for space
    
    # Add the last chunk if it has any content
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    # Filter out very short chunks (less than 50 characters)
    chunks = [chunk for chunk in chunks if len(chunk) >= 50]
    
    # If no chunks were created (e.g., empty text), return an empty list
    return chunks if chunks else []
