import re
from typing import List

# List of common stopwords to filter out
STOPWORDS = {
    "this", "that", "with", "from", "have", "been", "were", "they",
    "their", "what", "when", "which", "there", "about", "would",
    "could", "should", "into", "also", "more", "some", "than",
    "then", "just", "will"
}

def extract_keywords(text: str) -> List[str]:
    """
    Extract top 5 keywords from text by splitting on whitespace,
    filtering out short tokens and common stopwords.
    Returns a list of unique keywords (lowercased).
    """
    # Split on whitespace and normalize
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    
    # Filter: length >= 4 and not in stopwords
    filtered = [w for w in words if len(w) >= 4 and w not in STOPWORDS]
    
    # Get unique words and return top 5 (or all if less)
    unique = list(dict.fromkeys(filtered))  # preserve order but unique
    return unique[:5]

def highlight_keywords(source_text: str, keywords: List[str]) -> str:
    """
    Wrap each keyword (case-insensitive) in <mark> tags with a yellow background.
    Uses re.sub with a lambda to replace matches.
    """
    if not keywords:
        return source_text
    
    # Build a regex pattern that matches any of the keywords (word boundaries)
    # Escape keywords to handle special regex characters
    escaped_keywords = [re.escape(kw) for kw in keywords]
    pattern = r'\b(' + '|'.join(escaped_keywords) + r')\b'
    
    # Replacement function: wrap match in <mark> tag
    def repl(match):
        return f"<mark style='background-color: #FFFF00'>{match.group(0)}</mark>"
    
    # Apply replacement, case-insensitive
    highlighted = re.sub(pattern, repl, source_text, flags=re.IGNORECASE)
    return highlighted