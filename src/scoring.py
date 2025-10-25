"""
Scoring Module - Confidence Calculation
Calculate how confident we are in our responses

Author: Member 2 (LLM & Generation Engineer)
"""

from typing import Dict, List


def calculate_response_confidence(response_text: str, chunks: List[Dict], has_citations: bool = False, retrieved_count: int = 0) -> float:
    """
    Calculate confidence in response
    Returns 0-1 confidence score
    """
    score = 0.0

    # Factor 1: Retrieval quality (40%)
    if chunks and len(chunks) > 0:
        avg_similarity = sum(chunk.get('similarity_score', 0.5) for chunk in chunks[:5]) / min(5, len(chunks))
        score += avg_similarity * 0.4

    # Factor 2: Response length (20%)
    response_length = len(response_text or "")
    length_score = min(1.0, response_length / 200)
    score += length_score * 0.2

    # Factor 3: Citation presence (20%)
    citation_score = 1.0 if has_citations else 0.3
    score += citation_score * 0.2

    # Factor 4: Chunk count (20%)
    chunk_score = min(1.0, retrieved_count / 10)
    score += chunk_score * 0.2

    return max(0.0, min(1.0, score))


def extract_confidence_indicators(response_text: str) -> Dict:
    indicators = {
        "has_source_citation": "Source: FDA Label" in (response_text or ""),
        "has_uncertainty": any([
            "not specified" in (response_text or "").lower(),
            "consult your doctor" in (response_text or "").lower(),
            "beyond our scope" in (response_text or "").lower()
        ]),
        "response_length": len(response_text or ""),
        "is_too_short": len(response_text or "") < 50,
        "is_too_long": len(response_text or "") > 1000
    }

    return indicators
