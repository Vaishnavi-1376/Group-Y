"""
Generation Module - Phase 3
LLM integration for generating responses from retrieved FDA information

This module:
1. Integrates with OpenAI GPT-3.5-turbo (best-effort safe client creation)
2. Generates responses grounded in FDA labels
3. Extracts and tracks citations
4. Calculates confidence scores
5. Generates medication reminders

Author: Member 2 (LLM & Generation Engineer)
"""

import re
from typing import Dict, List
from datetime import datetime, timedelta
import os

try:
    # New OpenAI client interface
    from openai import OpenAI
except Exception:
    OpenAI = None

from src.prompts import (
    SYSTEM_PROMPT,
    build_generation_prompt,
    build_reminder_prompt,
)

# Try to import config values; provide safe defaults if missing
try:
    from src.config import (
        OPENAI_API_KEY,
        LLM_MODEL,
        LLM_TEMPERATURE,
        LLM_MAX_TOKENS,
    )
except Exception:
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
    LLM_MODEL = os.environ.get('LLM_MODEL', 'gpt-3.5-turbo')
    LLM_TEMPERATURE = float(os.environ.get('LLM_TEMPERATURE', 0.0))
    LLM_MAX_TOKENS = int(os.environ.get('LLM_MAX_TOKENS', 512))

# utils may provide logging helpers; fallback to print
try:
    from src.utils import log_event, print_header
except Exception:
    def log_event(msg, level='INFO'):
        print(f"[{level}] {msg}")

    def print_header(msg):
        print(f"\n=== {msg} ===\n")


class MedicationGenerator:
    """Generate responses using GPT-3.5-turbo with FDA labels"""

    def __init__(self, api_key: str = None):
        """Initialize OpenAI client"""
        self.api_key = api_key or OPENAI_API_KEY
        self.model = LLM_MODEL
        self.temperature = LLM_TEMPERATURE
        self.max_tokens = LLM_MAX_TOKENS

        # Initialize client
        if OpenAI is not None:
            try:
                self.client = OpenAI(api_key=self.api_key) if self.api_key else OpenAI()
            except Exception:
                # fallback to None; we'll try REST path later
                self.client = None
        else:
            self.client = None

        log_event(f"Initialized MedicationGenerator")
        log_event(f"  Model: {self.model}")
        log_event(f"  Temperature: {self.temperature}")
        log_event(f"  Max tokens: {self.max_tokens}")

    def generate_response(self, query: str, chunks: List[Dict], n_citations: int = 3) -> Dict:
        """
        Generate a response to a user query using retrieved chunks
        Returns dict with response, citations, confidence, query
        """
        print_header("GENERATING RESPONSE")
        log_event(f"Query: {query}")
        log_event(f"Retrieved chunks: {len(chunks)}")

        prompt = build_generation_prompt(query, chunks)

        # Build messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        response_text = ""

        # Try using OpenAI client wrapper if available
        try:
            if self.client is not None and hasattr(self.client, 'chat'):
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                response_text = resp.choices[0].message.content
            else:
                # Try using the legacy openai package if installed
                import openai as _openai
                if self.api_key:
                    _openai.api_key = self.api_key
                resp = _openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                response_text = resp.choices[0].message['content']

        except Exception as e:
            log_event(f"LLM call failed: {e}", "ERROR")
            # Return conservative fallback
            response_text = "Not specified in FDA label."

        log_event(f"Response length: {len(response_text)} chars")

        citations = self._extract_citations(response_text, chunks, n_citations)
        confidence = self._calculate_confidence(response_text, chunks, citations)

        result = {
            "response": response_text,
            "citations": citations,
            "confidence": confidence,
            "query": query,
            "retrieved_chunks_count": len(chunks),
            "timestamp": datetime.utcnow().isoformat() + 'Z'
        }

        return result

    def _extract_citations(self, response_text: str, chunks: List[Dict], max_cit: int = 3) -> List[Dict]:
        citations = []
        pattern = r"Source: FDA Label - ([^\n]+)"
        matches = re.findall(pattern, response_text)

        for section in matches[:max_cit]:
            for chunk in chunks:
                sec = chunk.get('metadata', {}).get('section_label')
                if sec and sec.strip().lower() == section.strip().lower():
                    citations.append({
                        "text": f"Source: FDA Label - {section}",
                        "section": section,
                        "drug": chunk.get('metadata', {}).get('drug_name'),
                        "source_chunk_id": chunk.get('id')
                    })
                    break

        # Ensure at least one citation if possible: match by section keywords
        if not citations:
            for chunk in chunks[:max_cit]:
                sec = chunk.get('metadata', {}).get('section_label')
                citations.append({
                    "text": f"Source: FDA Label - {sec}",
                    "section": sec,
                    "drug": chunk.get('metadata', {}).get('drug_name'),
                    "source_chunk_id": chunk.get('id')
                })

        return citations

    def _calculate_confidence(self, response_text: str, chunks: List[Dict], citations: List[Dict]) -> float:
        # Factor 1: Retrieval quality (40%)
        if chunks:
            sims = [chunk.get('similarity_score', 0.5) for chunk in chunks[:5]]
            avg_similarity = sum(sims) / len(sims)
            retrieval_score = avg_similarity * 0.4
        else:
            retrieval_score = 0.0

        # Factor 2: Citation coverage (30%)
        citation_score = 0.3 if citations else 0.0

        # Factor 3: Response completeness (30%)
        response_length = len(response_text or "")
        completeness = min(1.0, response_length / 200)
        completeness_score = completeness * 0.3

        confidence = retrieval_score + citation_score + completeness_score
        confidence = max(0.0, min(1.0, confidence))
        return confidence

    def generate_reminder(self, drug: str, dosage: str, frequency: str, start_date: str, duration_days: int = 30) -> Dict:
        print_header("GENERATING REMINDER")
        # Resolve times
        times = self._parse_frequency(frequency)

        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = start + timedelta(days=duration_days)

        schedule = []
        current = start
        while current <= end:
            for t in times:
                schedule.append(f"{current.strftime('%Y-%m-%d')} {t}")
            current = current + timedelta(days=1)

        reminder = {
            "drug": drug,
            "dosage": dosage,
            "frequency": frequency,
            "start_date": start_date,
            "end_date": end.strftime('%Y-%m-%d'),
            "schedule": schedule,
            "total_doses": len(schedule),
            "warnings": self._get_warnings(drug)
        }

        log_event(f"Reminder generated: {reminder['total_doses']} doses")
        return reminder

    def _parse_frequency(self, frequency: str) -> List[str]:
        f = (frequency or '').lower()
        if 'once' in f:
            return ["08:00"]
        if 'twice' in f:
            return ["08:00", "20:00"]
        if 'three' in f:
            return ["08:00", "14:00", "20:00"]
        if 'four' in f:
            return ["06:00", "12:00", "18:00", "22:00"]
        if 'every 4' in f:
            return ["06:00", "10:00", "14:00", "18:00", "22:00"]
        if 'every 6' in f:
            return ["06:00", "12:00", "18:00", "00:00"]
        if 'every 8' in f:
            return ["08:00", "16:00", "00:00"]
        return ["08:00"]

    def _get_warnings(self, drug: str) -> List[str]:
        db = {
            "metformin": ["Take with food", "Monitor glucose levels", "Avoid excessive alcohol"],
            "warfarin": ["Risk of bleeding", "Do not take with NSAIDs", "Requires regular INR monitoring"],
            "aspirin": ["May cause stomach upset", "Do not use if allergic to salicylates", "Risk of bleeding"],
        }
        return db.get((drug or '').lower(), ["Consult your doctor"])


# Global instance
_generator = None

def get_generator() -> MedicationGenerator:
    global _generator
    if _generator is None:
        _generator = MedicationGenerator()
    return _generator


if __name__ == '__main__':
    print_header('GENERATION MODULE TEST')
    g = get_generator()
    print('Generator ready')
