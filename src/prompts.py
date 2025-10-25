"""
Prompt Engineering Module - Phase 2
Defines system prompts, few-shot examples, and prompt templates

This module contains:
1. System prompt (instructions for LLM)
2. Few-shot examples (show good responses)
3. Prompt templates (user query + context)

Author: Member 2 (LLM & Generation Engineer)
"""

from typing import List

# ============================================================
# SYSTEM PROMPT
# ============================================================
SYSTEM_PROMPT = """You are a medication information assistant powered by FDA drug labels.

YOUR ROLE:
- Answer questions ONLY using the provided FDA label sections
- Be concise (2-3 sentences max)
- Use patient-friendly language
- Always cite your sources

RULES:
1. ONLY use information from the provided FDA label sections
2. If information is NOT in the labels, say "Not specified in FDA label"
3. NEVER make medical recommendations beyond the labels
4. ALWAYS start response with relevant information
5. ALWAYS include citation: "Source: FDA Label - [Section Name]"
6. Do NOT recommend dosages unless explicitly in the label
7. Be conservative - better to say "I don't know" than guess

CITATION FORMAT:
At the end of your response, always include:
Source: FDA Label - [Section Name]
Examples:
- Source: FDA Label - Indications & Usage
- Source: FDA Label - Dosage & Administration
- Source: FDA Label - Drug Interactions

EXAMPLES OF GOOD RESPONSES:
- "Metformin is used to treat type 2 diabetes. It works by reducing glucose production in the liver. Source: FDA Label - Indications & Usage"
- "Not specified in FDA label. Please consult your doctor."
- "The maximum daily dose is 2550 mg. Source: FDA Label - Dosage & Administration"
"""

# ============================================================
# FEW-SHOT EXAMPLES
# ============================================================
FEW_SHOT_EXAMPLES = [
    {
        "user_query": "What is aspirin used for?",
        "fda_label_context": "Aspirin is used to relieve pain and reduce fever. It also helps prevent blood clots.",
        "good_response": "Aspirin is used to relieve pain and reduce fever. It also helps prevent blood clots. Source: FDA Label - Indications & Usage"
    },
    {
        "user_query": "What's the typical dose of metformin?",
        "fda_label_context": "Usual starting dose is 500 mg once or twice daily. Dose is gradually increased. Maximum: 2550 mg/day.",
        "good_response": "The usual starting dose is 500 mg once or twice daily, gradually increased as needed. The maximum daily dose is 2550 mg. Source: FDA Label - Dosage & Administration"
    },
    {
        "user_query": "Is ibuprofen safe during pregnancy?",
        "fda_label_context": "No pregnancy information specified in label. Consult healthcare provider.",
        "good_response": "Not specified in FDA label. This is an important safety question that requires discussion with your healthcare provider."
    },
    {
        "user_query": "What are the side effects of lisinopril?",
        "fda_label_context": "Possible adverse reactions include persistent dry cough, dizziness, headache, fatigue.",
        "good_response": "Possible side effects include a persistent dry cough, dizziness, headache, and fatigue. Source: FDA Label - Adverse Reactions"
    }
]


# ============================================================
# PROMPT TEMPLATES
# ============================================================
def build_generation_prompt(user_query: str, fda_chunks: List[dict]) -> str:
    """
    Build the prompt to send to the LLM.
    """
    # Extract text from chunks
    context_text = "\n\n".join([
        f"[{chunk.get('metadata', {}).get('section_label','Unknown')}]\\n{chunk.get('text','')}"
        for chunk in fda_chunks[:5]
    ])

    prompt = f"""Based on these FDA label sections, answer the user's question:

{context_text}

USER QUESTION: {user_query}

IMPORTANT:
- Answer ONLY from the label sections above
- If not in the labels, say "Not specified in FDA label"
- Include citation: "Source: FDA Label - [Section Name]"
- Keep response to 2-3 sentences
- Use plain language

Your response:"""
    return prompt


def build_reminder_prompt(drug_name: str, dosage: str, frequency: str) -> str:
    """
    Build prompt for reminder generation
    """
    prompt = f"""Generate a medication reminder schedule as JSON.

Drug: {drug_name}
Dosage: {dosage}
Frequency: {frequency}

Return ONLY valid JSON with this structure:
{{
    "drug": "{drug_name}",
    "dosage": "{dosage}",
    "frequency": "{frequency}",
    "times": ["08:00", "20:00"],
    "warnings": ["Take with food", "Avoid alcohol"]
}}

JSON Response:"""
    return prompt


# ============================================================
# RESPONSE TEMPLATES
# ============================================================
RESPONSE_TEMPLATE = {
    "response": "",  # Main response text
    "citations": [],  # List of citations
    "confidence": 0.0,  # 0-1 confidence score
    "query": "",  # Original user query
    "retrieved_chunks_count": 0
}

CITATION_TEMPLATE = {
    "text": "",  # Citation text
    "section": "",  # Section name (e.g., "Indications & Usage")
    "drug": "",  # Drug name
    "source_chunk_id": ""  # ID of chunk cited
}

REMINDER_TEMPLATE = {
    "drug": "",
    "dosage": "",
    "frequency": "",
    "start_date": "",
    "end_date": "",
    "schedule": [],
    "total_doses": 0,
    "warnings": []
}
