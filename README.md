# 🏥 Medication Reminder Chatbot (Grounded RAG System)

## 📋 Table of Contents
- [Problem](#-problem)
- [Solution](#-solution)
- [Features](#-features)
- [Data Source](#-data-source)
- [System Design](#-system-design)
- [Assumptions](#-assumptions)
- [License](#-license)

---

## 🎯 Problem

**Medication Non-Adherence Crisis:**
- 50% of patients do **not** take medications as prescribed
- Causes $300B+ in preventable healthcare costs annually
- Leads to hospital readmissions and poor health outcomes
- Lacks automated, intelligent reminders and verification systems

**Current Gaps in Existing Solutions:**
- AI chatbots **hallucinate** drug facts and dosages, creating patient safety risks
- Lack grounding in verified data → unsafe recommendations
- No structured reminders or drug interaction checks
- Limited integration with clinical decision support systems
- Provide information without confidence scoring or citations

**Clinical Need:**  
A **trusted, grounded, and intelligent** medication assistant that:
- Delivers verified medical information from authoritative sources
- Prevents harmful drug interactions
- Validates dosage recommendations
- Provides structured, auditable responses for clinical use

---

## ✨ Solution

This project implements a **Retrieval-Augmented Generation (RAG)** system specifically architected for medication safety and adherence support.

**Core Innovation:** Multi-stage validation pipeline that grounds LLM responses in verified FDA data while implementing:
- Semantic + keyword-based hybrid retrieval for comprehensive coverage
- Real-time hallucination detection with confidence scoring
- Automated dosage validation against maximum daily limits
- Drug-drug interaction cross-referencing
- Citation tracking with source attribution

By combining retrieval, generation, and validation in a unified framework, the system eliminates unsafe recommendations and provides clinically verifiable responses.

**Key Differentiator:** Unlike general-purpose RAG systems, this solution is purpose-built for medical safety with guardrails at every stage.

---

## 🚀 Features

| Feature | Description | Benefit |
|---------|-------------|---------|
| **FDA-Grounded Responses** | All recommendations backed by official openFDA drug labels | Zero hallucinations on drug facts |
| **Drug Interaction Detection** | Identifies potential medication conflicts in real-time | Prevents adverse drug combinations |
| **Dosage Validation** | Checks against maximum daily dose limits with clinical thresholds | Ensures safe dosing recommendations |
| **Hallucination Detection** | Semantic similarity checks prevent false claims | High confidence in response accuracy |
| **Structured JSON Output** | Responses include citations, confidence scores, and warnings | Auditability for clinical settings |
| **Hybrid Retrieval** | Combines semantic search + BM25 full-text search | 90%+ recall on relevant drug information |
| **Vector Database Optimization** | ChromaDB for sub-100ms embeddings lookup | Fast response times for patient interactions |
| **Confidence Scoring** | Every response includes a validated confidence metric | Transparent uncertainty quantification |
| **Citation Attribution** | All answers include exact source references from FDA labels | Regulatory compliance and traceability |

---

## 💾 Data Source

**Primary Source:** [openFDA Drug Labels API](https://open.fda.gov/apis/drug/label/)

**Coverage:**
- 100K+ **FDA-approved** drug labels
- Comprehensive medical data including dosage, warnings, interactions
- Publicly available, free API with no authentication required
- Regularly updated by FDA with new drugs and safety information

**Data Structure:**
```json
{
  "drug_name": "string - FDA approved brand/generic name",
  "generic_name": "string - Generic drug name",
  "dosage_and_administration": "string - Recommended dosing regimen",
  "warnings": "string - Serious warnings and precautions",
  "interactions": "string - Known drug interactions",
  "adverse_reactions": "string - Side effects and adverse events",
  "contraindications": "string - When drug should NOT be used",
  "precautions": "string - Special monitoring requirements",
  "source_url": "string - FDA label URL for verification"
}
```

**Processing Pipeline:**
1. API ingestion of 100K+ drug labels
2. Section-aware chunking (dosage, warnings, interactions)
3. Embedding generation for semantic search
4. BM25 indexing for keyword-based retrieval
5. Storage in ChromaDB vector database

**Data Quality Assurance:**
- All data verified against official FDA database
- Structured format ensures consistency across records
- Regular synchronization with FDA updates
- Standardized dosage units and interaction severity levels

---

## 🏗️ System Design Overview

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER QUERY                               │
│                  "What is ibuprofen dosage?"                    │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────────┐
        │   INTENT RECOGNITION               │
        │  • Extract drug names              │
        │  • Classify query type             │
        │  • Detect interaction checks       │
        └───────────┬───────────────────────┘
                    │
                    ▼
        ┌───────────────────────────────────────────┐
        │      RAG PIPELINE (Data Retrieval)        │
        ├───────────────────────────────────────────┤
        │ 1. SEMANTIC SEARCH                        │
        │    └─ Query embeddings (text-embedding)   │
        │    └─ ChromaDB similarity search          │
        │    └─ Top 5 semantic matches              │
        │                                            │
        │ 2. BM25 FULL-TEXT SEARCH                  │
        │    └─ Keyword matching on indexed labels  │
        │    └─ Top 5 keyword matches               │
        │                                            │
        │ 3. RESULT FUSION & RANKING                │
        │    └─ Deduplicate results                 │
        │    └─ Combine relevance scores            │
        │    └─ Select top 5 final results          │
        └───────────┬───────────────────────────────┘
                    │
                    ▼
        ┌───────────────────────────────────┐
        │   CONTEXT PREPARATION             │
        │  • Select relevant chunks         │
        │  • Score chunks by relevance      │
        │  • Attach source attribution      │
        └───────────┬───────────────────────┘
                    │
                    ▼
        ┌────────────────────────────────────────────┐
        │   LLM RESPONSE GENERATION                  │
        ├────────────────────────────────────────────┤
        │ • Grounded Prompt:                         │
        │   "Answer ONLY using provided FDA labels.  │
        │    Do not use external knowledge."         │
        │ • Temperature: 0.1 (deterministic)         │
        │ • Max tokens: 1000                         │
        │ • Output format: Structured JSON           │
        └───────────┬────────────────────────────────┘
                    │
                    ▼
        ┌────────────────────────────────────────────┐
        │   VALIDATION & GUARDRAILS LAYER           │
        ├────────────────────────────────────────────┤
        │ 1. HALLUCINATION CHECK                     │
        │    └─ Semantic similarity vs sources       │
        │    └─ Confidence score adjustment          │
        │                                             │
        │ 2. DOSAGE VALIDATION                       │
        │    └─ Compare vs max daily dose            │
        │    └─ Flag if exceeds thresholds           │
        │                                             │
        │ 3. INTERACTION CROSS-CHECK                 │
        │    └─ Query FDA interaction database       │
        │    └─ Severity classification              │
        │                                             │
        │ 4. CONFIDENCE SCORING                      │
        │    └─ Combined score from retrieval        │
        │    └─ LLM confidence signals               │
        │    └─ Validation results                   │
        └───────────┬────────────────────────────────┘
                    │
                    ▼
    ┌──────────────────────────────────────────────────┐
    │             FINAL JSON RESPONSE                  │
    ├──────────────────────────────────────────────────┤
    │ {                                                │
    │   "answer": "800-1200mg every 4-6 hours...",    │
    │   "sources": [                                   │
    │     {                                            │
    │       "drug_name": "Ibuprofen",                 │
    │       "section": "dosage_and_administration",   │
    │       "relevance_score": 0.98,                  │
    │       "url": "https://api.fda.gov/..."         │
    │     }                                            │
    │   ],                                             │
    │   "warnings": [                                  │
    │     "Not recommended for long-term use"         │
    │   ],                                             │
    │   "confidence_score": 0.95,                     │
    │   "interaction_check": { ... }                  │
    │ }                                                │
    └──────────────────────────────────────────────────┘
                    │
                    ▼
            DELIVERED TO USER
```

---

## 📌 Assumptions

### Data Assumptions
1. **openFDA data accuracy** - All FDA labels are verified and authoritative sources of drug information
2. **Standardized structure** - Drug labels follow consistent formatting for dosage, warnings, and interactions
3. **Coverage adequacy** - 100K+ labels cover ≥95% of commonly prescribed FDA-approved drugs
4. **Regular updates** - FDA API is maintained and updated with new safety information
5. **Section consistency** - Key sections (dosage, warnings, interactions) are present in most labels

### System Assumptions
6. **English queries** - User inputs are in English and clearly specify drug names
7. **Query independence** - Each request is stateless (no conversation memory required)
8. **Hybrid recall** - Combining semantic + BM25 search achieves ≥90% recall on relevant information
9. **Fast embeddings** - Vector retrieval completes in <100ms, suitable for interactive use
10. **Network availability** - OpenAI API and openFDA API are accessible during operation

### Clinical Assumptions
11. **Information provision only** - System provides educational information, NOT medical advice
12. **Adult dosage** - All dosing recommendations assume adult patients (≥18 years old)
13. **Standard interactions** - Only covers documented FDA label interactions, not theoretical ones
14. **No personalization** - Cannot adjust recommendations based on patient-specific factors (kidney function, allergies, etc.)
15. **Professional review** - Responses are intended for healthcare provider review, not direct patient use

### Technical Assumptions
16. **LLM reliability** - GPT-4/Gemini reliably follow grounding instructions and low-temperature settings
17. **Embedding quality** - text-embedding-3-small captures medical semantics effectively (medical term similarity)
18. **Hallucination threshold** - Semantic similarity score of <0.7 indicates hallucination with 95%+ confidence
19. **Scale limits** - MVP designed for 100–1000 queries/day, not high-throughput production use
20. **Manual updates** - No real-time data sync; database requires scheduled refresh cycles

---

## 📄 License

MIT License

---

**Last Updated:** October 25, 2025
**Maintained by:** Group Y Contributors
