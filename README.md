# 🏥 Medication Reminder Chatbot (Grounded RAG System)

## 🎯 Problem

**Medication Non-Adherence Crisis:**
- 50% of patients do **not** take medications as prescribed.  
- Causes $300B+ in preventable healthcare costs annually.  
- Leads to hospital readmissions and poor health outcomes.  

**Current Gaps:**
- AI chatbots **hallucinate** drug facts and dosages.  
- Lack grounding in verified data → unsafe recommendations.  
- No structured reminders or drug interaction checks.  

**Need:**  
A **trusted, grounded, and intelligent** medication assistant that delivers verified medical information and safe reminders.

---

## 💾 Data Link

**Primary Source:** [openFDA Drug Labels API](https://open.fda.gov/apis/drug/label/)

- 100K+ **FDA-approved** drug labels  
- Verified medical data (dosage, warnings, interactions)  
- JSON format, free public API  

**Key Fields:**
`drug_name`, `dosage_and_administration`, `warnings`, `interactions`, `adverse_reactions`, `source_url`

---

## 🏗️ System Design Overview

User Query
│
▼
Intent Recognition
│
▼
RAG Pipeline (Data Retrieval)
• openFDA drug labels
• Section-aware chunking
• Embeddings (OpenAI / Sentence Transformers)
• Vector DB (ChromaDB)
│
▼
LLM Response Generation
• Grounded prompt: "Answer only from FDA labels"
• Low temperature (0.1)
• Structured JSON output with citations
│
▼
Validation & Guardrails
• Hallucination check (semantic similarity)
• Dosage validation (vs max daily dose)
• Interaction detection
• Confidence scoring
│
▼
Final JSON Output

---

**Stack:**  
FastAPI · LangChain · OpenAI Embeddings · ChromaDB · Pydantic · Python

---

## 📐 Assumptions

**Data:**
1. openFDA data is accurate and regularly updated.  
2. Drug labels are standardized and structured.  
3. 100K+ labels cover most FDA-approved drugs.  

**System:**
4. User queries are in English and specify drug names.  
5. Each request is independent (no chat memory).  
6. Dual retrieval (semantic + BM25) achieves ≥90% recall.  

**Clinical:**
7. Provides **information only**, not medical advice.  
8. Assumes **adult dosage** standards (no pediatric/personalized data).  
9. Interactions limited to documented FDA label data.  

**Technical:**
10. text-embedding-3-small captures medical semantics effectively.  
11. GPT-4/Gemini reliably follow grounding instructions.  
12. Designed for small-scale MVP (100–1000 queries).  

---

**⚠️ Disclaimer:**  
This chatbot provides *educational information only*. It is **not** a substitute for professional medical advice. Always consult a licensed healthcare provider.

---

