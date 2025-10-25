# System Architecture

This document details the architecture of the Medication Reminder Chatbot, a RAG (Retrieval-Augmented Generation) system.

## Core Components

The system is composed of four main microservices/modules, each handled by a team member.



### 1. Member 1: Data & Retrieval
-   **Responsibility:** Indexing all knowledge and retrieving it efficiently.
-   **`data_prep.py`:**
    -   Downloads the FDA drug label dataset.
    -   Parses and cleans the raw data (e.g., `fda_labels.csv`).
    -   Chunks large documents into smaller, semantically meaningful sections (e.g., "Indications", "Dosage", "Warnings").
    -   Saves these chunks as `data/fda_chunks.json`.
-   **`embeddings.py`:**
    -   Loads `data/fda_chunks.json`.
    -   Calls the OpenAI `text-embedding-3-small` API for each chunk.
    -   Saves the chunks *with* their embeddings as `data/fda_chunks_with_embeddings.json`.
-   **`chromadb_setup.py`:**
    -   Initializes a persistent ChromaDB vector store (`chroma_db/`).
    -   Loads `data/fda_chunks_with_embeddings.json` and inserts all documents and their embeddings into the "fda_labels" collection.
-   **`retrieval.py`:**
    -   Provides the main API function: `get_retriever()`.
    -   The retriever object has one key method: `retrieve_chunks(query, n_results=5)`, which embeds the query, searches ChromaDB, and returns the `n_results` most relevant document chunks.

### 2. Member 2: LLM & Generation
-   **Responsibility:** Generating human-readable answers based on retrieved context.
-   **`prompts.py`:**
    -   Contains all prompt templates (e.g., `QA_PROMPT`, `REMINDER_PROMPT`, `CONFIDENCE_SCORE_PROMPT`).
    -   The `QA_PROMPT` explicitly instructs the LLM to answer *only* using the provided context and to cite its sources.
-   **`generation.py`:**
    -   Provides the main API function: `get_generator()`.
    -   `generate_response(query, chunks)`:
        1.  Formats the `QA_PROMPT` with the query and context (chunks).
        2.  Calls the GPT-3.5-Turbo API to get a draft response.
        3.  Calls the LLM *again* with the `CONFIDENCE_SCORE_PROMPT` to make the LLM self-evaluate its response for grounding and confidence (0.0 - 1.0).
        4.  Returns a structured dictionary: `{ "response": "...", "confidence": 0.9, "citations": [...] }`.
    -   `generate_reminder(...)`: Generates a JSON schedule based on user form input.

### 3. Member 3: Safety & Validation
-   **Responsibility:** Ensuring all generated output is safe, accurate, and not harmful.
-   **`guardrails.py`:**
    -   Provides the main API function: `get_guardrails()`.
    -   `check_hallucination(response, chunks)`: Uses an LLM "judge" prompt to check if the `response` is fully supported by the `chunks`. Flags low-confidence responses.
    -   `check_dosage(response, chunks)`: (Simplified) Uses regex and keyword matching to find dosage information in the `response` and checks it against "Dosage" sections in the `chunks` for any contradictions (e.g., exceeding a stated maximum dose).
    -   `check_interaction(drug1, drug2)`:
        1.  Loads a pre-compiled list of dangerous pairs (`data/dangerous_drug_pairs.json`).
        2.  Checks if the (drug1, drug2) pair is in this list.
        3.  *If not found in list,* it performs a retrieval (`retriever.retrieve_chunks(...)`) on both drugs to find their "Drug Interactions" sections.
        4.  Uses an LLM prompt to read both interaction sections and determine if a dangerous interaction is described.
    -   `run_all_safety_checks(...)`: A master function that runs all checks and provides a final safety status (`SAFE`, `WARNING`, `CRITICAL`).

### 4. Member 4: UI & Demo
-   **Responsibility:** Integrating all modules into a user-facing application and proving it works.
-   **`notebooks/03_demo_interface.ipynb`:**
    -   The main demo application.
    -   Uses `ipywidgets` to create an interactive UI with text boxes, buttons, and dropdowns.
    -   **Section 1 (Q&A):** Calls `retriever.retrieve_chunks`, `generator.generate_response`, and `guardrails.run_all_safety_checks` in a single pipeline to display a safe, cited answer.
    -   **Section 2 (Interaction):** Calls `guardrails.check_interaction` on two dropdowns.
    -   **Section 3 (Reminder):** Calls `generator.generate_reminder`.
-   **`evaluation/run_evaluation.py`:**
    -   A standalone script to automatically test the entire system.
    -   Loads `evaluation/test_queries.json`.
    -   Runs the Q&A pipeline for each query and calculates key metrics:
        -   **Retrieval Accuracy:** (Did the correct drug appear in the retrieved chunks?)
        -   **Hallucination Rate:** (% of responses flagged by guardrails or with low confidence)
        -   **Interaction Precision:** (% of known dangerous pairs correctly identified)
    -   Saves the final scores to `evaluation/results/metrics.json`.
-   **Documentation:** `README.md`, `docs/*.md`, `presentation/*.md`.

## Data Flow (Q&A Example)

1.  User types "What is metformin for?" into `03_demo_interface.ipynb`.
2.  UI button click calls `on_ask_click()`.
3.  `retriever.retrieve_chunks(...)` is called.
    -   -> `ChromaDB` returns 5 chunks related to "metformin" and "uses".
4.  `generator.generate_response(...)` is called with the query and chunks.
    -   -> `GPT-3.5-Turbo` returns a text response and a confidence score.
5.  `guardrails.run_all_safety_checks(...)` is called with the response and chunks.
    -   -> All 3 checks pass.
6.  The final, safe response is displayed in the notebook's output cell.