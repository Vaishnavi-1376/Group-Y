# Generation API Documentation

## Overview
The generation module creates responses from retrieved FDA label chunks.

## Main Functions

### `generate_response(query, chunks, n_citations=3)`

Generate a response to a user query.

**Parameters:**
- `query` (str): User's question
- `chunks` (List[Dict]): Retrieved FDA chunks from Member 1
- `n_citations` (int): Max citations to extract

**Returns:**
```json
{
  "response": "Aspirin is used to relieve pain...",
  "citations": [
    {
      "text": "Source: FDA Label - Indications & Usage",
      "section": "Indications & Usage",
      "drug": "aspirin",
      "source_chunk_id": "aspirin_indications_0"
    }
  ],
  "confidence": 0.92,
  "query": "What is aspirin used for?",
  "retrieved_chunks_count": 10,
  "timestamp": "2025-10-25T10:35:56Z"
}
```

**Example Usage:**
```python
from src.generation import get_generator
from src.retrieval import get_retriever

retriever = get_retriever()
generator = get_generator()

# Get chunks
results = retriever.retrieve_chunks("What is metformin?", n_results=10)

# Generate response
response = generator.generate_response(
    "What is metformin?",
    results['chunks']
)

print(response['response'])
print(f"Confidence: {response['confidence']}")
```

### `generate_reminder(drug, dosage, frequency, start_date, duration_days)`

Generate a medication reminder schedule.

**Parameters:**
- `drug` (str): Drug name
- `dosage` (str): Dosage (e.g., "500mg")
- `frequency` (str): How often (e.g., "twice daily")
- `start_date` (str): YYYY-MM-DD format
- `duration_days` (int): Number of days

**Returns:**
```json
{
  "drug": "metformin",
  "dosage": "500mg",
  "frequency": "twice daily",
  "start_date": "2025-10-25",
  "end_date": "2025-11-24",
  "schedule": ["2025-10-25 08:00", "2025-10-25 20:00", ...],
  "total_doses": 60,
  "warnings": ["Take with food", "Monitor glucose"]
}
```

**Example Usage:**
```python
reminder = generator.generate_reminder(
    drug="metformin",
    dosage="500mg",
    frequency="twice daily",
    start_date="2025-10-25",
    duration_days=30
)

print(f"Total doses: {reminder['total_doses']}")
print(f"Schedule: {reminder['schedule'][:3]}")
```

## Integration with Member 3 (Safety)

Member 3 will receive:
- `response['response']`: The text response
- `response['confidence']`: Confidence score (0-1)
- `response['citations']`: List of citations

Member 3 will use these to:
1. Check if confidence < 0.60 (hallucination detection)
2. Validate against FDA dosage limits
3. Check for dangerous drug interactions

## Key Points

- Always cite sources: "Source: FDA Label - [Section]"
- Confidence > 0.75 = high quality response
- Confidence < 0.60 = risky (Member 3 will flag)
- "Not specified in FDA label" = conservative approach
- All responses must be grounded in chunks
