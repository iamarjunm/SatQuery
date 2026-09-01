# SatQuery Backend

This is the Node.js Express backend for SatQuery. It serves as the primary router and data pipeline connecting the frontend to the underlying AI remote-sensing models.

## Core Responsibilities

- **AI Router**: Interprets incoming natural language queries and determines the correct AI workflow (`vqa`, `grounding`, `change_detection`, `multi_step`).
- **Standardized Communication**: Exposes a unified API to the frontend so the frontend doesn't need to know the intricacies of every remote-sensing model.
- **Model API Endpoints**: Formats and triggers the specific AI models, then processes their output into a unified JSON format.

## API Endpoints & JSON Contracts

### `POST /query`
This is the primary endpoint used by the frontend to submit any natural language query.

**Request:**
```json
{
  "query_id": "q_001",
  "query": "Find all buildings in this image",
  "image_id": "img_001"
}
```
*For temporal/change queries:*
```json
{
  "query_id": "q_002",
  "query": "What changed between these two images?",
  "image_before": "img_2023",
  "image_after": "img_2026"
}
```

**Unified Response format (Backend -> Frontend):**
The backend always returns a unified structure, regardless of the models used underneath.
```json
{
  "query_id": "q_004",
  "status": "success",
  "task": "multi_step",
  "result": {
    "summary": "3 newly constructed buildings were detected."
  },
  "visualization": {
    "type": "bounding_boxes",
    "items": [
      {
        "label": "new_building",
        "bbox": [120, 80, 260, 200],
        "confidence": 0.92
      }
    ]
  },
  "statistics": {
    "objects_detected": 3,
    "changes_detected": 1,
    "processing_time_ms": 1840
  }
}
```

### AI Model Contracts (Internal)
Internally, the backend routes requests to specific models using these contracts:

**VQA Input:**
```json
{
  "task": "vqa",
  "image_id": "img_001",
  "question": "What is visible in this image?"
}
```
**Grounding Input:**
```json
{
  "task": "grounding",
  "image_id": "img_001",
  "query": "buildings"
}
```
**Change Detection Input:**
```json
{
  "task": "change_detection",
  "image_before": "img_2023",
  "image_after": "img_2026"
}
```

### `GET /text` & `POST /text`
A generic endpoint that echoes and processes text (primarily used for testing).

## How to Run

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the server (runs on `http://localhost:3001` by default):
   ```bash
   npm run dev
   ```
