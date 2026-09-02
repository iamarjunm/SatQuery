# SatQuery

SatQuery is an AI-powered natural-language assistant for satellite imagery.

Instead of making users choose different remote-sensing models, they simply ask a question like:
- “What changed between these two satellite images?”
- “Find the water bodies.”
- “Describe the land cover.”

SatQuery’s AI controller understands the query, automatically selects and chains the appropriate specialist models (VQA, grounding, change detection, optical + SAR analysis), and returns the answer with visual evidence on the satellite image/map.

**In one line:**
SatQuery lets anyone analyze satellite imagery by simply asking questions in natural language, while AI handles the underlying remote-sensing models automatically.

## Project Structure

This repository is split into three main components:

- **[`fronted/`](./fronted/)**: The React/Next.js frontend application. It provides the premium user interface, image viewer, and chat interaction for users.
- **[`backend/`](./backend/)**: The Node.js Express backend. It acts as the intelligent router that receives natural language queries, decides which AI models to call, and returns a unified response to the frontend.
- **[`AI/`](./AI/)**: The core machine learning and AI inference modules. It handles the actual execution of the specialist remote-sensing models (VQA, change detection, grounding).

For more details on how to run each component, refer to their respective READMEs.
