# SatQuery Frontend

This is the Next.js React frontend for the SatQuery application. It provides a premium, highly responsive user interface for analyzing satellite imagery using natural language.

## Features

- **Component Architecture**: Built using modular React components (`Sidebar`, `Header`, `ImageViewer`, `QueryInput`, `ResultsPanel`).
- **Dynamic Image Viewer**: Supports interactive viewing modes including split-screen and slider comparisons for temporal change detection.
- **AI Analysis Visualization**: Animates processing steps and renders bounding boxes directly over satellite imagery based on unified backend responses.
- **Glassmorphism Aesthetics**: Utilizes a dark-mode theme with sleek gradients, Tailwind CSS, and `motion/react` animations for a professional feel.

## Integration

The frontend is designed to interact with the Express backend via the `POST /query` endpoint. It expects a **Unified Result JSON** contract which it seamlessly parses into visual evidence (summary text, confidence scores, and bounding boxes).

*(Note: In the current prototype phase, the frontend may still utilize mock logic internally if not fully wired to the backend URL).*

## How to Run

1. Navigate to the frontend directory:
   ```bash
   cd fronted
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server (runs on `http://localhost:3000` by default):
   ```bash
   npm run dev
   ```
