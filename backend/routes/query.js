const express = require('express');
const router = express.Router();

// POST /query
// The frontend's single entry point. The backend does no routing of its own:
// it forwards the query to the agent controller (controller/server.py), which
// plans it with an LLM, runs the plan against the model services, validates
// every geometry, and returns the answer plus a pixel-space "display" layer
// for the viewer. See docs/json-contracts-v2.md for the result shape.
const CONTROLLER_URL = process.env.CONTROLLER_URL || 'http://127.0.0.1:8080';

router.post('/', async (req, res) => {
  const { query_id, query, image_ids } = req.body || {};

  if (!query || typeof query !== 'string' || !query.trim()) {
    return res.status(400).json({
      query_id: query_id || 'unknown',
      status: 'error',
      error: { code: 'INVALID_QUERY', message: 'query is required.' }
    });
  }

  try {
    const upstream = await fetch(`${CONTROLLER_URL}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, image_ids })
    });
    const payload = await upstream.json();
    if (!upstream.ok) {
      return res.status(upstream.status).json({
        query_id: query_id || 'unknown',
        status: 'error',
        error: { code: 'CONTROLLER_ERROR', message: payload.error || `controller returned ${upstream.status}` }
      });
    }
    // The controller's own status is "ok" or "partial"; both carry a valid answer.
    return res.json({ query_id: query_id || null, ...payload });
  } catch (error) {
    console.error('controller unreachable:', error.message);
    return res.status(502).json({
      query_id: query_id || 'unknown',
      status: 'error',
      error: {
        code: 'CONTROLLER_UNREACHABLE',
        message: `Could not reach the controller at ${CONTROLLER_URL}. Start it with: cd controller && python server.py`
      }
    });
  }
});

// GET /query/images
// The scenes the controller has loaded, for the frontend's site picker.
router.get('/images', async (req, res) => {
  try {
    const upstream = await fetch(`${CONTROLLER_URL}/images`);
    const payload = await upstream.json();
    // Preview paths are relative to the controller; make them absolute so
    // the browser can load them directly.
    payload.images = (payload.images || []).map(img => ({ ...img, preview: `${CONTROLLER_URL}${img.preview}` }));
    return res.json(payload);
  } catch (error) {
    return res.status(502).json({ status: 'error', error: { code: 'CONTROLLER_UNREACHABLE', message: error.message } });
  }
});

module.exports = router;
