const express = require('express');
const router = express.Router();
const modelService = require('../services/modelService');

// POST /vqa
router.post('/vqa', async (req, res) => {
  try {
    const { image_id, question } = req.body;
    if (!image_id || !question) {
      return res.status(400).json({ status: 'error', error: { code: 'INVALID_QUERY', message: 'Missing image_id or question' } });
    }
    const result = await modelService.runVQA(image_id, question);
    res.json(result);
  } catch (error) {
    res.status(500).json({ status: 'error', error: { code: 'MODEL_FAILED', message: 'VQA model failed' } });
  }
});

// POST /grounding
router.post('/grounding', async (req, res) => {
  try {
    const { image_id, query } = req.body;
    if (!image_id || !query) {
      return res.status(400).json({ status: 'error', error: { code: 'INVALID_QUERY', message: 'Missing image_id or query' } });
    }
    const result = await modelService.runGrounding(image_id, query);
    res.json(result);
  } catch (error) {
    res.status(500).json({ status: 'error', error: { code: 'MODEL_FAILED', message: 'Grounding model failed' } });
  }
});

// POST /change-detection
router.post('/change-detection', async (req, res) => {
  try {
    const { image_before, image_after } = req.body;
    if (!image_before || !image_after) {
      return res.status(400).json({ status: 'error', error: { code: 'INVALID_QUERY', message: 'Missing image_before or image_after' } });
    }
    const result = await modelService.runChangeDetection(image_before, image_after);
    res.json(result);
  } catch (error) {
    res.status(500).json({ status: 'error', error: { code: 'MODEL_FAILED', message: 'Change detection model failed' } });
  }
});

module.exports = router;
