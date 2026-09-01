const express = require('express');
const router = express.Router();

// GET /text
// A simple text endpoint as requested.
router.get('/', (req, res) => {
  res.json({
    status: 'success',
    message: 'This is a standard text endpoint.',
    data: {
      text: 'SatQuery backend is fully operational.',
      timestamp: new Date().toISOString()
    }
  });
});

// POST /text
// Echo back text if provided
router.post('/', (req, res) => {
  const { text } = req.body;
  
  if (!text) {
    return res.status(400).json({
      status: 'error',
      message: 'Text field is required in the request body.'
    });
  }

  res.json({
    status: 'success',
    message: 'Text received and processed.',
    data: {
      original_text: text,
      processed_text: text.toUpperCase(),
      length: text.length
    }
  });
});

module.exports = router;
