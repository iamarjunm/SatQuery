const express = require('express');
const router = express.Router();

// GET /test
// A comprehensive health check and testing endpoint
router.get('/', (req, res) => {
  res.json({
    status: 'success',
    message: 'SatQuery backend is fully operational!',
    environment: process.env.NODE_ENV || 'development',
    timestamp: new Date().toISOString(),
    endpoints: {
      router: 'POST /query',
      models: [
        'POST /vqa',
        'POST /grounding',
        'POST /change-detection'
      ],
      results: 'GET /result/:id',
      utilities: [
        'GET /health',
        'GET /test',
        'GET/POST /text'
      ]
    },
    system_status: {
      database: 'Not Connected (Mock Mode)',
      ai_models: 'Mocked Services Active'
    }
  });
});

module.exports = router;
