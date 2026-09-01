const express = require('express');
const router = express.Router();

// GET /result/:id
// Mocks retrieving a previously stored result by ID
router.get('/:id', (req, res) => {
  const { id } = req.params;
  
  // Return a mock unified result
  res.json({
    query_id: id,
    status: 'success',
    task: 'vqa',
    result: {
      summary: 'The image contains buildings, roads and vegetation.'
    },
    statistics: {
      processing_time_ms: 1200
    }
  });
});

module.exports = router;
