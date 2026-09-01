const express = require('express');
const router = express.Router();
const modelService = require('../services/modelService');

// POST /query
// Main endpoint for the frontend to submit queries
router.post('/', async (req, res) => {
  try {
    const { query_id, query, image_id, image_before, image_after } = req.body;

    if (!query_id || !query) {
      return res.status(400).json({
        query_id: query_id || 'unknown',
        status: 'error',
        error: {
          code: 'INVALID_QUERY',
          message: 'query_id and query are required fields.'
        }
      });
    }

    // --- 1. Query Router Logic ---
    // In a real app, an LLM would parse the query to determine the intent and workflow.
    // For this prototype, we'll use simple keyword matching to route the request.
    
    let intent = '';
    let workflow = [];
    const qLower = query.toLowerCase();

    if (qLower.includes('change') || qLower.includes('newly')) {
      if (qLower.includes('building')) {
         intent = 'multi_step';
         workflow = [
           { step: 1, task: 'change_detection' },
           { step: 2, task: 'grounding', input_from: 'step_1' }
         ];
      } else {
        intent = 'change_detection';
        workflow = [{ step: 1, task: 'change_detection' }];
      }
    } else if (qLower.includes('find') || qLower.includes('locate') || qLower.includes('where')) {
      intent = 'grounding';
      workflow = [{ step: 1, task: 'grounding' }];
    } else {
      intent = 'vqa';
      workflow = [{ step: 1, task: 'vqa' }];
    }

    console.log(`Router Intent: ${intent} for query: "${query}"`);

    // --- 2. Execute Workflow ---
    // We mock the AI models calling based on the determined intent.
    let finalResult = null;
    let visualization = null;

    if (intent === 'vqa') {
      const vqaResult = await modelService.runVQA(image_id, query);
      finalResult = { summary: vqaResult.result.answer };
      visualization = null; // No viz for simple text answer
    } 
    else if (intent === 'grounding') {
      const groundingResult = await modelService.runGrounding(image_id, 'buildings'); // Simple extraction
      finalResult = { summary: `${groundingResult.result.count} objects were detected.` };
      visualization = {
        type: 'bounding_boxes',
        items: groundingResult.result.objects
      };
    }
    else if (intent === 'change_detection') {
      const changeResult = await modelService.runChangeDetection(image_before, image_after);
      finalResult = { summary: `${changeResult.result.change_count} changes were detected.` };
      visualization = {
        type: 'bounding_boxes',
        items: changeResult.result.changes
      };
    }
    else if (intent === 'multi_step') {
      // Execute change detection, then grounding
      const changeResult = await modelService.runChangeDetection(image_before, image_after);
      // In a real scenario, grounding would run ON the changed regions. 
      // For the mock, we'll just return a combined/specific response.
      finalResult = { summary: `3 newly constructed buildings were detected.` };
      visualization = {
        type: 'bounding_boxes',
        items: [
          {
            label: 'new_building',
            bbox: [120, 80, 260, 200],
            confidence: 0.92
          }
        ]
      };
    }

    // --- 3. Unified Result JSON ---
    const response = {
      query_id,
      status: 'success',
      task: intent,
      result: finalResult
    };

    if (visualization) {
      response.visualization = visualization;
    }

    // Optional: Add mock statistics
    response.statistics = {
      objects_detected: visualization?.items?.length || 0,
      changes_detected: intent.includes('change') ? (visualization?.items?.length || 0) : 0,
      processing_time_ms: Math.floor(Math.random() * 2000) + 500
    };

    return res.json(response);

  } catch (error) {
    console.error('Error processing query:', error);
    res.status(500).json({
      query_id: req.body.query_id || 'unknown',
      status: 'error',
      error: {
        code: 'MODEL_FAILED',
        message: 'The selected model could not process the image.'
      }
    });
  }
});

module.exports = router;
