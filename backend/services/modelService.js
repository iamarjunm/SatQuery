/**
 * This service mocks the calls to the actual AI models.
 * It returns responses that exactly match the provided JSON contracts.
 */

// Mock VQA Model
exports.runVQA = async (imageId, question) => {
  // Simulate processing time
  await new Promise(resolve => setTimeout(resolve, 800));
  
  return {
    task: "vqa",
    status: "success",
    result: {
      type: "text",
      answer: "The image contains buildings, roads and vegetation."
    }
  };
};

// Mock Grounding Model
exports.runGrounding = async (imageId, query) => {
  // Simulate processing time
  await new Promise(resolve => setTimeout(resolve, 1200));

  return {
    task: "grounding",
    status: "success",
    result: {
      type: "objects",
      count: 3,
      objects: [
        {
          id: "obj_001",
          label: "building",
          bbox: [120, 80, 240, 190],
          confidence: 0.93
        },
        {
          id: "obj_002",
          label: "building",
          bbox: [300, 120, 410, 250],
          confidence: 0.91
        },
        {
          id: "obj_003",
          label: "building",
          bbox: [500, 200, 620, 330],
          confidence: 0.88
        }
      ]
    }
  };
};

// Mock Change Detection Model
exports.runChangeDetection = async (imageBefore, imageAfter) => {
  // Simulate processing time
  await new Promise(resolve => setTimeout(resolve, 1500));

  return {
    task: "change_detection",
    status: "success",
    result: {
      type: "change_regions",
      change_count: 2,
      changes: [
        {
          id: "change_001",
          label: "change",
          bbox: [120, 80, 260, 200],
          confidence: 0.92
        },
        {
          id: "change_002",
          label: "change",
          bbox: [400, 200, 520, 320],
          confidence: 0.87
        }
      ]
    }
  };
};
