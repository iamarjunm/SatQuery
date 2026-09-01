const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const queryRouter = require('./routes/query');
const textRouter = require('./routes/text');

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3001;

app.use(cors());
app.use(express.json());

// Routes
app.use('/query', queryRouter);
app.use('/text', textRouter);
const modelsRouter = require('./routes/models');
const resultRouter = require('./routes/result');
app.use('/', modelsRouter); // Mounts /vqa, /grounding, /change-detection
app.use('/result', resultRouter);
const testRouter = require('./routes/test');
app.use('/test', testRouter);

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'ok', message: 'SatQuery backend is running' });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({
    status: 'error',
    error: {
      code: 'SERVER_ERROR',
      message: 'An unexpected error occurred.'
    }
  });
});

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
