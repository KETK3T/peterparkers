import express from 'express';
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

// 1. Setup path constants for ES Modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const port = 3001;

// Allow your React frontend (on a different port) to access this server
app.use((req, res, next) => {
    res.header('Access-Control-Allow-Origin', 'http://localhost:5173');
    res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
    next();
});

app.get('/run-python-test', (req, res) => {
    // Determine the path to your Python script (assuming it's at src/backend/MainCamera.py)
    const scriptPath = path.join(__dirname, 'src', 'backend', 'MainCamera.py');

    // Spawn the Python process
    const python = spawn('python', [scriptPath]);
    let dataToSend = '';

    // Collect data from the Python script's stdout
    python.stdout.on('data', (data) => {
        dataToSend += data.toString();
    });

    // Handle when the Python script exits
    python.on('close', (code) => {
        if (code === 0) {
            console.log(`Python script exited successfully.`);
            res.send({ output: dataToSend.trim() });
        } else {
            console.error(`Python script exited with code ${code}`);
            res.status(500).send({ error: 'Python script failed to run.' });
        }
    });

    // Handle errors in running the Python process itself
    python.on('error', (err) => {
        console.error('Failed to start Python process:', err);
        res.status(500).send({ error: 'Failed to start Python process. Is Python installed?' });
    });
});

app.listen(port, () => {
    console.log(`Node.js test server running at http://localhost:5173);
});