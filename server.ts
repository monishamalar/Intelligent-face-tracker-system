import express, { Request, Response } from 'express';
import { exec, execSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import { createServer as createViteServer } from 'vite';

const app = express();
const PORT = 3000;
const ROOT_DIR = process.cwd();

// Auto-detect the right python command (virtualenv or system)
function getPythonCommand(): string {
  if (process.env.VIRTUAL_ENV) {
    const venvWin = path.join(process.env.VIRTUAL_ENV, 'Scripts', 'python.exe');
    if (fs.existsSync(venvWin)) return `"${venvWin}"`;
  }
  const localWin = path.join(ROOT_DIR, 'venv', 'Scripts', 'python.exe');
  if (fs.existsSync(localWin)) return `"${localWin}"`;
  return 'python';
}

const py = getPythonCommand();

app.use(express.json());

// API: Get live statistics, faces, and events from SQLite
app.get('/api/data', (_req: Request, res: Response) => {
  try {
    const output = execSync(`${py} app/api_helper.py get_all`, {
      cwd: ROOT_DIR,
      encoding: 'utf-8',
    });
    res.json(JSON.parse(output));
  } catch (error: any) {
    res.status(500).json({ error: error.message || 'Failed to fetch database data' });
  }
});

// API: Get logs from logs/events.log
app.get('/api/logs', (_req: Request, res: Response) => {
  try {
    const logPath = path.join(ROOT_DIR, 'logs', 'events.log');
    if (fs.existsSync(logPath)) {
      const content = fs.readFileSync(logPath, 'utf-8');
      const lines = content.trim().split('\n').slice(-150);
      res.json({ logs: lines });
    } else {
      res.json({ logs: ['No logs found yet. Pipeline has not generated logs.'] });
    }
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

// API: Get config.json
app.get('/api/config', (_req: Request, res: Response) => {
  try {
    const configPath = path.join(ROOT_DIR, 'config.json');
    if (fs.existsSync(configPath)) {
      const data = JSON.parse(fs.readFileSync(configPath, 'utf-8'));
      res.json(data);
    } else {
      res.status(404).json({ error: 'config.json not found' });
    }
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

// API: Save config.json
app.post('/api/config', (req: Request, res: Response) => {
  try {
    const configPath = path.join(ROOT_DIR, 'config.json');
    fs.writeFileSync(configPath, JSON.stringify(req.body, null, 2), 'utf-8');
    res.json({ status: 'success', message: 'Configuration saved' });
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

// API: Trigger Python pipeline run
app.post('/api/pipeline/run', (req: Request, res: Response) => {
  const frames = req.body?.frames || 25;
  exec(`${py} app/main.py --max-frames ${frames}`, { cwd: ROOT_DIR }, (err, stdout, stderr) => {
    if (err) {
      res.status(500).json({ error: stderr || err.message });
    } else {
      res.json({ status: 'success', output: stdout });
    }
  });
});

// API: Ingress - register/log visitor entry in SQLite & disk
app.post('/api/ingress', (req: Request, res: Response) => {
  try {
    const faceId = req.body?.face_id || 'auto';
    const trackerId = req.body?.tracker_id || 'auto';
    const confidence = req.body?.confidence !== undefined ? req.body.confidence : 'auto';
    const output = execSync(`${py} app/api_helper.py ingress "${faceId}" "${trackerId}" "${confidence}"`, {
      cwd: ROOT_DIR,
      encoding: 'utf-8',
    });
    res.json(JSON.parse(output));
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

// API: Egress - log visitor exit in SQLite & disk
app.post('/api/egress', (req: Request, res: Response) => {
  try {
    const faceId = req.body?.face_id || 'auto';
    const trackerId = req.body?.tracker_id || 'auto';
    const confidence = req.body?.confidence !== undefined ? req.body.confidence : 'auto';
    const output = execSync(`${py} app/api_helper.py egress "${faceId}" "${trackerId}" "${confidence}"`, {
      cwd: ROOT_DIR,
      encoding: 'utf-8',
    });
    res.json(JSON.parse(output));
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

// API: Reset database
app.post('/api/pipeline/reset', (_req: Request, res: Response) => {
  try {
    const output = execSync(`${py} app/api_helper.py reset`, {
      cwd: ROOT_DIR,
      encoding: 'utf-8',
    });
    res.json(JSON.parse(output));
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

async function startServer() {
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.join(ROOT_DIR, 'dist')));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(ROOT_DIR, 'dist', 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Intelligent Face Tracker backend running at http://localhost:${PORT}`);
  });
}

startServer();