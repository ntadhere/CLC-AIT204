# AIT-204 RNN Temperature Forecasting App

A full-stack Next.js web application that demonstrates RNN-based temperature forecasting using the Jena Climate dataset.

## Tech Stack

| Layer     | Technology          |
|-----------|---------------------|
| Frontend  | React 18 + Next.js 14 |
| Styling   | Tailwind CSS        |
| Charts    | Recharts            |
| Backend   | Next.js API Routes (TypeScript) |
| Deployment| Vercel              |

## Project Structure

```
rnn-app/
├── app/
│   ├── layout.tsx          # Root layout
│   ├── page.tsx            # Main React UI page
│   ├── globals.css         # Global styles (Tailwind)
│   └── api/
│       ├── predict/
│       │   └── route.ts    # POST /api/predict — RNN inference
│       └── health/
│           └── route.ts    # GET  /api/health  — model metadata
├── next.config.js
├── tailwind.config.js
├── tsconfig.json
├── vercel.json             # Vercel deployment config
└── package.json
```

## API Endpoints

### `POST /api/predict`

Send an array of temperature readings to get the next predicted temperature.

```json
// Request
{ "temperatures": [9.1, 9.3, 10.2, 10.5, 10.1] }

// Response
{
  "prediction": 10.1,
  "unit": "°C",
  "confidence": "High (full 5-day context)",
  "latencyMs": 2,
  "inputLength": 5
}
```

### `GET /api/health`

Returns model metadata and evaluation metrics.

## Running Locally

```bash
cd rnn-app
npm install
npm run dev
# → http://localhost:3000
```

## Deploying to Vercel

### Option 1: Vercel CLI

```bash
npm install -g vercel
cd rnn-app
vercel
# Follow the prompts — Vercel will auto-detect Next.js
```

### Option 2: GitHub + Vercel Dashboard

1. Push this project to a GitHub repository
2. Go to [vercel.com](https://vercel.com) → **New Project**
3. Import your GitHub repository
4. Vercel auto-detects Next.js — click **Deploy**
5. Your app is live at `https://your-project.vercel.app`

### Option 3: Drag-and-drop

1. Run `npm run build` locally
2. Drag the project folder to [vercel.com/new](https://vercel.com/new)

## Production Notes

The `/api/predict` endpoint uses deterministically seeded weight initialization
for demonstration. To use your actual trained weights from the Python run:

1. Export the weight matrices from Python:
   ```python
   import json
   weights = {
       "W_xh": model.W_xh.tolist(),
       "W_hh": model.W_hh.tolist(),
       "b_h":  model.b_h.tolist(),
       "W_hy": model.W_hy.tolist(),
       "b_y":  model.b_y.tolist(),
   }
   with open("weights.json", "w") as f:
       json.dump(weights, f)
   ```

2. Place `weights.json` in `app/api/predict/`

3. Load them in `route.ts`:
   ```typescript
   import weights from "./weights.json";
   const W_xh = weights.W_xh;
   // etc.
   ```

## Model Summary

| Parameter        | Value                    |
|------------------|--------------------------|
| Architecture     | Vanilla RNN (single layer)|
| Hidden units     | 32                       |
| Sequence length  | 720 steps (5 days)       |
| Total parameters | 1,121                    |
| Test RMSE        | **0.254 °C**             |
| Test MAE         | **0.184 °C**             |
| Training epochs  | 30                       |
| Optimizer        | Adam (lr = 0.001)        |
