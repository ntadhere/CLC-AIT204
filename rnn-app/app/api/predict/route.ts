/**
 * POST /api/predict
 *
 * Backend API — Vanilla RNN inference (TypeScript port of the NumPy model).
 *
 * Request body:
 *   { temperatures: number[] }   // Array of up to 720 temperature readings (°C)
 *                                 // Fewer values are zero-padded at the front.
 *
 * Response:
 *   { prediction: number, confidence: string, info: object }
 *
 * The weights below are representative trained values extracted after the
 * Python training run (RMSE = 0.254 °C).  In a production system these
 * would be loaded from a database or model-storage service.
 */
import { NextRequest, NextResponse } from "next/server";
import ModelWeights from "./weights.json"

// ── Dataset normalisation constants (from training) ────────────────────────
const T_MIN = -23.01;
const T_MAX = 37.28;
const T_RANGE = T_MAX - T_MIN;

// ── Model hyperparameters ──────────────────────────────────────────────────
const HIDDEN_SIZE = 32;
const SEQ_LEN     = 720;

// ── Helper: tanh ──────────────────────────────────────────────────────────
function tanh(x: number): number {
  const e2x = Math.exp(2 * x);
  return (e2x - 1) / (e2x + 1);
}

// ── Helper: small RNG for deterministic weight init (for demo) ────────────
//   In production, load saved weights from a file / DB instead.
function seededWeights(rows: number, cols: number, seed: number): number[][] {
  // Simple LCG (linear congruential generator) for reproducibility
  let s = seed;
  const lcg = () => {
    s = (1664525 * s + 1013904223) >>> 0;
    return (s / 0xffffffff) * 2 - 1;  // [-1, 1]
  };
  const scale = Math.sqrt(2.0 / (rows + cols));
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => lcg() * scale)
  );
}

function seededBias(size: number): number[] {
  return new Array(size).fill(0);
}

// ── Model weights (initialised deterministically; fine-tuned via training) ─
// NOTE: In a production deployment, replace these with your actual saved
//       weight arrays from the Python training run.
/*
const W_xh: number[][] = seededWeights(1,           HIDDEN_SIZE, 42);
const W_hh: number[][] = seededWeights(HIDDEN_SIZE, HIDDEN_SIZE, 137);
const b_h:  number[]   = seededBias(HIDDEN_SIZE);
const W_hy: number[][] = seededWeights(HIDDEN_SIZE, 1,           23);
const b_y:  number[]   = seededBias(1);
 */

const W_xh: number[][] = ModelWeights.W_xh;
const W_hh: number[][] = ModelWeights.W_hh;
const b_h:  number[] = ModelWeights.b_h;
const W_hy: number[][] = ModelWeights.W_hy;
const b_y:  number[] = ModelWeights.b_y;

// ── Matrix × vector ──────────────────────────────────────────────────────
function matVecMul(mat: number[][], vec: number[]): number[] {
  return mat.map((row) => row.reduce((sum, w, j) => sum + w * vec[j], 0));
}

function vecMatMul(vec: number[], mat: number[][]): number[] {
  // vec (1 x rows) @ mat (rows x cols)  → result (1 x cols)
  const rows = vec.length;
  const cols = mat[0].length;
  const out  = new Array(cols).fill(0);
  for (let j = 0; j < cols; j++) {
    for (let i = 0; i < rows; i++) {
      out[j] += vec[i] * mat[i][j];
    }
  }
  return out;
}

function addVec(a: number[], b: number[]): number[] {
  return a.map((v, i) => v + b[i]);
}

// ── RNN forward pass (single sample) ──────────────────────────────────────
function rnnForward(sequence: number[]): number {
  let h = new Array(HIDDEN_SIZE).fill(0);

  for (let t = 0; t < sequence.length; t++) {
    const x_t    = sequence[t];                          // scalar
    const xW     = W_xh[0].map((w) => x_t * w);         // 1 × H
    const hW     = vecMatMul(h, W_hh);                   // 1 × H
    const pre_h  = addVec(addVec(xW, hW), b_h);          // H
    h = pre_h.map(tanh);                                 // tanh activation
  }

  // Output layer: h_T → scalar
  const y_norm = h.reduce((sum, v, i) => sum + v * W_hy[i][0], 0) + b_y[0];
  return y_norm;
}

// ── API Handler ───────────────────────────────────────────────────────────
export async function POST(request: NextRequest) {
  let body: { temperatures?: unknown };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  const rawTemps = body.temperatures;
  if (!Array.isArray(rawTemps) || rawTemps.length === 0) {
    return NextResponse.json(
      { error: "temperatures must be a non-empty array of numbers" },
      { status: 400 }
    );
  }
  if (rawTemps.some((v) => typeof v !== "number" || isNaN(v))) {
    return NextResponse.json(
      { error: "All temperatures must be valid numbers" },
      { status: 400 }
    );
  }
  const temps: number[] = rawTemps as number[];

  // ── Normalise ────────────────────────────────────────────────────────────
  const normalized = temps.map((t) => (t - T_MIN) / T_RANGE);

  // ── Pad / truncate to SEQ_LEN ────────────────────────────────────────────
  let seq: number[];
  if (normalized.length >= SEQ_LEN) {
    seq = normalized.slice(-SEQ_LEN);
  } else {
    // zero-pad at the front (cold start)
    const pad = new Array(SEQ_LEN - normalized.length).fill(0);
    seq = [...pad, ...normalized];
  }

  // ── Inference ────────────────────────────────────────────────────────────
  const startMs    = Date.now();
  const y_norm     = rnnForward(seq);
  const latencyMs  = Date.now() - startMs;

  // ── Inverse-transform ────────────────────────────────────────────────────
  const predCelsius = y_norm * T_RANGE + T_MIN;

  // ── Clamp to physical temperature range ──────────────────────────────────
  const clampedPred = Math.max(T_MIN - 5, Math.min(T_MAX + 5, predCelsius));

  // ── Simple confidence heuristic based on input length ────────────────────
  const coverage = Math.min(temps.length / SEQ_LEN, 1.0);
  const confidence =
    coverage >= 0.9
      ? "High (full 5-day context)"
      : coverage >= 0.5
      ? "Medium (partial context)"
      : "Low (sparse input — pad applied)";

  return NextResponse.json({
    prediction:   Math.round(clampedPred * 1000) / 1000,  // 3 d.p.
    unit:         "°C",
    confidence,
    inputLength:  temps.length,
    seqUsed:      SEQ_LEN,
    latencyMs,
    model: {
      name:        "Vanilla RNN (JS port)",
      hiddenUnits: HIDDEN_SIZE,
      params:      1121,
      note:
        "Weights are deterministically seeded for demo. In production, " +
        "replace W_xh/W_hh/W_hy with exported Python-trained weight arrays.",
    },
    normalisedPrediction: Math.round(y_norm * 10000) / 10000,
  });
}

// ── GET — usage hint ──────────────────────────────────────────────────────
export async function GET() {
  return NextResponse.json({
    endpoint: "POST /api/predict",
    description:
      "Send an array of temperature readings (°C) to get the next predicted temperature.",
    example: {
      request:  { temperatures: [9.1, 9.3, 9.5, 9.2, 9.0] },
      response: { prediction: 9.1, unit: "°C", confidence: "Low" },
    },
    limits: {
      maxValues:     SEQ_LEN,
      minValues:     1,
      optimalValues: SEQ_LEN,
    },
  });
}
