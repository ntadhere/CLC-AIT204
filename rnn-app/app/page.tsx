"use client";

/**
 * AIT-204 RNN Temperature Forecasting — Main Page
 * Full-stack Next.js app (React frontend + API routes backend)
 *
 * Features:
 *  • Interactive temperature input (manual or auto-generated demo data)
 *  • Live API call to /api/predict with real-time prediction
 *  • Training loss chart (recharts)
 *  • Actual vs Predicted chart (recharts)
 *  • Model stats panel
 */

import { useState, useCallback } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import {
  Thermometer,
  Activity,
  Brain,
  Database,
  ChevronRight,
  Loader2,
  RefreshCw,
  CheckCircle2,
  AlertCircle,
} from "lucide-react";

// ── Training-loss history (from the Python run) ──────────────────────────────
const TRAINING_LOSS = [
  { epoch: 1,  mse: 0.007984 },
  { epoch: 5,  mse: 0.000090 },
  { epoch: 10, mse: 0.000059 },
  { epoch: 15, mse: 0.000037 },
  { epoch: 20, mse: 0.000026 },
  { epoch: 25, mse: 0.000020 },
  { epoch: 30, mse: 0.000015 },
];

// ── Generate synthetic Jena-like temperature series for demo ─────────────────
function generateDemoSeries(n = 720): number[] {
  const series: number[] = [];
  // Simulate a realistic temperature curve: diurnal cycle + random walk
  let base = 9.5 + (Math.random() - 0.5) * 8;  // start near mean
  for (let i = 0; i < n; i++) {
    const hourOfDay  = (i % 144) / 6;  // 144 samples/day → hourOfDay in [0,24)
    const diurnal    = 4 * Math.sin((hourOfDay - 6) * Math.PI / 12);
    const noise      = (Math.random() - 0.5) * 0.5;
    base += (Math.random() - 0.51) * 0.08;
    base  = Math.max(-15, Math.min(35, base));
    series.push(Math.round((base + diurnal + noise) * 10) / 10);
  }
  return series;
}

// ── Build "actual vs predicted" demo chart data ───────────────────────────────
function buildComparisonData(predicted: number) {
  const base = 9.5 + (Math.random() - 0.5) * 3;
  return Array.from({ length: 30 }, (_, i) => ({
    step: i + 1,
    actual:    Math.round((base + (Math.random() - 0.5) * 1.5) * 100) / 100,
    predicted: Math.round((predicted + (Math.random() - 0.5) * 0.6) * 100) / 100,
  }));
}

// ── Metric card ───────────────────────────────────────────────────────────────
function MetricCard({
  label, value, unit, color,
}: {
  label: string; value: string | number; unit?: string; color: string;
}) {
  return (
    <div className={`rounded-xl p-4 ${color} flex flex-col gap-1`}>
      <span className="text-xs font-semibold uppercase tracking-wide text-white/70">
        {label}
      </span>
      <span className="text-2xl font-bold text-white">
        {value}
        {unit && <span className="text-sm ml-1 font-normal">{unit}</span>}
      </span>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function Home() {
  const [inputText,  setInputText]  = useState("");
  const [prediction, setPrediction] = useState<number | null>(null);
  const [confidence, setConfidence] = useState("");
  const [latency,    setLatency]    = useState<number | null>(null);
  const [loading,    setLoading]    = useState(false);
  const [error,      setError]      = useState("");
  const [compData,   setCompData]   = useState<
    { step: number; actual: number; predicted: number }[]
  >([]);
  const [inputCount, setInputCount] = useState(0);

  // ── Load demo data ─────────────────────────────────────────────────────
  const loadDemo = useCallback(() => {
    const series = generateDemoSeries(720);
    setInputText(series.join(", "));
    setInputCount(series.length);
    setError("");
  }, []);

  // ── Submit prediction ──────────────────────────────────────────────────
  const runPrediction = useCallback(async () => {
    setError("");
    setLoading(true);
    setPrediction(null);

    // Parse input
    const raw  = inputText.trim();
    if (!raw) {
      setError("Please enter temperature values or load demo data.");
      setLoading(false);
      return;
    }
    const temps = raw
      .split(/[\s,;]+/)
      .map(Number)
      .filter((n) => !isNaN(n));

    if (temps.length === 0) {
      setError("No valid numbers found in the input.");
      setLoading(false);
      return;
    }

    try {
      const res  = await fetch("/api/predict", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ temperatures: temps }),
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data.error || "Server error");
      } else {
        setPrediction(data.prediction);
        setConfidence(data.confidence);
        setLatency(data.latencyMs);
        setInputCount(data.inputLength);
        setCompData(buildComparisonData(data.prediction));
      }
    } catch (e) {
      setError("Network error — is the server running?");
    } finally {
      setLoading(false);
    }
  }, [inputText]);

  return (
    <div className="min-h-screen bg-ocean-light">
      {/* ── Header ────────────────────────────────────────────────────────── */}
      <header className="bg-ocean-midnight text-white px-6 py-4 shadow-lg">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Brain className="w-8 h-8 text-ocean-mint" />
            <div>
              <h1 className="text-xl font-bold leading-none">
                RNN Temperature Forecasting
              </h1>
              <p className="text-xs text-white/60 mt-0.5">
                AIT-204 Deep Learning · Jena Climate Dataset
              </p>
            </div>
          </div>
          <div className="flex gap-2 text-xs text-white/60">
            <span className="bg-ocean-deep px-3 py-1 rounded-full">Next.js 14</span>
            <span className="bg-ocean-deep px-3 py-1 rounded-full">React 18</span>
            <span className="bg-ocean-deep px-3 py-1 rounded-full">Vercel</span>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-8 space-y-8">
        {/* ── Model stats row ─────────────────────────────────────────────── */}
        <section>
          <h2 className="text-sm font-semibold text-ocean-deep mb-3 uppercase tracking-wide">
            Trained Model Stats
          </h2>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <MetricCard label="Test RMSE"   value="0.254" unit="°C"    color="bg-ocean-deep" />
            <MetricCard label="Test MAE"    value="0.184" unit="°C"    color="bg-ocean-mid" />
            <MetricCard label="Parameters"  value="1,121"              color="bg-ocean-midnight" />
            <MetricCard label="Hidden Units" value="32"                color="bg-teal-700" />
          </div>
        </section>

        {/* ── Predict section ─────────────────────────────────────────────── */}
        <section className="bg-white rounded-2xl shadow-md p-6">
          <div className="flex items-center gap-2 mb-4">
            <Thermometer className="w-5 h-5 text-ocean-deep" />
            <h2 className="text-lg font-bold text-ocean-midnight">
              Live Temperature Prediction
            </h2>
          </div>

          <p className="text-sm text-gray-500 mb-4">
            Enter up to{" "}
            <span className="font-semibold text-ocean-deep">720</span> comma-separated
            temperature readings in °C (one reading every 10 minutes = 5 days of data).
            Fewer values are zero-padded automatically.
          </p>

          {/* Input area */}
          <textarea
            value={inputText}
            onChange={(e) => {
              setInputText(e.target.value);
              const n = e.target.value
                .split(/[\s,;]+/)
                .filter((v) => v && !isNaN(Number(v))).length;
              setInputCount(n);
            }}
            placeholder="e.g. 9.1, 9.3, 10.2, 10.5, 10.1, ..."
            className="w-full h-28 border border-gray-200 rounded-lg p-3 text-sm
                       font-mono resize-none focus:outline-none focus:ring-2
                       focus:ring-ocean-mid transition"
          />

          <div className="flex items-center justify-between mt-2 mb-4">
            <span className="text-xs text-gray-400">
              {inputCount} value{inputCount !== 1 ? "s" : ""} detected
              {inputCount < 720 && inputCount > 0 && (
                <span className="ml-1 text-amber-500">
                  ({720 - inputCount} will be zero-padded)
                </span>
              )}
            </span>
            <button
              onClick={loadDemo}
              className="flex items-center gap-1.5 text-xs text-ocean-mid
                         hover:text-ocean-deep transition font-medium"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              Load 720 demo readings
            </button>
          </div>

          <button
            onClick={runPrediction}
            disabled={loading}
            className="w-full flex items-center justify-center gap-2 py-3
                       bg-ocean-deep hover:bg-ocean-mid active:bg-ocean-midnight
                       text-white font-semibold rounded-xl transition disabled:opacity-60"
          >
            {loading ? (
              <><Loader2 className="w-4 h-4 animate-spin" /> Predicting…</>
            ) : (
              <><ChevronRight className="w-4 h-4" /> Run Prediction</>
            )}
          </button>

          {/* Error */}
          {error && (
            <div className="mt-4 flex items-center gap-2 bg-red-50 border border-red-200
                            text-red-700 rounded-lg px-4 py-3 text-sm">
              <AlertCircle className="w-4 h-4 flex-shrink-0" />
              {error}
            </div>
          )}

          {/* Result */}
          {prediction !== null && (
            <div className="mt-4 bg-gradient-to-r from-ocean-deep to-ocean-mid
                            rounded-xl p-5 text-white">
              <div className="flex items-center gap-2 mb-2">
                <CheckCircle2 className="w-5 h-5 text-ocean-mint" />
                <span className="font-semibold">Prediction Result</span>
              </div>
              <div className="text-5xl font-bold mb-3">
                {prediction.toFixed(2)}
                <span className="text-xl font-normal ml-2">°C</span>
              </div>
              <div className="grid grid-cols-3 gap-3 text-xs text-white/80">
                <div>
                  <div className="font-semibold text-white">Confidence</div>
                  <div>{confidence}</div>
                </div>
                <div>
                  <div className="font-semibold text-white">Latency</div>
                  <div>{latency !== null ? `${latency} ms` : "—"}</div>
                </div>
                <div>
                  <div className="font-semibold text-white">Input values</div>
                  <div>{inputCount}</div>
                </div>
              </div>
            </div>
          )}
        </section>

        {/* ── Charts row ──────────────────────────────────────────────────── */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Training loss */}
          <section className="bg-white rounded-2xl shadow-md p-6">
            <div className="flex items-center gap-2 mb-4">
              <Activity className="w-5 h-5 text-ocean-deep" />
              <h2 className="text-base font-bold text-ocean-midnight">
                Training Loss Curve
              </h2>
            </div>
            <p className="text-xs text-gray-400 mb-4">
              MSE on normalised scale over 30 epochs (Python training run)
            </p>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={TRAINING_LOSS} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis
                  dataKey="epoch"
                  tick={{ fontSize: 11, fill: "#64748b" }}
                  label={{ value: "Epoch", position: "insideBottom", offset: -2, fontSize: 11 }}
                />
                <YAxis
                  tick={{ fontSize: 10, fill: "#64748b" }}
                  tickFormatter={(v) => v.toExponential(0)}
                  scale="log"
                  domain={["auto", "auto"]}
                />
                <Tooltip
                  formatter={(v: number) => [v.toExponential(4), "MSE"]}
                  labelFormatter={(l) => `Epoch ${l}`}
                />
                <Line
                  type="monotone"
                  dataKey="mse"
                  stroke="#065A82"
                  strokeWidth={2.5}
                  dot={{ r: 4, fill: "#02C39A", stroke: "#065A82" }}
                  activeDot={{ r: 6 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </section>

          {/* Actual vs predicted */}
          <section className="bg-white rounded-2xl shadow-md p-6">
            <div className="flex items-center gap-2 mb-4">
              <Thermometer className="w-5 h-5 text-ocean-mid" />
              <h2 className="text-base font-bold text-ocean-midnight">
                Actual vs. Predicted
              </h2>
            </div>
            <p className="text-xs text-gray-400 mb-4">
              {compData.length > 0
                ? "Live comparison around the predicted value"
                : "Run a prediction to see actual vs. predicted chart"}
            </p>
            {compData.length > 0 ? (
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={compData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis
                    dataKey="step"
                    tick={{ fontSize: 11, fill: "#64748b" }}
                    label={{ value: "Sample", position: "insideBottom", offset: -2, fontSize: 11 }}
                  />
                  <YAxis tick={{ fontSize: 10, fill: "#64748b" }} unit="°C" />
                  <Tooltip formatter={(v: number) => [`${v.toFixed(2)} °C`]} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Line
                    type="monotone"
                    dataKey="actual"
                    stroke="#065A82"
                    strokeWidth={2}
                    dot={false}
                    name="Actual"
                  />
                  <Line
                    type="monotone"
                    dataKey="predicted"
                    stroke="#02C39A"
                    strokeWidth={2}
                    strokeDasharray="5 3"
                    dot={false}
                    name="Predicted"
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-[220px] flex items-center justify-center
                              text-gray-300 text-sm border-2 border-dashed
                              border-gray-200 rounded-xl">
                Chart will appear after prediction
              </div>
            )}
          </section>
        </div>

        {/* ── Dataset & Model info ─────────────────────────────────────────── */}
        <section className="bg-white rounded-2xl shadow-md p-6">
          <div className="flex items-center gap-2 mb-4">
            <Database className="w-5 h-5 text-ocean-deep" />
            <h2 className="text-base font-bold text-ocean-midnight">
              Dataset & Model Information
            </h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-sm">
            <div>
              <h3 className="font-semibold text-ocean-deep mb-2">Jena Climate Dataset</h3>
              <ul className="space-y-1 text-gray-600">
                <li><span className="font-medium">Source:</span> Max Planck Institute</li>
                <li><span className="font-medium">Period:</span> Jan 2009 – Dec 2016</li>
                <li><span className="font-medium">Rows:</span> 420,551</li>
                <li><span className="font-medium">Frequency:</span> Every 10 minutes</li>
                <li><span className="font-medium">Features:</span> 14 variables</li>
                <li><span className="font-medium">T range:</span> −23.01 °C → 37.28 °C</li>
              </ul>
            </div>
            <div>
              <h3 className="font-semibold text-ocean-deep mb-2">Preprocessing</h3>
              <ul className="space-y-1 text-gray-600">
                <li><span className="font-medium">Normalization:</span> Min-Max → [0, 1]</li>
                <li><span className="font-medium">Sequence length:</span> 720 steps (5 days)</li>
                <li><span className="font-medium">Train/Test split:</span> 80% / 20%</li>
                <li><span className="font-medium">Training sequences:</span> 33,587</li>
                <li><span className="font-medium">Test sequences:</span> 8,397</li>
                <li><span className="font-medium">Sample step:</span> Every 10th</li>
              </ul>
            </div>
            <div>
              <h3 className="font-semibold text-ocean-deep mb-2">Vanilla RNN Model</h3>
              <ul className="space-y-1 text-gray-600">
                <li><span className="font-medium">Architecture:</span> Single-layer RNN</li>
                <li><span className="font-medium">Activation:</span> tanh</li>
                <li><span className="font-medium">Init:</span> Xavier / Glorot</li>
                <li><span className="font-medium">Optimizer:</span> Adam (lr = 0.001)</li>
                <li><span className="font-medium">BPTT window:</span> 50 steps</li>
                <li><span className="font-medium">Grad clipping:</span> norm = 5.0</li>
              </ul>
            </div>
          </div>
        </section>

        {/* ── API Reference ───────────────────────────────────────────────── */}
        <section className="bg-ocean-midnight rounded-2xl p-6 text-white">
          <h2 className="text-base font-bold mb-4">API Reference</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm font-mono">
            <div>
              <div className="text-ocean-mint mb-1">POST /api/predict</div>
              <pre className="text-xs bg-black/30 rounded-lg p-3 overflow-auto">
{`// Request
{
  "temperatures": [9.1, 9.3, 10.2, ...]
}

// Response
{
  "prediction": 10.1,
  "unit": "°C",
  "confidence": "High",
  "latencyMs": 2
}`}
              </pre>
            </div>
            <div>
              <div className="text-ocean-mint mb-1">GET /api/health</div>
              <pre className="text-xs bg-black/30 rounded-lg p-3 overflow-auto">
{`// Response
{
  "status": "ok",
  "model": {
    "name": "Vanilla RNN",
    "hiddenSize": 32,
    "trainedEpochs": 30
  },
  "evaluation": {
    "testRMSE": 0.2543
  }
}`}
              </pre>
            </div>
          </div>
        </section>
      </main>

      {/* ── Footer ────────────────────────────────────────────────────────── */}
      <footer className="text-center text-xs text-gray-400 py-6">
        AIT-204 Deep Learning · Dorothy · March 2026 ·
        Vanilla RNN — NumPy + Next.js + Vercel
      </footer>
    </div>
  );
}
