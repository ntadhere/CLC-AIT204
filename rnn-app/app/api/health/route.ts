/**
 * GET /api/health
 * Simple health-check endpoint — returns model metadata and server status.
 */
import { NextResponse } from "next/server";

export async function GET() {
  return NextResponse.json({
    status: "ok",
    timestamp: new Date().toISOString(),
    model: {
      name: "Vanilla RNN",
      hiddenSize: 32,
      seqLen: 720,
      totalParams: 1121,
      trainedEpochs: 30,
      optimizer: "Adam (lr=0.001)",
      bpttWindow: 50,
    },
    dataset: {
      name: "Jena Climate 2009-2016",
      totalRows: 420551,
      targetFeature: "T (degC)",
      tMin: -23.01,
      tMax: 37.28,
      tMean: 9.5,
      freqMinutes: 10,
    },
    evaluation: {
      testRMSE: 0.2543,
      testMAE: 0.1836,
      testMSE_normalized: 0.0646,
      trainMSE_final: 0.000015,
    },
  });
}
