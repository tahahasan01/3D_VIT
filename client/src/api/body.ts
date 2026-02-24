/**
 * Body generation API client.
 */

import { API_BASE_URL } from "../utils/constants";
import type { BodyMeasurements } from "../types/body";

export interface BodyGenerationResult {
  /** Blob containing the binary GLB data. */
  blob: Blob;
  /** Body landmark positions for garment fitting (SMPL only). */
  landmarks: Record<string, number> | null;
  /** Whether the body is 'smpl' (animated), 'parametric', or 'base_mesh'. */
  modelType: "smpl" | "parametric" | "base_mesh" | null;
}

/**
 * Send body measurements to the backend and receive a GLB blob
 * along with optional body landmarks.
 */
export async function generateBody(
  measurements: BodyMeasurements,
): Promise<BodyGenerationResult> {
  const response = await fetch(`${API_BASE_URL}/body/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(measurements),
  });

  if (!response.ok) {
    const error = await response
      .json()
      .catch(() => ({ detail: "Unknown server error" }));
    throw new Error(
      (error as { detail?: string }).detail ?? `HTTP ${response.status}`,
    );
  }

  // Extract body landmarks from custom header (SMPL bodies only)
  let landmarks: Record<string, number> | null = null;
  const landmarksHeader = response.headers.get("X-Body-Landmarks");
  if (landmarksHeader) {
    try {
      landmarks = JSON.parse(landmarksHeader);
    } catch {
      // Ignore parse errors
    }
  }

  const modelType = (response.headers.get("X-Body-Model-Type") as BodyGenerationResult["modelType"]) ?? null;

  const blob = await response.blob();
  return { blob, landmarks, modelType };
}
