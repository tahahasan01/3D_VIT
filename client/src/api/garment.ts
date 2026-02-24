/**
 * Garment processing API client.
 */

import { API_BASE_URL } from "../utils/constants";
import type { GarmentMeasurements } from "../types/garment";

/**
 * Upload a garment image with sizing data and receive a textured GLB blob.
 *
 * @param image - The garment photograph file.
 * @param measurements - Garment sizing parameters.
 * @returns A Blob containing the binary GLB data.
 * @throws Error if the request fails or returns a non-OK status.
 */
export async function processGarment(
  image: File,
  measurements: GarmentMeasurements,
): Promise<Blob> {
  const formData = new FormData();
  formData.append("image", image);
  formData.append("measurements", JSON.stringify(measurements));

  const response = await fetch(`${API_BASE_URL}/garment/process`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    let detail: string;
    try {
      const body = await response.json();
      detail = (body as { detail?: string }).detail ?? (body as { message?: string }).message ?? `HTTP ${response.status}`;
    } catch {
      const text = await response.text().catch(() => "");
      detail = text?.slice(0, 200) || `Server error (${response.status})`;
    }
    throw new Error(detail);
  }

  return response.blob();
}
