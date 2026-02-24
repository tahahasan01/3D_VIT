/**
 * Combined try-on API: body + garment in one request.
 *
 * Parametric approach: body from measurements, garment from template + texture.
 * Garment is fitted to the same body (conforming when T-shirt).
 */

import { API_BASE_URL } from "../utils/constants";
import type { BodyMeasurements } from "../types/body";
import type { GarmentMeasurements } from "../types/garment";

export interface CreateTryOnResult {
  success: boolean;
  body_glb_base64: string;
  garment_glb_base64: string;
  body_landmarks: Record<string, number> | null;
  garment_fit: "isp" | "conforming" | "parametric";
  body_model_type: "smpl" | "parametric" | "base_mesh";
}

/**
 * Create full try-on: generate body from measurements, process garment image
 * (parametric/conforming template + texture), return both GLBs.
 */
export async function createTryOn(
  image: File,
  bodyMeasurements: BodyMeasurements,
  garmentMeasurements: GarmentMeasurements,
): Promise<CreateTryOnResult> {
  const formData = new FormData();
  formData.append("garment_image", image);
  formData.append("body_measurements", JSON.stringify(bodyMeasurements));
  formData.append("garment_measurements", JSON.stringify(garmentMeasurements));

  const response = await fetch(`${API_BASE_URL}/tryon/create`, {
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

  return response.json();
}
