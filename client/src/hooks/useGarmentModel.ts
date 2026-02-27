/**
 * Custom hook for garment processing.
 *
 * When body measurements exist, uses the combined try-on endpoint so the
 * backend generates body + garment in one pipeline (parametric/conforming).
 * Otherwise falls back to garment-only processing.
 */

import { useCallback } from "react";
import { useGarmentStore } from "../store/garmentStore";
import { useBodyStore } from "../store/bodyStore";
import { processGarment } from "../api/garment";
import { createTryOn } from "../api/tryon";
import type { GarmentMeasurements } from "../types/garment";
import type { GarmentSubmitOptions } from "../types/garment";

interface UseGarmentModelReturn {
  /** Whether a processing request is in progress. */
  isProcessing: boolean;
  /** Last error message, or null. */
  error: string | null;
  /** Upload and process a garment image (refreshes body when using combined try-on). */
  process: (
    image: File,
    measurements: GarmentMeasurements,
    options?: GarmentSubmitOptions,
    additionalImages?: File[],
  ) => Promise<void>;
}

function base64ToBlobUrl(base64: string, mime: string): string {
  const bin = atob(base64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const blob = new Blob([bytes], { type: mime });
  return URL.createObjectURL(blob);
}

export function useGarmentModel(): UseGarmentModelReturn {
  const isProcessing = useGarmentStore((s) => s.isProcessing);
  const error = useGarmentStore((s) => s.error);
  const addGarment = useGarmentStore((s) => s.addGarment);
  const setIsProcessing = useGarmentStore((s) => s.setIsProcessing);
  const setError = useGarmentStore((s) => s.setError);
  const bodyMeasurements = useBodyStore((s) => s.measurements);
  const setModelUrl = useBodyStore((s) => s.setModelUrl);
  const setBodyLandmarks = useBodyStore((s) => s.setBodyLandmarks);
  const setModelType = useBodyStore((s) => s.setModelType);

  const process = useCallback(
    async (
      image: File,
      measurements: GarmentMeasurements,
      _options?: GarmentSubmitOptions,
      additionalImages?: File[],
    ) => {
      setIsProcessing(true);
      setError(null);

      try {
        // Combined try-on when we have body measurements: one request builds body + garment
        if (bodyMeasurements) {
          const result = await createTryOn(image, bodyMeasurements, measurements, additionalImages);
          const bodyUrl = base64ToBlobUrl(result.body_glb_base64, "model/gltf-binary");
          const garmentUrl = base64ToBlobUrl(result.garment_glb_base64, "model/gltf-binary");
          setModelUrl(bodyUrl);
          if (result.body_landmarks) setBodyLandmarks(result.body_landmarks);
          if (result.body_model_type) setModelType(result.body_model_type);
          addGarment({
            id: crypto.randomUUID(),
            type: measurements.garment_type,
            modelUrl: garmentUrl,
            measurements,
            visible: true,
            imageName: image.name,
          });
          return;
        }

        // No body yet: garment-only (parametric)
        const payload: GarmentMeasurements = {
          ...measurements,
          ...(bodyMeasurements
            ? { body_measurements: bodyMeasurements as GarmentMeasurements["body_measurements"] }
            : {}),
        };
        const blob = await processGarment(image, payload);
        const url = URL.createObjectURL(blob);
        addGarment({
          id: crypto.randomUUID(),
          type: measurements.garment_type,
          modelUrl: url,
          measurements,
          visible: true,
          imageName: image.name,
        });
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Failed to process garment";
        setError(message);
      } finally {
        setIsProcessing(false);
      }
    },
    [
      addGarment,
      bodyMeasurements,
      setBodyLandmarks,
      setModelType,
      setModelUrl,
      setIsProcessing,
      setError,
    ],
  );

  return { isProcessing, error, process };
}
