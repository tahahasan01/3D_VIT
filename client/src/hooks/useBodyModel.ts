/**
 * Custom hook for body model generation.
 *
 * Wraps the body store + API call into a single callable
 * that manages loading state, blob URLs, and error handling.
 */

import { useCallback } from "react";
import { useBodyStore } from "../store/bodyStore";
import { generateBody } from "../api/body";
import type { BodyMeasurements } from "../types/body";

interface UseBodyModelReturn {
  /** Object URL of the current body GLB, or null. */
  modelUrl: string | null;
  /** Whether a generation request is in progress. */
  isGenerating: boolean;
  /** Last error message, or null. */
  error: string | null;
  /** Trigger body generation with the given measurements. */
  generate: (measurements: BodyMeasurements) => Promise<void>;
}

export function useBodyModel(): UseBodyModelReturn {
  const modelUrl = useBodyStore((s) => s.modelUrl);
  const isGenerating = useBodyStore((s) => s.isGenerating);
  const error = useBodyStore((s) => s.error);
  const setModelUrl = useBodyStore((s) => s.setModelUrl);
  const setBodyLandmarks = useBodyStore((s) => s.setBodyLandmarks);
  const setIsGenerating = useBodyStore((s) => s.setIsGenerating);
  const setError = useBodyStore((s) => s.setError);
  const setMeasurements = useBodyStore((s) => s.setMeasurements);
  const setModelType = useBodyStore((s) => s.setModelType);

  const generate = useCallback(
    async (measurements: BodyMeasurements) => {
      setIsGenerating(true);
      setError(null);

      try {
        const { blob, landmarks, modelType } = await generateBody(measurements);
        const url = URL.createObjectURL(blob);
        setModelUrl(url);
        setBodyLandmarks(landmarks);
        setMeasurements(measurements);
        setModelType(modelType);
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Failed to generate body model";
        setError(message);
      } finally {
        setIsGenerating(false);
      }
    },
    [setModelUrl, setBodyLandmarks, setIsGenerating, setError, setMeasurements, setModelType],
  );

  return { modelUrl, isGenerating, error, generate };
}
