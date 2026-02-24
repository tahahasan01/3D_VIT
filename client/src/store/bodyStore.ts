/**
 * Zustand store for body model state.
 *
 * Manages body measurements, the generated GLB model URL,
 * loading state, and error feedback.
 */

import { create } from "zustand";
import type { BodyMeasurements } from "../types/body";

export type BodyModelType = "smpl" | "parametric" | "base_mesh" | null;

interface BodyState {
  /** Current body measurements (null until first submission). */
  measurements: BodyMeasurements | null;
  /** Object URL of the generated body GLB (null before generation). */
  modelUrl: string | null;
  /** Body landmark positions for garment fitting (SMPL only). */
  bodyLandmarks: Record<string, number> | null;
  /** Whether the body is SMPL (animated), parametric, or base_mesh. */
  modelType: BodyModelType;
  /** Whether a body generation request is in flight. */
  isGenerating: boolean;
  /** Error message from the last failed generation, if any. */
  error: string | null;

  // -- Actions ---------------------------------------------------------------
  setMeasurements: (m: BodyMeasurements) => void;
  setModelUrl: (url: string | null) => void;
  setBodyLandmarks: (l: Record<string, number> | null) => void;
  setModelType: (t: BodyModelType) => void;
  setIsGenerating: (v: boolean) => void;
  setError: (e: string | null) => void;
  reset: () => void;
}

const INITIAL_STATE = {
  measurements: null,
  modelUrl: null,
  bodyLandmarks: null,
  modelType: null as BodyModelType,
  isGenerating: false,
  error: null,
};

export const useBodyStore = create<BodyState>()((set, get) => ({
  ...INITIAL_STATE,

  setMeasurements: (measurements) => set({ measurements }),

  setBodyLandmarks: (bodyLandmarks) => set({ bodyLandmarks }),

  setModelType: (modelType) => set({ modelType }),

  setModelUrl: (url) => {
    // Revoke the previous blob URL to avoid memory leaks
    const prev = get().modelUrl;
    if (prev) {
      URL.revokeObjectURL(prev);
    }
    set({ modelUrl: url });
  },

  setIsGenerating: (isGenerating) => set({ isGenerating }),

  setError: (error) => set({ error }),

  reset: () => {
    const prev = get().modelUrl;
    if (prev) {
      URL.revokeObjectURL(prev);
    }
    set(INITIAL_STATE);
  },
}));
