/**
 * Zustand store for garment state.
 *
 * Manages the list of garments loaded into the try-on scene,
 * processing state, and error feedback.
 */

import { create } from "zustand";
import type { GarmentItem } from "../types/garment";

interface GarmentState {
  /** Garments currently loaded in the scene. */
  garments: GarmentItem[];
  /** Whether a garment processing request is in flight. */
  isProcessing: boolean;
  /** Error message from the last failed processing, if any. */
  error: string | null;

  // -- Actions ---------------------------------------------------------------
  addGarment: (g: GarmentItem) => void;
  removeGarment: (id: string) => void;
  toggleVisibility: (id: string) => void;
  setIsProcessing: (v: boolean) => void;
  setError: (e: string | null) => void;
  reset: () => void;
}

const INITIAL_STATE = {
  garments: [] as GarmentItem[],
  isProcessing: false,
  error: null,
};

export const useGarmentStore = create<GarmentState>()((set, get) => ({
  ...INITIAL_STATE,

  addGarment: (garment) =>
    set((state) => ({ garments: [...state.garments, garment] })),

  removeGarment: (id) => {
    const target = get().garments.find((g) => g.id === id);
    if (target) {
      URL.revokeObjectURL(target.modelUrl);
    }
    set((state) => ({
      garments: state.garments.filter((g) => g.id !== id),
    }));
  },

  toggleVisibility: (id) =>
    set((state) => ({
      garments: state.garments.map((g) =>
        g.id === id ? { ...g, visible: !g.visible } : g,
      ),
    })),

  setIsProcessing: (isProcessing) => set({ isProcessing }),

  setError: (error) => set({ error }),

  reset: () => {
    for (const g of get().garments) {
      URL.revokeObjectURL(g.modelUrl);
    }
    set(INITIAL_STATE);
  },
}));
