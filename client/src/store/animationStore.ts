/**
 * Zustand store for animation state.
 *
 * Controls which animation clip is playing (walk, twirl, or none/T-pose)
 * and the playback speed. Both BodyModel and GarmentModel consume this
 * store so they animate in sync.
 */

import { create } from "zustand";

export type AnimationType = "walk" | "twirl" | "a_pose" | "natural_stand" | "t_pose" | null;

interface AnimationState {
  /** Currently active animation clip name; null falls back to natural_stand in viewer. */
  activeAnimation: AnimationType;
  /** Playback speed multiplier (0.25 to 2.0). */
  speed: number;

  // -- Actions ---------------------------------------------------------------
  setAnimation: (anim: AnimationType) => void;
  setSpeed: (speed: number) => void;
}

export const useAnimationStore = create<AnimationState>()((set) => ({
  activeAnimation: "natural_stand",
  speed: 1.0,

  setAnimation: (activeAnimation) => set({ activeAnimation }),
  setSpeed: (speed) => set({ speed: Math.max(0.25, Math.min(2.0, speed)) }),
}));
