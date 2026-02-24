/**
 * Barrel export for all shared types.
 */

export type { BodyMeasurements, Gender } from "./body";
export {
  DEFAULT_MALE_MEASUREMENTS,
  DEFAULT_FEMALE_MEASUREMENTS,
} from "./body";

export type {
  GarmentType,
  GarmentMeasurements,
  GarmentItem,
} from "./garment";
export {
  GARMENT_TYPE_LABELS,
  DEFAULT_GARMENT_MEASUREMENTS,
} from "./garment";

/** Possible wizard step identifiers. */
export type WizardStep = "measurements" | "garment" | "tryon";
