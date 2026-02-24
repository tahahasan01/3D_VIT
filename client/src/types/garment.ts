/**
 * Garment types and measurement schemas.
 *
 * Mirrors the server-side GarmentMeasurements model.
 */

export type GarmentType = "tshirt" | "pants" | "dress";

/** Human-readable labels for garment types. */
export const GARMENT_TYPE_LABELS: Record<GarmentType, string> = {
  tshirt: "T-Shirt",
  pants: "Pants",
  dress: "Dress",
};

/** Body measurements (for conforming garment from body mesh). */
export type BodyMeasurementsForGarment = Record<string, number | string | boolean>;

/** Measurements for sizing a 3D garment. */
export interface GarmentMeasurements {
  garment_type: GarmentType;
  chest_cm?: number;
  waist_cm?: number;
  hip_cm?: number;
  length_cm?: number;
  sleeve_length_cm?: number;
  inseam_cm?: number;
  /** Body landmarks from SMPL (optional). */
  body_landmarks?: Record<string, number>;
  /** Body measurements used to generate the model (optional). When present, garment is built as conforming offset. */
  body_measurements?: BodyMeasurementsForGarment;
}

/** Options when submitting the garment upload form (reserved for future use). */
export interface GarmentSubmitOptions {
  [key: string]: unknown;
}

/** A single garment loaded into the try-on scene. */
export interface GarmentItem {
  /** Unique identifier for this garment instance. */
  id: string;
  /** Garment type. */
  type: GarmentType;
  /** Object URL pointing to the loaded GLB blob. */
  modelUrl: string;
  /** Measurements used to generate this garment. */
  measurements: GarmentMeasurements;
  /** Whether the garment is visible in the scene. */
  visible: boolean;
  /** Original filename of the uploaded image. */
  imageName: string;
}

/** Default measurements per garment type. */
export const DEFAULT_GARMENT_MEASUREMENTS: Record<GarmentType, GarmentMeasurements> = {
  tshirt: {
    garment_type: "tshirt",
    chest_cm: 100,
    length_cm: 72,
    sleeve_length_cm: 24,
  },
  pants: {
    garment_type: "pants",
    waist_cm: 84,
    hip_cm: 100,
    inseam_cm: 78,
    length_cm: 100,
  },
  dress: {
    garment_type: "dress",
    chest_cm: 92,
    waist_cm: 74,
    hip_cm: 98,
    length_cm: 100,
  },
};
