/**
 * Body measurement types.
 *
 * Mirror the server-side Pydantic models so that the frontend
 * can construct valid payloads without runtime surprises.
 */

export type Gender = "male" | "female";

/** Measurements required to generate a 3D human body. */
export interface BodyMeasurements {
  gender: Gender;
  /** Total height in centimetres. */
  height_cm: number;
  /** Chest circumference in centimetres. */
  chest_cm: number;
  /** Waist circumference in centimetres. */
  waist_cm: number;
  /** Hip circumference in centimetres. */
  hip_cm: number;
  /** Shoulder-to-shoulder distance in centimetres. */
  shoulder_width_cm: number;
  /** Shoulder to wrist length in centimetres. */
  arm_length_cm: number;
  /** Crotch to ankle length in centimetres. */
  inseam_cm: number;
  /** If true, use the built-in male base mesh (OBJ) scaled to height instead of parametric generation. */
  use_base_mesh?: boolean;
  /** If true, use SMPL body model (male/female) from server assets, scaled to height; requires SMPL registration. */
  use_smpl?: boolean;
  /** Optional skin color as hex (e.g. "#DEC3AA"). Applied as body base color when set. */
  skin_color_hex?: string | null;
}

/** Default skin color (hex) when not specified. */
export const DEFAULT_SKIN_COLOR_HEX = "#E8C4A0";

/** Default male body measurements for form initialisation. */
export const DEFAULT_MALE_MEASUREMENTS: BodyMeasurements = {
  gender: "male",
  height_cm: 175,
  chest_cm: 96,
  waist_cm: 82,
  hip_cm: 98,
  shoulder_width_cm: 45,
  arm_length_cm: 60,
  inseam_cm: 80,
  use_base_mesh: false,
  use_smpl: true, // Use SMPL when available so garments conform to body
  skin_color_hex: DEFAULT_SKIN_COLOR_HEX,
};

/** Default female body measurements for form initialisation. */
export const DEFAULT_FEMALE_MEASUREMENTS: BodyMeasurements = {
  gender: "female",
  height_cm: 165,
  chest_cm: 88,
  waist_cm: 70,
  hip_cm: 96,
  shoulder_width_cm: 40,
  arm_length_cm: 55,
  inseam_cm: 75,
  use_base_mesh: false,
  use_smpl: true,
  skin_color_hex: DEFAULT_SKIN_COLOR_HEX,
};
