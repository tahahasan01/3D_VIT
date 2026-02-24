/**
 * Application-wide constants.
 */

/** Base URL for backend API calls. */
export const API_BASE_URL = "/api";

/** Application display name. */
export const APP_NAME = "VIT";

/** Application tagline — 3D body + garment image → fit on body. */
export const APP_TAGLINE = "3D Virtual Try-On";

/** Maximum image upload size in megabytes. */
export const MAX_UPLOAD_SIZE_MB = 10;

/** Accepted image MIME types for garment upload. */
export const ACCEPTED_IMAGE_TYPES: Record<string, string[]> = {
  "image/jpeg": [".jpg", ".jpeg"],
  "image/png": [".png"],
  "image/webp": [".webp"],
};
