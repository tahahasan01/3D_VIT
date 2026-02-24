/**
 * API client configuration.
 *
 * Re-exports feature-specific API functions and provides
 * the shared base URL constant.
 */

export { API_BASE_URL } from "../utils/constants";

export { generateBody } from "./body";
export { processGarment } from "./garment";
export { createTryOn, type CreateTryOnResult } from "./tryon";
