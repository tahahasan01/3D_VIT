/**
 * Soft 3-point studio lighting matching the classic SMPL clay-mannequin look.
 *
 * The reference image shows balanced studio lighting — body features (chest,
 * abs, shoulders, knees) are clearly visible but shadows are never harsh.
 * A moderate key-to-fill ratio (~3:1) with good ambient achieves this.
 *
 * Ratios:
 *   Key  : 2.0  (front-left, slightly above body height)
 *   Fill : 0.65 (front-right, ~33 % of key)
 *   Rim  : 0.30 (back-top, edge separation)
 *   Amb  : 0.25 (keeps shadow side well-lit, matching the reference)
 *   HDRI : 0.30 (soft wrap light)
 */

import { Environment } from "@react-three/drei";

export function Lighting() {
  return (
    <>
      {/* Soft wrap light matching diffuse studio environment */}
      <Environment
        files="/textures/studio_small_08_1k.hdr"
        background={false}
        environmentIntensity={0.30}
      />

      {/* Raised ambient keeps shadow side from going dark */}
      <ambientLight intensity={0.25} />

      {/*
        Key light: front-left, slightly above body height.
        Creates visible shadow gradients across chest, abs, and
        shoulders without harsh contrast.
      */}
      <directionalLight
        position={[2, 1.8, 3]}
        intensity={2.0}
        castShadow
        shadow-mapSize-width={2048}
        shadow-mapSize-height={2048}
      />

      {/*
        Fill light: front-right at ~33 % of key.
        Keeps the right side of the body readable like in the reference.
      */}
      <directionalLight position={[-2, 1.0, 2.5]} intensity={0.65} />

      {/* Gentle rim — separates head and shoulders from background */}
      <directionalLight position={[0, 4, -2]} intensity={0.30} />
    </>
  );
}
