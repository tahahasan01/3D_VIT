/**
 * Ground plane and reference grid for the 3D scene.
 *
 * Provides spatial context so the user can gauge scale and
 * orientation while inspecting the model.
 */

import { Grid } from "@react-three/drei";

export function Stage() {
  return (
    <>
      {/* Infinite-style grid */}
      <Grid
        position={[0, 0, 0]}
        args={[10, 10]}
        cellSize={0.25}
        cellThickness={0.5}
        cellColor="#e5e7eb"
        sectionSize={1}
        sectionThickness={1}
        sectionColor="#d1d5db"
        fadeDistance={8}
        fadeStrength={1.5}
        infiniteGrid
      />

      {/* Invisible ground plane for shadow reception */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.001, 0]} receiveShadow>
        <planeGeometry args={[20, 20]} />
        <shadowMaterial opacity={0.15} />
      </mesh>
    </>
  );
}
