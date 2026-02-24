/**
 * Scene lighting setup.
 *
 * Uses a combination of ambient and directional lights to
 * produce soft, even illumination on the 3D model.
 */

export function Lighting() {
  return (
    <>
      {/* Soft overall fill */}
      <ambientLight intensity={0.6} />

      {/* Key light (front-right, slightly above) */}
      <directionalLight
        position={[4, 6, 4]}
        intensity={0.8}
        castShadow
        shadow-mapSize-width={1024}
        shadow-mapSize-height={1024}
      />

      {/* Fill light (left, lower) */}
      <directionalLight position={[-3, 3, -2]} intensity={0.35} />

      {/* Rim / back light */}
      <directionalLight position={[0, 4, -5]} intensity={0.25} />
    </>
  );
}
