/**
 * Main 3D model viewer.
 *
 * Sets up the React Three Fiber canvas, camera, orbit controls,
 * lighting, ground stage, and the composite try-on scene.
 *
 * Also renders an overlay toolbar with camera reset and screenshot
 * actions.
 */

import { useCallback, useRef } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import * as THREE from "three";

import { Lighting } from "./Lighting";
import { Stage } from "./Stage";
import { TryOnScene } from "./TryOnScene";
import { ViewerControls } from "./ViewerControls";

/** Default camera position (front, slightly above). */
const DEFAULT_CAMERA_POSITION = new THREE.Vector3(0, 1.0, 2.8);
/** Default orbit target (roughly at the model's centre of mass). */
const DEFAULT_TARGET = new THREE.Vector3(0, 0.85, 0);

interface ModelViewerProps {
  /** When false, overlay controls are hidden (rendered externally). */
  showOverlayControls?: boolean;
}

export function ModelViewer({ showOverlayControls = true }: ModelViewerProps) {
  const controlsRef = useRef<OrbitControlsImpl>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const handleResetCamera = useCallback(() => {
    const controls = controlsRef.current;
    if (!controls) return;

    controls.object.position.copy(DEFAULT_CAMERA_POSITION);
    controls.target.copy(DEFAULT_TARGET);
    controls.update();
  }, []);

  const handleScreenshot = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const link = document.createElement("a");
    link.download = `vit-tryon-${Date.now()}.png`;
    link.href = canvas.toDataURL("image/png");
    link.click();
  }, []);

  return (
    <div className="relative h-full w-full min-h-[400px] rounded-xl overflow-hidden border border-gray-200 bg-gradient-to-b from-gray-100 to-gray-50">
      <Canvas
        ref={canvasRef}
        camera={{
          position: DEFAULT_CAMERA_POSITION.toArray(),
          fov: 45,
          near: 0.1,
          far: 100,
        }}
        gl={{ preserveDrawingBuffer: true, antialias: true, stencil: true }}
        shadows
      >
        <Lighting />
        <TryOnScene />
        <Stage />
        <OrbitControls
          ref={controlsRef}
          target={DEFAULT_TARGET.toArray() as [number, number, number]}
          minDistance={0.8}
          maxDistance={8}
          maxPolarAngle={Math.PI * 0.85}
          enablePan
          enableZoom
          enableDamping
          dampingFactor={0.08}
        />
      </Canvas>

      {showOverlayControls && (
        <ViewerControls
          onResetCamera={handleResetCamera}
          onScreenshot={handleScreenshot}
        />
      )}
    </div>
  );
}
