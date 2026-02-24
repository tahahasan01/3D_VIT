/**
 * Renders a single 3D garment GLB in the scene.
 *
 * Preserves the embedded texture (baseColorTexture) from the GLB so the
 * garment displays the uploaded photo. When the GLB contains a skeleton
 * and animations (conforming garment on SMPL), plays animations in sync
 * with the body via the shared animation store.
 */

import { useEffect, useMemo, useRef } from "react";
import { useGLTF } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { clone as skeletonClone } from "three/examples/jsm/utils/SkeletonUtils.js";
import { useAnimationStore } from "../../store/animationStore";

interface GarmentModelProps {
  /** Object URL pointing to the garment GLB blob. */
  url: string;
}

export function GarmentModel({ url }: GarmentModelProps) {
  const { scene, animations } = useGLTF(url);
  const activeAnimation = useAnimationStore((s) => s.activeAnimation);
  const speed = useAnimationStore((s) => s.speed);
  const mixerRef = useRef<THREE.AnimationMixer | null>(null);
  const currentActionRef = useRef<THREE.AnimationAction | null>(null);

  const cloned = useMemo(() => {
    let hasSkinned = false;
    scene.traverse((child) => {
      if ((child as THREE.SkinnedMesh).isSkinnedMesh) hasSkinned = true;
    });
    const c = hasSkinned ? skeletonClone(scene) : scene.clone(true);

    c.traverse((child) => {
      if (child instanceof THREE.SkinnedMesh) {
        child.frustumCulled = false;
        child.geometry.computeVertexNormals();
      }

      if (child instanceof THREE.Mesh) {
        child.geometry.computeVertexNormals();
        child.renderOrder = 0; // garment renders first to fill stencil

        const hasVertexColors =
          child.geometry.attributes.color != null &&
          child.geometry.attributes.color.count > 0;

        const materials = Array.isArray(child.material)
          ? child.material
          : [child.material];

        for (const mat of materials) {
          if (!mat) continue;
          mat.side = THREE.DoubleSide;
          mat.depthWrite = true;
          mat.depthTest = true;
          // Stencil: mark covered regions so body is hidden behind garment
          mat.stencilWrite = true;
          mat.stencilRef = 1;
          mat.stencilFunc = THREE.AlwaysStencilFunc;
          mat.stencilZPass = THREE.ReplaceStencilOp;
          mat.stencilZFail = THREE.KeepStencilOp;
          mat.stencilFail = THREE.KeepStencilOp;
          if (hasVertexColors) {
            mat.vertexColors = true;
            // White base so vertex colors show unmodified
            mat.color.setHex(0xffffff);
          }
          // When no vertex colors, preserve the baseColorFactor loaded from GLB
          // (do NOT override mat.color — it already holds the garment colour)
          if (mat.map) {
            mat.map.colorSpace = THREE.SRGBColorSpace;
            mat.map.anisotropy = 4;
            mat.map.needsUpdate = true;
          }
          if (mat instanceof THREE.MeshStandardMaterial) {
            mat.flatShading = false;
            mat.envMapIntensity = 1;
            mat.roughness = 0.9;
            mat.metalness = 0;
            mat.emissive.setHex(0x000000);
          }
          if (mat instanceof THREE.MeshBasicMaterial) {
            mat.transparent = false;
          }
          // Ensure fully opaque to prevent any body bleed-through
          mat.transparent = false;
          mat.opacity = 1.0;
          mat.needsUpdate = true;
        }
      }
    });

    return c;
  }, [scene]);

  // Create mixer when scene or animations change
  useEffect(() => {
    if (animations && animations.length > 0) {
      const mixer = new THREE.AnimationMixer(cloned);
      mixerRef.current = mixer;
      return () => {
        mixer.stopAllAction();
        mixerRef.current = null;
      };
    }
  }, [cloned, animations]);

  // Switch animation clips based on store state
  useEffect(() => {
    const mixer = mixerRef.current;
    if (!mixer || !animations || animations.length === 0) return;

    if (currentActionRef.current) {
      currentActionRef.current.fadeOut(0.3);
    }

    if (activeAnimation === null) {
      mixer.stopAllAction();
      currentActionRef.current = null;
      return;
    }

    const clip = animations.find((a) => a.name === activeAnimation);
    if (!clip) {
      mixer.stopAllAction();
      currentActionRef.current = null;
      return;
    }

    const action = mixer.clipAction(clip);
    action.reset();
    action.fadeIn(0.3);
    action.play();
    currentActionRef.current = action;
  }, [activeAnimation, animations]);

  // Tick the mixer every frame
  useFrame((_, delta) => {
    if (mixerRef.current) {
      mixerRef.current.update(delta * speed);
    }
  });

  useEffect(() => {
    return () => {
      useGLTF.clear(url);
    };
  }, [url]);

  return <primitive object={cloned} />;
}
