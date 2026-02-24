/**
 * Renders the 3D body GLB in the scene.
 *
 * Loads the model from a blob URL. When the GLB contains a skeleton
 * and animations (SMPL skinned mesh), creates an AnimationMixer to
 * play walk/twirl clips based on the animation store.
 */

import { useEffect, useMemo, useRef } from "react";
import { useGLTF } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { clone as skeletonClone } from "three/examples/jsm/utils/SkeletonUtils.js";
import { useAnimationStore } from "../../store/animationStore";

interface BodyModelProps {
  /** Object URL pointing to the body GLB blob. */
  url: string;
}

export function BodyModel({ url }: BodyModelProps) {
  const { scene, animations } = useGLTF(url);
  const activeAnimation = useAnimationStore((s) => s.activeAnimation);
  const speed = useAnimationStore((s) => s.speed);
  const mixerRef = useRef<THREE.AnimationMixer | null>(null);
  const currentActionRef = useRef<THREE.AnimationAction | null>(null);

  const cloned = useMemo(() => {
    // SkeletonUtils.clone properly rebinds SkinnedMesh skeletons to the
    // cloned bone hierarchy. scene.clone(true) does NOT do this — the
    // SkinnedMesh skeleton references separate Skeleton.clone() bones,
    // so AnimationMixer drives the wrong bone set and mesh never deforms.
    let hasSkinned = false;
    scene.traverse((child) => {
      if ((child as THREE.SkinnedMesh).isSkinnedMesh) hasSkinned = true;
    });
    const c = hasSkinned ? skeletonClone(scene) : scene.clone(true);

    c.traverse((child) => {
      if (child instanceof THREE.SkinnedMesh) {
        // Animated vertices can leave the bounding box
        child.frustumCulled = false;
        child.geometry.computeVertexNormals();
        child.renderOrder = 1; // body renders after garments for stencil test

        const materials = Array.isArray(child.material)
          ? child.material
          : [child.material];

        for (const mat of materials) {
          if (mat instanceof THREE.MeshStandardMaterial) {
            mat.flatShading = false;
            mat.side = THREE.DoubleSide;
            // Stencil: only render where garment has NOT drawn (stencil != 1)
            mat.stencilWrite = false;
            mat.stencilRef = 1;
            mat.stencilFunc = THREE.NotEqualStencilFunc;
            mat.stencilFail = THREE.KeepStencilOp;
            mat.stencilZFail = THREE.KeepStencilOp;
            mat.stencilZPass = THREE.KeepStencilOp;
            mat.needsUpdate = true;
          }
        }
      } else if (child instanceof THREE.Mesh) {
        child.geometry.computeVertexNormals();
        child.renderOrder = 1; // body renders after garments for stencil test

        const materials = Array.isArray(child.material)
          ? child.material
          : [child.material];

        for (const mat of materials) {
          if (mat instanceof THREE.MeshStandardMaterial) {
            mat.flatShading = false;
            mat.side = THREE.DoubleSide;
            // Stencil: only render where garment has NOT drawn (stencil != 1)
            mat.stencilWrite = false;
            mat.stencilRef = 1;
            mat.stencilFunc = THREE.NotEqualStencilFunc;
            mat.stencilFail = THREE.KeepStencilOp;
            mat.stencilZFail = THREE.KeepStencilOp;
            mat.stencilZPass = THREE.KeepStencilOp;
            mat.needsUpdate = true;
          }
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

    // Stop current action
    if (currentActionRef.current) {
      currentActionRef.current.fadeOut(0.3);
    }

    if (activeAnimation === null) {
      // T-pose: stop all
      mixer.stopAllAction();
      currentActionRef.current = null;
      return;
    }

    // Find the clip by name
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
