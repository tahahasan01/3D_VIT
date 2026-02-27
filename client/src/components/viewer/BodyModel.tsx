/**
 * Renders the 3D body GLB in the scene.
 *
 * Loads the model from a blob URL. When the GLB contains a skeleton
 * and animations (SMPL skinned mesh), creates an AnimationMixer to
 * play walk/twirl clips based on the animation store.
 *
 * Material is a matte gray (roughness 0.65) matching the classic SMPL
 * clay-mannequin look. Face texture (material index 1) keeps its own
 * color and lower roughness.
 */

import { useEffect, useMemo, useRef } from "react";
import { useGLTF } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { clone as skeletonClone } from "three/examples/jsm/utils/SkeletonUtils.js";
import { useAnimationStore } from "../../store/animationStore";

const STATIC_POSES = new Set(["natural_stand", "a_pose", "t_pose"]);

interface BodyModelProps {
  url: string;
}

export function BodyModel({ url }: BodyModelProps) {
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
      const isSkinned = (child as THREE.SkinnedMesh).isSkinnedMesh;
      if (child instanceof THREE.Mesh || isSkinned) {
        child.frustumCulled = false;
        child.geometry.computeVertexNormals();

        const materials = Array.isArray(child.material)
          ? child.material
          : [child.material];

        materials.forEach((mat, idx) => {
          if (mat instanceof THREE.MeshStandardMaterial) {
            mat.flatShading = false;
            mat.metalness = 0.0;
            mat.side = THREE.DoubleSide;
            if (idx === 0) {
              // Body primitive: matte gray clay look (matches SMPL reference)
              mat.roughness = 0.65;
            } else {
              // Face primitive: slightly less matte so features read clearly
              mat.roughness = 0.55;
            }
            mat.needsUpdate = true;
          }
        });
      }
    });

    return c;
  }, [scene]);

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

  useEffect(() => {
    const mixer = mixerRef.current;
    if (!mixer || !animations || animations.length === 0) return;

    mixer.stopAllAction();
    currentActionRef.current = null;

    const clipName = activeAnimation ?? "natural_stand";
    const isStatic = STATIC_POSES.has(clipName);

    const clip = animations.find((a) => a.name === clipName);
    if (!clip) return;

    const action = mixer.clipAction(clip);
    action.reset();

    if (isStatic) {
      action.loop = THREE.LoopOnce;
      action.clampWhenFinished = true;
    } else {
      action.loop = THREE.LoopRepeat;
    }

    action.play();
    currentActionRef.current = action;

    mixer.update(0);
  }, [activeAnimation, animations]);

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
