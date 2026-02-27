/**
 * Renders a single 3D garment GLB in the scene with PBR fabric materials.
 *
 * Preserves the embedded texture (baseColorTexture) from the GLB so the
 * garment displays the uploaded photo. Overlays tileable fabric normal /
 * roughness maps to simulate real cloth weave. Uses MeshPhysicalMaterial
 * with sheen for a natural fabric look.
 *
 * When the GLB contains a skeleton and animations (conforming garment on
 * SMPL), plays animations in sync with the body via the shared animation
 * store.
 */

import { useEffect, useMemo, useRef } from "react";
import { useGLTF, useTexture } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { clone as skeletonClone } from "three/examples/jsm/utils/SkeletonUtils.js";
import { useAnimationStore } from "../../store/animationStore";
import type { GarmentType } from "../../types/garment";

const STATIC_POSES = new Set(["natural_stand", "a_pose", "t_pose"]);

const AO_KNIT = "/textures/knit/Fabric007_1K-JPG_AmbientOcclusion.jpg";
const WRINKLE_BUMP = "/textures/wrinkles/wrinkle_bump.png";

const FABRIC_TEXTURES: Record<GarmentType, { normal: string; roughness: string; ao: string; useAo: boolean }> = {
  tshirt: {
    normal: "/textures/cotton/Fabric005_1K-JPG_NormalGL.jpg",
    roughness: "/textures/cotton/Fabric005_1K-JPG_Roughness.jpg",
    ao: AO_KNIT,
    useAo: false,
  },
  polo: {
    normal: "/textures/cotton/Fabric005_1K-JPG_NormalGL.jpg",
    roughness: "/textures/cotton/Fabric005_1K-JPG_Roughness.jpg",
    ao: AO_KNIT,
    useAo: false,
  },
  button_down: {
    normal: "/textures/cotton/Fabric005_1K-JPG_NormalGL.jpg",
    roughness: "/textures/cotton/Fabric005_1K-JPG_Roughness.jpg",
    ao: AO_KNIT,
    useAo: false,
  },
  hoodie: {
    normal: "/textures/knit/Fabric007_1K-JPG_NormalGL.jpg",
    roughness: "/textures/knit/Fabric007_1K-JPG_Roughness.jpg",
    ao: AO_KNIT,
    useAo: true,
  },
  jacket: {
    normal: "/textures/denim/Fabric072_1K-JPG_NormalGL.jpg",
    roughness: "/textures/denim/Fabric072_1K-JPG_Roughness.jpg",
    ao: AO_KNIT,
    useAo: false,
  },
  pants: {
    normal: "/textures/denim/Fabric072_1K-JPG_NormalGL.jpg",
    roughness: "/textures/denim/Fabric072_1K-JPG_Roughness.jpg",
    ao: AO_KNIT,
    useAo: false,
  },
  dress: {
    normal: "/textures/knit/Fabric007_1K-JPG_NormalGL.jpg",
    roughness: "/textures/knit/Fabric007_1K-JPG_Roughness.jpg",
    ao: AO_KNIT,
    useAo: true,
  },
};

const FABRIC_TILE = 6;

function configureTile(tex: THREE.Texture) {
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(FABRIC_TILE, FABRIC_TILE);
  tex.needsUpdate = true;
}

interface GarmentModelProps {
  url: string;
  garmentType?: GarmentType;
}

export function GarmentModel({ url, garmentType = "tshirt" }: GarmentModelProps) {
  const { scene, animations } = useGLTF(url);
  const activeAnimation = useAnimationStore((s) => s.activeAnimation);
  const speed = useAnimationStore((s) => s.speed);
  const mixerRef = useRef<THREE.AnimationMixer | null>(null);
  const currentActionRef = useRef<THREE.AnimationAction | null>(null);

  const paths = FABRIC_TEXTURES[garmentType] ?? FABRIC_TEXTURES.tshirt;
  const normalMap = useTexture(paths.normal);
  const roughnessMap = useTexture(paths.roughness);
  const aoTex = useTexture(paths.ao);
  const wrinkleBump = useTexture(WRINKLE_BUMP);

  useMemo(() => {
    configureTile(normalMap);
    normalMap.colorSpace = THREE.LinearSRGBColorSpace;
    configureTile(roughnessMap);
    roughnessMap.colorSpace = THREE.LinearSRGBColorSpace;
    configureTile(aoTex);
    aoTex.colorSpace = THREE.LinearSRGBColorSpace;
    configureTile(wrinkleBump);
    wrinkleBump.colorSpace = THREE.LinearSRGBColorSpace;
  }, [normalMap, roughnessMap, aoTex, wrinkleBump]);

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
        child.renderOrder = 0;

        const hasVertexColors =
          child.geometry.attributes.color != null &&
          child.geometry.attributes.color.count > 0;

        const oldMaterials = Array.isArray(child.material)
          ? child.material
          : [child.material];

        const newMaterials = oldMaterials.map((mat) => {
          if (!mat) return mat;

          const params: THREE.MeshPhysicalMaterialParameters = {
            side: THREE.DoubleSide,
            depthWrite: true,
            depthTest: true,
            transparent: false,
            opacity: 1.0,

            normalMap,
            normalScale: new THREE.Vector2(0.35, 0.35),

            bumpMap: wrinkleBump,
            bumpScale: 0.12,

            roughnessMap,
            roughness: 0.85,
            metalness: 0,

            sheen: 0.4,
            sheenColor: new THREE.Color(0xffffff),
            sheenRoughness: 0.6,

            envMapIntensity: 1.2,
            flatShading: false,
          };

          if (paths.useAo) {
            params.aoMap = aoTex;
            params.aoMapIntensity = 0.6;
          }

          if (mat.map) {
            mat.map.colorSpace = THREE.SRGBColorSpace;
            mat.map.anisotropy = 4;
            mat.map.needsUpdate = true;
            params.map = mat.map;
          }

          if (hasVertexColors) {
            params.vertexColors = true;
            params.color = new THREE.Color(0xffffff);
          } else if (mat.color) {
            params.color = mat.color.clone();
          }

          const physical = new THREE.MeshPhysicalMaterial(params);

          physical.stencilWrite = true;
          physical.stencilRef = 1;
          physical.stencilFunc = THREE.AlwaysStencilFunc;
          physical.stencilZPass = THREE.ReplaceStencilOp;
          physical.stencilZFail = THREE.KeepStencilOp;
          physical.stencilFail = THREE.KeepStencilOp;

          return physical;
        });

        child.material =
          newMaterials.length === 1 ? newMaterials[0] : newMaterials;
      }
    });

    return c;
  }, [scene, normalMap, roughnessMap, aoTex, wrinkleBump, paths.useAo]);

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
