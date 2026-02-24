/**
 * Composite try-on scene.
 *
 * Renders the body model (if generated) and all visible garment
 * layers within a single Three.js group so they share the same
 * coordinate space and can be inspected together.
 */

import { Suspense } from "react";
import { useBodyStore } from "../../store/bodyStore";
import { useGarmentStore } from "../../store/garmentStore";
import { BodyModel } from "./BodyModel";
import { GarmentModel } from "./GarmentModel";

export function TryOnScene() {
  const bodyModelUrl = useBodyStore((s) => s.modelUrl);
  const garments = useGarmentStore((s) => s.garments);

  const visibleGarments = garments.filter((g) => g.visible);

  return (
    <group>
      <Suspense fallback={null}>
        {bodyModelUrl && <BodyModel url={bodyModelUrl} />}

        {visibleGarments.map((garment) => (
          <GarmentModel key={garment.id} url={garment.modelUrl} />
        ))}
      </Suspense>
    </group>
  );
}
