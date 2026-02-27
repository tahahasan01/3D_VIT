/**
 * Main try-on page.
 *
 * Implements a 3-step wizard:
 *  1. Enter body measurements  ->  generate 3D body
 *  2. Upload garment image     ->  generate 3D garment
 *  3. Inspect the composed try-on in the 3D viewer
 *
 * The viewer is always visible on the right. The left panel
 * shows the form corresponding to the current step.
 */

import { useCallback, useState } from "react";
import { ArrowLeft, ArrowRight } from "lucide-react";
import { Button } from "../components/ui/Button";
import { Stepper, type Step } from "../components/ui/Stepper";
import { MeasurementForm } from "../components/forms/MeasurementForm";
import { GarmentUpload } from "../components/forms/GarmentUpload";
import { ModelViewer } from "../components/viewer/ModelViewer";
import { ViewerControls } from "../components/viewer/ViewerControls";
import { useBodyModel } from "../hooks/useBodyModel";
import { useGarmentModel } from "../hooks/useGarmentModel";
import type { BodyMeasurements } from "../types/body";
import type { GarmentMeasurements, GarmentSubmitOptions } from "../types/garment";

const STEPS: Step[] = [
  { id: "measurements", label: "Your 3D model" },
  { id: "garment", label: "Upload garment" },
  { id: "tryon", label: "3D Try-On" },
];

export function TryOnPage() {
  const [stepIndex, setStepIndex] = useState(0);
  const { modelUrl: bodyUrl, isGenerating, error: bodyError, generate } = useBodyModel();
  const { isProcessing, error: garmentError, process } = useGarmentModel();

  // -- Step 1 handler --------------------------------------------------------
  const handleBodySubmit = useCallback(
    async (measurements: BodyMeasurements) => {
      await generate(measurements);
      setStepIndex(1);
    },
    [generate],
  );

  // -- Step 2 handler --------------------------------------------------------
  const handleGarmentSubmit = useCallback(
    async (
      image: File,
      measurements: GarmentMeasurements,
      options?: GarmentSubmitOptions,
      additionalImages?: File[],
    ) => {
      await process(image, measurements, options, additionalImages);
      setStepIndex(2);
    },
    [process],
  );

  // -- Navigation helpers ----------------------------------------------------
  const canGoBack = stepIndex > 0;
  const canGoForward =
    (stepIndex === 0 && bodyUrl !== null) ||
    (stepIndex === 1);

  return (
    <div className="flex h-full flex-col">
      {/* Stepper header */}
      <div className="flex items-center justify-between border-b border-gray-200 bg-white px-6 py-3">
        <Stepper steps={STEPS} currentIndex={stepIndex} />

        <div className="flex items-center gap-2">
          {canGoBack && (
            <Button
              variant="secondary"
              size="sm"
              onClick={() => setStepIndex((i) => i - 1)}
            >
              <ArrowLeft className="h-4 w-4" />
              Back
            </Button>
          )}
          {canGoForward && stepIndex < STEPS.length - 1 && (
            <Button
              size="sm"
              onClick={() => setStepIndex((i) => i + 1)}
            >
              Next
              <ArrowRight className="h-4 w-4" />
            </Button>
          )}
        </div>
      </div>

      {/* Content: sidebar + viewer + right controls */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left panel (forms) */}
        <aside className="w-[340px] shrink-0 overflow-y-auto border-r border-gray-200 bg-gray-50 p-5">
          {stepIndex === 0 && (
            <>
              <p className="text-sm font-medium text-gray-700 mb-3">
                Try on with your 3D model
              </p>
              <p className="text-xs text-gray-500 mb-4">
                Generate a 3D body from your measurements; then we&apos;ll fit the garment image on it.
              </p>
              <MeasurementForm
                isLoading={isGenerating}
                onSubmit={handleBodySubmit}
              />
            </>
          )}

          {stepIndex === 1 && (
            <>
              <p className="text-sm font-medium text-gray-700 mb-3">
                Upload your garment image
              </p>
              <p className="text-xs text-gray-500 mb-4">
                We&apos;ll fit this garment on your 3D body and show it in the viewer.
              </p>
              <GarmentUpload
                isLoading={isProcessing}
                onSubmit={handleGarmentSubmit}
              />
              <p className="mt-3 text-xs text-gray-400 text-center">
                You can add more garments; each is layered on the 3D body.
              </p>
            </>
          )}

          {stepIndex === 2 && (
            <div className="flex flex-col gap-4">
              <div className="rounded-xl border border-gray-200 bg-white p-5">
                <h3 className="text-sm font-semibold text-gray-800 mb-2">
                  3D Try-On
                </h3>
                <p className="text-xs text-gray-500 leading-relaxed">
                  Your garment is fitted on your 3D model. Rotate and zoom in the viewer
                  to inspect the fit from every angle.
                </p>
                <p className="mt-3 text-xs text-gray-500 leading-relaxed">
                  Use the layer controls below the viewer to toggle garments or add more.
                </p>
              </div>

              {/* Quick-add another garment */}
              <GarmentUpload
                isLoading={isProcessing}
                onSubmit={handleGarmentSubmit}
              />
            </div>
          )}

          {/* Error feedback */}
          {(bodyError || garmentError) && (
            <div className="mt-4 rounded-lg border border-red-200 bg-red-50 p-3">
              <p className="text-xs text-red-600">
                {bodyError || garmentError}
              </p>
            </div>
          )}
        </aside>

        {/* Center panel (3D viewer) */}
        <section className="flex-1 p-4">
          <ModelViewer showOverlayControls={false} />
        </section>

        {/* Right panel (animation + garment controls) */}
        <aside className="w-[280px] shrink-0 overflow-y-auto border-l border-gray-200 bg-gray-50 p-4">
          <ViewerControls layout="sidebar" />
        </aside>
      </div>
    </div>
  );
}
