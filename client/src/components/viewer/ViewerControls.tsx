/**
 * Toolbar overlay for the 3D viewer.
 *
 * Provides animation controls (T-Pose, Walk, Twirl), speed slider,
 * garment visibility toggles, camera reset, and screenshot.
 */

import { RotateCcw, Camera, Eye, EyeOff, Trash2 } from "lucide-react";
import { useGarmentStore } from "../../store/garmentStore";
import { useAnimationStore, type AnimationType } from "../../store/animationStore";
import { useBodyStore } from "../../store/bodyStore";
import { GARMENT_TYPE_LABELS } from "../../types/garment";

interface ViewerControlsProps {
  /** Callback to reset camera to default position. */
  onResetCamera?: () => void;
  /** Callback to capture and download a screenshot. */
  onScreenshot?: () => void;
  /** "overlay" = absolute-positioned over canvas, "sidebar" = static. */
  layout?: "overlay" | "sidebar";
}

export function ViewerControls({
  onResetCamera,
  onScreenshot,
  layout = "overlay",
}: ViewerControlsProps) {
  const garments = useGarmentStore((s) => s.garments);
  const toggleVisibility = useGarmentStore((s) => s.toggleVisibility);
  const removeGarment = useGarmentStore((s) => s.removeGarment);

  const modelType = useBodyStore((s) => s.modelType);
  const activeAnimation = useAnimationStore((s) => s.activeAnimation);
  const speed = useAnimationStore((s) => s.speed);
  const setAnimation = useAnimationStore((s) => s.setAnimation);
  const setSpeed = useAnimationStore((s) => s.setSpeed);

  const isSmpl = modelType === "smpl";

  const animButtons: { label: string; value: AnimationType }[] = [
    { label: "T-Pose", value: null },
    { label: "Walk", value: "walk" },
    { label: "Twirl", value: "twirl" },
  ];

  const isOverlay = layout === "overlay";

  return (
    <div className={isOverlay ? "absolute bottom-3 left-3 right-3 flex flex-col gap-2 pointer-events-none max-h-[40%] overflow-y-auto" : "flex flex-col gap-3"}>
      {/* Animation controls */}
      <div className={isOverlay ? "rounded-lg bg-white/80 p-2 backdrop-blur-sm border border-gray-200 shadow-sm pointer-events-auto" : "rounded-lg bg-white p-3 border border-gray-200 shadow-sm"}>
        <details className="group" open={!isOverlay}>
          <summary className="text-xs font-medium text-gray-500 cursor-pointer select-none px-1 list-none flex items-center gap-1">
            <span className="transition-transform group-open:rotate-90">&#9654;</span>
            Animation
          </summary>
          <div className="mt-1.5">
            <div className="flex gap-1 mb-2">
          {animButtons.map((btn) => {
            const disabled = btn.value !== null && !isSmpl;
            return (
              <button
                key={btn.label}
                onClick={() => !disabled && setAnimation(btn.value)}
                disabled={disabled}
                className={`flex-1 rounded-md px-2 py-1 text-xs font-medium transition-colors ${
                  disabled
                    ? "bg-gray-50 text-gray-300 cursor-not-allowed"
                    : activeAnimation === btn.value
                      ? "bg-indigo-600 text-white"
                      : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                }`}
              >
                {btn.label}
              </button>
            );
          })}
        </div>
        {!isSmpl && modelType != null && (
          <p className="text-[10px] text-amber-600 px-1 mb-1">
            Animations require SMPL body model
          </p>
        )}
        <div className="flex items-center gap-2 px-1">
          <span className="text-xs text-gray-500 whitespace-nowrap">
            Speed: {speed.toFixed(2)}x
          </span>
          <input
            type="range"
            min="0.25"
            max="2"
            step="0.25"
            value={speed}
            onChange={(e) => setSpeed(parseFloat(e.target.value))}
            className="flex-1 h-1 accent-indigo-600"
          />
        </div>
          </div>
        </details>
      </div>

      {/* Garment layer controls */}
      {garments.length > 0 && (
        <div className={isOverlay ? "rounded-lg bg-white/80 p-2 backdrop-blur-sm border border-gray-200 shadow-sm pointer-events-auto" : "rounded-lg bg-white p-3 border border-gray-200 shadow-sm"}>
          <details className="group" open>
            <summary className="text-xs font-medium text-gray-500 cursor-pointer select-none px-1 list-none flex items-center gap-1">
              <span className="transition-transform group-open:rotate-90">&#9654;</span>
              Garment Layers
            </summary>
          <div className="flex flex-col gap-1 mt-1.5">
            {garments.map((g) => (
              <div
                key={g.id}
                className="flex items-center justify-between rounded-md px-2 py-1 hover:bg-gray-50"
              >
                <span className="text-xs text-gray-700 truncate max-w-[140px]">
                  {GARMENT_TYPE_LABELS[g.type]} &mdash; {g.imageName}
                </span>
                <div className="flex items-center gap-1">
                  <button
                    onClick={() => toggleVisibility(g.id)}
                    className="p-1 rounded hover:bg-gray-200 text-gray-500"
                    title={g.visible ? "Hide garment" : "Show garment"}
                  >
                    {g.visible ? (
                      <Eye className="h-3.5 w-3.5" />
                    ) : (
                      <EyeOff className="h-3.5 w-3.5" />
                    )}
                  </button>
                  <button
                    onClick={() => removeGarment(g.id)}
                    className="p-1 rounded hover:bg-red-100 text-gray-500 hover:text-red-600"
                    title="Remove garment"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>
              </div>
            ))}
          </div>
          </details>
        </div>
      )}

      {/* Action buttons */}
      {onResetCamera && onScreenshot && (
      <div className={isOverlay ? "flex gap-2 pointer-events-auto" : "flex gap-2"}>
        <button
          onClick={onResetCamera}
          className="flex items-center gap-1.5 rounded-lg bg-white/80 px-3 py-1.5 text-xs font-medium text-gray-700 backdrop-blur-sm border border-gray-200 shadow-sm hover:bg-gray-50 transition-colors"
          title="Reset camera"
        >
          <RotateCcw className="h-3.5 w-3.5" />
          Reset View
        </button>
        <button
          onClick={onScreenshot}
          className="flex items-center gap-1.5 rounded-lg bg-white/80 px-3 py-1.5 text-xs font-medium text-gray-700 backdrop-blur-sm border border-gray-200 shadow-sm hover:bg-gray-50 transition-colors"
          title="Take screenshot"
        >
          <Camera className="h-3.5 w-3.5" />
          Screenshot
        </button>
      </div>
      )}
    </div>
  );
}
