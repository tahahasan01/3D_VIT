/**
 * Landing page.
 *
 * Brief introduction to VIT with a CTA that navigates to the try-on flow.
 */

import { useNavigate } from "react-router-dom";
import { Box, Shirt, RotateCw, Layers } from "lucide-react";
import { Button } from "../components/ui/Button";
import { APP_NAME, APP_TAGLINE } from "../utils/constants";

const FEATURES = [
  {
    icon: Box,
    title: "3D model generated",
    desc: "Enter your measurements and we generate a 3D body model.",
  },
  {
    icon: Shirt,
    title: "Upload garment image",
    desc: "Upload a photo of the outfit; we turn it into a 3D garment and fit it on your body.",
  },
  {
    icon: Layers,
    title: "Fit on body",
    desc: "The garment is draped and textured on your 3D model so it matches the image.",
  },
  {
    icon: RotateCw,
    title: "View in 3D",
    desc: "Rotate, zoom, and inspect the try-on from every angle in the viewer.",
  },
];

export function HomePage() {
  const navigate = useNavigate();

  return (
    <div className="flex flex-col items-center justify-center px-6 py-16 text-center">
      {/* Hero */}
      <div className="flex items-center justify-center h-16 w-16 rounded-2xl bg-brand-600 mb-6 shadow-lg">
        <Box className="h-8 w-8 text-white" />
      </div>
      <h2 className="text-3xl font-bold text-gray-900 sm:text-4xl">
        {APP_NAME} &mdash; {APP_TAGLINE}
      </h2>
      <p className="mt-3 max-w-lg text-gray-500">
        Generate your 3D body from measurements, upload a garment image, and we fit it
        on your model. Inspect the result in 3D from every angle.
      </p>
      <Button
        size="lg"
        className="mt-8"
        onClick={() => navigate("/tryon")}
      >
        Virtual Try-On
      </Button>

      {/* Feature grid */}
      <div className="mt-16 grid max-w-3xl grid-cols-1 gap-6 sm:grid-cols-2">
        {FEATURES.map((f) => (
          <div
            key={f.title}
            className="flex flex-col items-center rounded-xl border border-gray-200 bg-white p-6 shadow-sm"
          >
            <f.icon className="mb-3 h-7 w-7 text-brand-600" />
            <h3 className="text-sm font-semibold text-gray-800">{f.title}</h3>
            <p className="mt-1 text-xs text-gray-500 leading-relaxed">
              {f.desc}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
