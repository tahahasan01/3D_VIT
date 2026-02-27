/**
 * Garment image upload form.
 *
 * Upload a flat garment image, select type, and enter measurements.
 * The garment is built from a parametric/conforming template and textured from the image.
 */

import { useCallback, useMemo, useState } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, Shirt, ImagePlus, X } from "lucide-react";
import { Button } from "../ui/Button";
import { Input } from "../ui/Input";
import { Select } from "../ui/Select";
import { Card } from "../ui/Card";
import type {
  GarmentMeasurements,
  GarmentType,
  GarmentSubmitOptions,
} from "../../types/garment";
import {
  GARMENT_TYPE_LABELS,
  DEFAULT_GARMENT_MEASUREMENTS,
} from "../../types/garment";
import { ACCEPTED_IMAGE_TYPES, MAX_UPLOAD_SIZE_MB } from "../../utils/constants";

interface GarmentUploadProps {
  /** Whether the backend is currently processing. */
  isLoading: boolean;
  /** Called with the selected file, measurements, and optional additional angle images on submit. */
  onSubmit: (
    image: File,
    measurements: GarmentMeasurements,
    options?: GarmentSubmitOptions,
    additionalImages?: File[],
  ) => void;
}

const GARMENT_TYPE_OPTIONS = Object.entries(GARMENT_TYPE_LABELS).map(
  ([value, label]) => ({ value, label }),
);

/** Upper-body field set (shared by tshirt, polo, button_down, hoodie, jacket). */
const UPPER_BODY_FIELDS: {
  key: keyof GarmentMeasurements;
  label: string;
  min: number;
  max: number;
}[] = [
  { key: "chest_cm", label: "Chest", min: 60, max: 160 },
  { key: "length_cm", label: "Length", min: 40, max: 100 },
  { key: "sleeve_length_cm", label: "Sleeve Length", min: 10, max: 90 },
];

/** Field definitions per garment type. */
const FIELDS_BY_TYPE: Record<
  GarmentType,
  { key: keyof GarmentMeasurements; label: string; min: number; max: number }[]
> = {
  tshirt: UPPER_BODY_FIELDS,
  polo: UPPER_BODY_FIELDS,
  button_down: UPPER_BODY_FIELDS,
  hoodie: UPPER_BODY_FIELDS,
  jacket: UPPER_BODY_FIELDS,
  pants: [
    { key: "waist_cm", label: "Waist", min: 50, max: 150 },
    { key: "hip_cm", label: "Hip", min: 60, max: 160 },
    { key: "inseam_cm", label: "Inseam", min: 20, max: 100 },
    { key: "length_cm", label: "Total Length", min: 50, max: 130 },
  ],
  dress: [
    { key: "chest_cm", label: "Bust", min: 60, max: 160 },
    { key: "waist_cm", label: "Waist", min: 50, max: 150 },
    { key: "hip_cm", label: "Hip", min: 60, max: 160 },
    { key: "length_cm", label: "Length", min: 50, max: 150 },
  ],
};

export function GarmentUpload({ isLoading, onSubmit }: GarmentUploadProps) {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [additionalFiles, setAdditionalFiles] = useState<File[]>([]);
  const [additionalPreviews, setAdditionalPreviews] = useState<string[]>([]);
  const [measurements, setMeasurements] = useState<GarmentMeasurements>(
    DEFAULT_GARMENT_MEASUREMENTS.tshirt,
  );

  // ---- Primary dropzone setup ------------------------------------------------
  const onDrop = useCallback((accepted: File[]) => {
    const first = accepted[0];
    if (!first) return;
    setFile(first);
    setPreview(URL.createObjectURL(first));
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPTED_IMAGE_TYPES,
    maxSize: MAX_UPLOAD_SIZE_MB * 1024 * 1024,
    multiple: false,
  });

  // ---- Additional angles dropzone setup --------------------------------------
  const onDropAdditional = useCallback((accepted: File[]) => {
    setAdditionalFiles((prev) => [...prev, ...accepted]);
    setAdditionalPreviews((prev) => [
      ...prev,
      ...accepted.map((f) => URL.createObjectURL(f)),
    ]);
  }, []);

  const {
    getRootProps: getAdditionalRootProps,
    getInputProps: getAdditionalInputProps,
    isDragActive: isAdditionalDragActive,
  } = useDropzone({
    onDrop: onDropAdditional,
    accept: ACCEPTED_IMAGE_TYPES,
    maxSize: MAX_UPLOAD_SIZE_MB * 1024 * 1024,
    multiple: true,
  });

  const removeAdditionalImage = useCallback((index: number) => {
    setAdditionalFiles((prev) => prev.filter((_, i) => i !== index));
    setAdditionalPreviews((prev) => {
      URL.revokeObjectURL(prev[index]);
      return prev.filter((_, i) => i !== index);
    });
  }, []);

  // ---- Handlers -------------------------------------------------------------
  const handleTypeChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      const type = e.target.value as GarmentType;
      setMeasurements(DEFAULT_GARMENT_MEASUREMENTS[type]);
    },
    [],
  );

  const handleFieldChange = useCallback(
    (key: keyof GarmentMeasurements) =>
      (e: React.ChangeEvent<HTMLInputElement>) => {
        const value = parseFloat(e.target.value);
        if (!isNaN(value)) {
          setMeasurements((prev) => ({ ...prev, [key]: value }));
        }
      },
    [],
  );

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      if (!file) return;
      onSubmit(
        file,
        measurements,
        undefined,
        additionalFiles.length > 0 ? additionalFiles : undefined,
      );
    },
    [file, measurements, additionalFiles, onSubmit],
  );

  const fields = useMemo(
    () => FIELDS_BY_TYPE[measurements.garment_type],
    [measurements.garment_type],
  );

  return (
    <Card title="Upload garment image">
      <form onSubmit={handleSubmit} className="flex flex-col gap-4">
        {/* Dropzone */}
        <div
          {...getRootProps()}
          className={[
            "flex flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed p-6 cursor-pointer",
            "transition-colors duration-150",
            isDragActive
              ? "border-brand-400 bg-brand-50"
              : "border-gray-300 hover:border-gray-400 bg-gray-50",
          ].join(" ")}
        >
          <input {...getInputProps()} />

          {preview ? (
            <img
              src={preview}
              alt="Garment preview"
              className="h-32 w-32 object-contain rounded"
            />
          ) : (
            <Upload className="h-8 w-8 text-gray-400" />
          )}

          <p className="text-sm text-gray-500 text-center">
            {isDragActive
              ? "Drop the image here"
              : file
                ? file.name
                : "Drag & drop a flat garment image, or click to browse"}
          </p>
        </div>

        <p className="text-xs text-gray-500">
          Use a flat photo of the garment. We&apos;ll build a 3D template from your measurements and apply this image as the texture. Complete step 1 first so the garment fits your 3D body.
        </p>

        {/* Additional angle images */}
        <div className="flex flex-col gap-2">
          <p className="text-xs font-medium text-gray-500">
            Add more angles (optional)
          </p>

          {additionalPreviews.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {additionalPreviews.map((src, i) => (
                <div key={src} className="relative group">
                  <img
                    src={src}
                    alt={`Angle ${i + 1}`}
                    className="h-16 w-16 rounded border border-gray-200 object-cover"
                  />
                  <button
                    type="button"
                    onClick={() => removeAdditionalImage(i)}
                    className="absolute -right-1.5 -top-1.5 hidden rounded-full bg-gray-700 p-0.5 text-white group-hover:block"
                  >
                    <X className="h-3 w-3" />
                  </button>
                </div>
              ))}
            </div>
          )}

          <div
            {...getAdditionalRootProps()}
            className={[
              "flex items-center justify-center gap-2 rounded-md border border-dashed p-3 cursor-pointer",
              "transition-colors duration-150 text-xs",
              isAdditionalDragActive
                ? "border-brand-400 bg-brand-50"
                : "border-gray-200 hover:border-gray-300 bg-gray-50/50",
            ].join(" ")}
          >
            <input {...getAdditionalInputProps()} />
            <ImagePlus className="h-4 w-4 text-gray-400" />
            <span className="text-gray-400">
              {isAdditionalDragActive
                ? "Drop images here"
                : "Back, side, or detail shots"}
            </span>
          </div>
        </div>

        {/* Garment type selector */}
        <Select
          label="Garment Type"
          options={GARMENT_TYPE_OPTIONS}
          value={measurements.garment_type}
          onChange={handleTypeChange}
        />

        {/* Dynamic measurement fields */}
        <div className="grid grid-cols-2 gap-3">
          {fields.map((field) => (
            <Input
              key={field.key}
              label={field.label}
              type="number"
              suffix="cm"
              min={field.min}
              max={field.max}
              step={0.5}
              value={(measurements[field.key] as number | undefined) ?? ""}
              onChange={handleFieldChange(field.key)}
            />
          ))}
        </div>

        <Button
          type="submit"
          isLoading={isLoading}
          disabled={!file}
          className="mt-2"
        >
          <Shirt className="h-4 w-4" />
          Virtual try on
        </Button>
      </form>
    </Card>
  );
}
