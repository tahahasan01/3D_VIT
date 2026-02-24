/**
 * Body measurement input form.
 *
 * Collects all measurements needed to generate a 3D human body.
 * Provides gender presets that populate sensible defaults.
 */

import { useCallback, useState } from "react";
import { Ruler } from "lucide-react";
import { Button } from "../ui/Button";
import { Input } from "../ui/Input";
import { Select } from "../ui/Select";
import { Card } from "../ui/Card";
import type { BodyMeasurements, Gender } from "../../types/body";
import {
  DEFAULT_MALE_MEASUREMENTS,
  DEFAULT_FEMALE_MEASUREMENTS,
} from "../../types/body";

interface MeasurementFormProps {
  /** Whether the backend is currently generating. */
  isLoading: boolean;
  /** Called when the user submits valid measurements. */
  onSubmit: (measurements: BodyMeasurements) => void;
}

const GENDER_OPTIONS = [
  { value: "male", label: "Male" },
  { value: "female", label: "Female" },
];

/** Maps numeric input field keys to their display label and unit. */
const MEASUREMENT_FIELDS: {
  key: keyof Omit<BodyMeasurements, "gender">;
  label: string;
  min: number;
  max: number;
}[] = [
  { key: "height_cm", label: "Height", min: 140, max: 220 },
  { key: "chest_cm", label: "Chest", min: 60, max: 160 },
  { key: "waist_cm", label: "Waist", min: 50, max: 150 },
  { key: "hip_cm", label: "Hip", min: 60, max: 160 },
  { key: "shoulder_width_cm", label: "Shoulder Width", min: 30, max: 60 },
  { key: "arm_length_cm", label: "Arm Length", min: 40, max: 90 },
  { key: "inseam_cm", label: "Inseam", min: 60, max: 100 },
];

export function MeasurementForm({ isLoading, onSubmit }: MeasurementFormProps) {
  const [form, setForm] = useState<BodyMeasurements>(
    DEFAULT_MALE_MEASUREMENTS,
  );

  const handleGenderChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      const gender = e.target.value as Gender;
      setForm(
        gender === "male"
          ? DEFAULT_MALE_MEASUREMENTS
          : DEFAULT_FEMALE_MEASUREMENTS,
      );
    },
    [],
  );

  const handleFieldChange = useCallback(
    (key: keyof BodyMeasurements) =>
      (e: React.ChangeEvent<HTMLInputElement>) => {
        const value = parseFloat(e.target.value);
        if (!isNaN(value)) {
          setForm((prev) => ({ ...prev, [key]: value }));
        }
      },
    [],
  );

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      onSubmit(form);
    },
    [form, onSubmit],
  );

  return (
    <Card title="Body Measurements">
      <form onSubmit={handleSubmit} className="flex flex-col gap-4">
        <Select
          label="Gender"
          options={GENDER_OPTIONS}
          value={form.gender}
          onChange={handleGenderChange}
        />

        <div className="grid grid-cols-2 gap-3">
          {MEASUREMENT_FIELDS.map((field) => (
            <Input
              key={field.key}
              label={field.label}
              type="number"
              suffix="cm"
              min={field.min}
              max={field.max}
              step={0.5}
              value={(form[field.key] as number | undefined) ?? ""}
              onChange={handleFieldChange(field.key)}
            />
          ))}
        </div>

        {form.gender === "male" && (
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={form.use_base_mesh ?? false}
              onChange={(e) =>
                setForm((prev) => ({ ...prev, use_base_mesh: e.target.checked }))
              }
              className="rounded border-gray-300"
            />
            <span className="text-sm text-gray-700">
              Use realistic base mesh (male body model)
            </span>
          </label>
        )}

        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={form.use_smpl ?? false}
            onChange={(e) =>
              setForm((prev) => ({ ...prev, use_smpl: e.target.checked }))
            }
            className="rounded border-gray-300"
          />
          <span className="text-sm text-gray-700">
            Use SMPL body (realistic, animation-ready; requires SMPL files in server)
          </span>
        </label>

        <Button type="submit" isLoading={isLoading} className="mt-2">
          <Ruler className="h-4 w-4" />
          Generate 3D Body
        </Button>
      </form>
    </Card>
  );
}
