/**
 * Wizard step indicator.
 *
 * Renders a horizontal bar of numbered steps with labels.
 * The current step is highlighted; completed steps show a check.
 */

import { Check } from "lucide-react";

export interface Step {
  id: string;
  label: string;
}

interface StepperProps {
  steps: Step[];
  currentIndex: number;
}

export function Stepper({ steps, currentIndex }: StepperProps) {
  return (
    <nav aria-label="Progress" className="flex items-center gap-2">
      {steps.map((step, idx) => {
        const isCompleted = idx < currentIndex;
        const isCurrent = idx === currentIndex;

        return (
          <div key={step.id} className="flex items-center gap-2">
            {/* Step circle */}
            <div
              className={[
                "flex h-8 w-8 items-center justify-center rounded-full text-xs font-semibold",
                "transition-colors duration-200",
                isCompleted
                  ? "bg-brand-600 text-white"
                  : isCurrent
                    ? "border-2 border-brand-600 text-brand-600 bg-brand-50"
                    : "border-2 border-gray-300 text-gray-400 bg-white",
              ].join(" ")}
            >
              {isCompleted ? (
                <Check className="h-4 w-4" />
              ) : (
                idx + 1
              )}
            </div>

            {/* Label */}
            <span
              className={[
                "text-sm font-medium hidden sm:inline",
                isCurrent ? "text-brand-700" : isCompleted ? "text-gray-700" : "text-gray-400",
              ].join(" ")}
            >
              {step.label}
            </span>

            {/* Connector line */}
            {idx < steps.length - 1 && (
              <div
                className={[
                  "h-px w-8 mx-1",
                  isCompleted ? "bg-brand-500" : "bg-gray-300",
                ].join(" ")}
              />
            )}
          </div>
        );
      })}
    </nav>
  );
}
