/**
 * Reusable Input component with label, optional suffix, and error state.
 */

import { type InputHTMLAttributes, forwardRef } from "react";

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  /** Visible label text. */
  label: string;
  /** Optional unit suffix displayed inside the input (e.g. "cm"). */
  suffix?: string;
  /** Error message to display below the input. */
  error?: string;
}

export const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ label, suffix, error, id, className = "", ...rest }, ref) => {
    const inputId = id ?? label.toLowerCase().replace(/\s+/g, "-");

    return (
      <div className={`flex flex-col gap-1 ${className}`}>
        <label
          htmlFor={inputId}
          className="text-sm font-medium text-gray-700"
        >
          {label}
        </label>

        <div className="relative">
          <input
            ref={ref}
            id={inputId}
            className={[
              "w-full rounded-lg border bg-white px-3 py-2 text-sm text-gray-900",
              "placeholder:text-gray-400",
              "focus:outline-none focus:ring-2 focus:ring-brand-500 focus:border-brand-500",
              "transition-colors duration-150",
              suffix ? "pr-10" : "",
              error
                ? "border-red-400 focus:ring-red-500 focus:border-red-500"
                : "border-gray-300",
            ].join(" ")}
            {...rest}
          />

          {suffix && (
            <span className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-sm text-gray-400">
              {suffix}
            </span>
          )}
        </div>

        {error && (
          <p className="text-xs text-red-500">{error}</p>
        )}
      </div>
    );
  },
);

Input.displayName = "Input";
