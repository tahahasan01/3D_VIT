/**
 * Reusable Select (dropdown) component.
 */

import { type SelectHTMLAttributes, forwardRef } from "react";

interface Option {
  value: string;
  label: string;
}

interface SelectProps extends SelectHTMLAttributes<HTMLSelectElement> {
  /** Visible label text. */
  label: string;
  /** Dropdown options. */
  options: Option[];
  /** Error message to display below the select. */
  error?: string;
}

export const Select = forwardRef<HTMLSelectElement, SelectProps>(
  ({ label, options, error, id, className = "", ...rest }, ref) => {
    const selectId = id ?? label.toLowerCase().replace(/\s+/g, "-");

    return (
      <div className={`flex flex-col gap-1 ${className}`}>
        <label
          htmlFor={selectId}
          className="text-sm font-medium text-gray-700"
        >
          {label}
        </label>

        <select
          ref={ref}
          id={selectId}
          className={[
            "w-full rounded-lg border bg-white px-3 py-2 text-sm text-gray-900",
            "focus:outline-none focus:ring-2 focus:ring-brand-500 focus:border-brand-500",
            "transition-colors duration-150",
            error
              ? "border-red-400 focus:ring-red-500 focus:border-red-500"
              : "border-gray-300",
          ].join(" ")}
          {...rest}
        >
          {options.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>

        {error && (
          <p className="text-xs text-red-500">{error}</p>
        )}
      </div>
    );
  },
);

Select.displayName = "Select";
