/**
 * Reusable Card component with optional title.
 */

import type { ReactNode } from "react";

interface CardProps {
  /** Optional heading rendered at the top of the card. */
  title?: string;
  /** Card body content. */
  children: ReactNode;
  /** Additional CSS classes. */
  className?: string;
}

export function Card({ title, children, className = "" }: CardProps) {
  return (
    <div
      className={[
        "rounded-xl border border-gray-200 bg-white shadow-sm",
        className,
      ].join(" ")}
    >
      {title && (
        <div className="border-b border-gray-100 px-5 py-3">
          <h3 className="text-sm font-semibold text-gray-800">{title}</h3>
        </div>
      )}
      <div className="p-5">{children}</div>
    </div>
  );
}
