/**
 * Full-area loading overlay with a spinner and optional message.
 */

import { Loader2 } from "lucide-react";

interface LoadingSpinnerProps {
  /** Message displayed below the spinner. */
  message?: string;
}

export function LoadingSpinner({
  message = "Loading...",
}: LoadingSpinnerProps) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-12">
      <Loader2 className="h-8 w-8 animate-spin text-brand-600" />
      <p className="text-sm text-gray-500">{message}</p>
    </div>
  );
}
