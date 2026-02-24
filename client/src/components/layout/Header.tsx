/**
 * Application header bar.
 */

import { Link, useLocation } from "react-router-dom";
import { Box, Shirt } from "lucide-react";
import { APP_NAME, APP_TAGLINE } from "../../utils/constants";

export function Header() {
  const location = useLocation();
  const isTryOn = location.pathname === "/tryon";

  return (
    <header className="flex items-center justify-between border-b border-gray-200 bg-white px-6 py-3 shadow-sm">
      <Link to="/" className="flex items-center gap-3">
        <div className="flex items-center justify-center h-9 w-9 rounded-lg bg-brand-600">
          <Box className="h-5 w-5 text-white" />
        </div>
        <div>
          <h1 className="text-lg font-bold text-gray-900 leading-tight">
            {APP_NAME}
          </h1>
          <p className="text-xs text-gray-500">{APP_TAGLINE}</p>
        </div>
      </Link>
      <nav className="flex items-center gap-1">
        <Link
          to="/tryon"
          className={`flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
            isTryOn
              ? "bg-brand-100 text-brand-700"
              : "text-gray-600 hover:bg-gray-100 hover:text-gray-900"
          }`}
        >
          <Shirt className="h-4 w-4" />
          Virtual Try-On
        </Link>
      </nav>
    </header>
  );
}
