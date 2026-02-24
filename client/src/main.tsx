/**
 * Application entry point.
 *
 * Mounts the React root and imports global styles.
 */

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./index.css";

const rootElement = document.getElementById("root");
if (!rootElement) {
  throw new Error("Root element not found. Check index.html for <div id='root'>.");
}

createRoot(rootElement).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
