/**
 * Root application component.
 *
 * Sets up client-side routing and wraps all pages in the shared layout.
 */

import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Layout } from "./components/layout/Layout";
import { HomePage } from "./pages/HomePage";
import { TryOnPage } from "./pages/TryOnPage";

export default function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/tryon" element={<TryOnPage />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}
