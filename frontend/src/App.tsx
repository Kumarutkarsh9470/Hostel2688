import { BrowserRouter, Routes, Route } from "react-router-dom";
import { lazy, Suspense } from "react";
import { Layout } from "./components/Layout";
import { HomePage } from "./pages/HomePage";
import { ArchitecturePage } from "./pages/ArchitecturePage";
import { SparsityPage } from "./pages/SparsityPage";
import { MonosemanticityPage } from "./pages/MonosemanticityPage";
import { HebbianPage } from "./pages/HebbianPage";
import { MergePage } from "./pages/MergePage";
import { FindingsPage } from "./pages/FindingsPage";
import { LearnBDHPage } from "./pages/LearnBDHPage";

// Lazy-load 3D-heavy pages to isolate Three.js and prevent app-wide crashes
const GraphPage = lazy(() =>
  import("./pages/GraphPage").then((m) => ({ default: m.GraphPage }))
);
const GraphTest = lazy(() =>
  import("./pages/GraphTest").then((m) => ({ default: m.GraphTest }))
);

function Loading() {
  return (
    <div className="flex items-center justify-center h-full text-[#4A5568]">
      Loading 3D view…
    </div>
  );
}

function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Diagnostic route — remove after testing */}
        <Route
          path="/graph-test"
          element={
            <Suspense fallback={<Loading />}>
              <GraphTest />
            </Suspense>
          }
        />
        <Route path="/" element={<Layout />}>
          <Route index element={<HomePage />} />
          <Route path="architecture" element={<ArchitecturePage />} />
          <Route path="sparsity" element={<SparsityPage />} />
          <Route
            path="graph"
            element={
              <Suspense fallback={<Loading />}>
                <GraphPage />
              </Suspense>
            }
          />
          <Route path="monosemanticity" element={<MonosemanticityPage />} />
          <Route path="hebbian" element={<HebbianPage />} />
          <Route path="merge" element={<MergePage />} />
          <Route path="findings" element={<FindingsPage />} />
          <Route path="learn" element={<LearnBDHPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
