import * as React from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { AppShell } from "@/components/layout/app-shell";
import { ToastProvider, ToastViewport, Toast, ToastClose, ToastTitle, ToastDescription, ToastIcon } from "@/components/ui/toast";
import Dashboard from "@/pages/dashboard";
import ReIDSearch from "@/pages/reid-search";
import RouteReconstruction from "@/pages/route-reconstruction";
import SightingTimeline from "@/pages/sighting-timeline";
import CameraLocations from "@/pages/camera-locations";
import WatchlistPersons from "@/pages/watchlist-persons";
import PipelineVideoStudio from "@/pages/pipeline-video-studio";

export default function App() {
  return (
    <BrowserRouter>
      <ToastProvider>
        <AppShell>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/reid-search" element={<ReIDSearch />} />
            <Route path="/route-reconstruction" element={<RouteReconstruction />} />
            <Route path="/sighting-timeline" element={<SightingTimeline />} />
            <Route path="/camera-locations" element={<CameraLocations />} />
            <Route path="/watchlist" element={<WatchlistPersons />} />
            <Route path="/pipeline-studio" element={<PipelineVideoStudio />} />
          </Routes>
        </AppShell>
        <ToastViewport />
      </ToastProvider>
    </BrowserRouter>
  );
}
