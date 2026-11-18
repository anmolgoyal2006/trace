import * as React from "react";
import { useLocation } from "react-router-dom";
import { RefreshCw, Menu } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

const ROUTE_TITLES: Record<string, { title: string; subtitle: string }> = {
  "/": { title: "Dashboard", subtitle: "Operational overview" },
  "/reid-search": { title: "Re-ID Search", subtitle: "Match identity across camera galleries" },
  "/route-reconstruction": { title: "Route Reconstruction", subtitle: "Spatial-temporal trajectory analysis" },
  "/sighting-timeline": { title: "Sighting Timeline", subtitle: "Multi-camera event timeline" },
  "/camera-locations": { title: "Camera Locations", subtitle: "Camera network management" },
  "/watchlist": { title: "Watchlist & Persons", subtitle: "Target profile directory" },
  "/pipeline-studio": { title: "Pipeline Video Studio", subtitle: "Video processing pipeline" },
};

interface TopbarProps {
  onToggleSidebar: () => void;
}

export function Topbar({ onToggleSidebar }: TopbarProps) {
  const location = useLocation();
  const routeKey = Object.keys(ROUTE_TITLES).find(
    (key) => key === "/" ? location.pathname === "/" : location.pathname.startsWith(key),
  ) || "/";
  const { title, subtitle } = ROUTE_TITLES[routeKey];

  return (
    <header className="sticky top-0 z-30 flex h-16 items-center justify-between border-b border-border bg-surface/95 px-6 backdrop-blur-sm">
      <div className="flex items-center gap-3">
        <button
          onClick={onToggleSidebar}
          className="text-ink-muted transition-colors hover:text-ink lg:hidden"
        >
          <Menu className="h-5 w-5" />
        </button>
        <div>
          <h2 className="text-lg font-semibold text-ink">{title}</h2>
          <p className="hidden text-xs text-ink-muted sm:block">{subtitle}</p>
        </div>
      </div>

      <div className="flex items-center gap-2">
        <TooltipProvider delayDuration={200}>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => window.location.reload()}
              >
                <RefreshCw className="h-4 w-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Refresh</TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>
    </header>
  );
}
