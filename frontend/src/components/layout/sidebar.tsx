import * as React from "react";
import { NavLink, useLocation } from "react-router-dom";
import { motion, AnimatePresence } from "motion/react";
import {
  LayoutDashboard,
  Search,
  Route,
  Clock,
  MapPin,
  Users,
  Video,
  PanelLeftClose,
  PanelLeft,
  type LucideIcon,
} from "lucide-react";
import { wsClient } from "@/services/websocket";
import { cn } from "@/lib/utils";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

interface NavItem {
  to: string;
  label: string;
  icon: LucideIcon;
  description: string;
}

const NAV_ITEMS: NavItem[] = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard, description: "Operational overview" },
  { to: "/reid-search", label: "Re-ID Search", icon: Search, description: "Match identity across cameras" },
  { to: "/route-reconstruction", label: "Route Reconstruction", icon: Route, description: "Spatial-temporal trajectory" },
  { to: "/sighting-timeline", label: "Sighting Timeline", icon: Clock, description: "Multi-camera event timeline" },
  { to: "/camera-locations", label: "Camera Locations", icon: MapPin, description: "Camera network management" },
  { to: "/watchlist", label: "Watchlist & Persons", icon: Users, description: "Target profile directory" },
  { to: "/pipeline-studio", label: "Pipeline Video Studio", icon: Video, description: "Video processing pipeline" },
];

interface SidebarProps {
  collapsed: boolean;
  onToggle: () => void;
}

export function Sidebar({ collapsed, onToggle }: SidebarProps) {
  const location = useLocation();

  return (
    <TooltipProvider delayDuration={200}>
      <aside
        className={cn(
          "fixed left-0 top-0 z-40 flex h-screen flex-col border-r border-border bg-background-warm transition-[width] duration-200",
          collapsed ? "w-16" : "w-60",
        )}
      >
        {/* Logo / Wordmark */}
        <div className="flex h-16 items-center justify-between px-4 border-b border-border">
          <AnimatePresence mode="wait">
            {!collapsed ? (
              <motion.div
                key="expanded-logo"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex items-center gap-2.5"
              >
                <div className="flex h-8 w-8 items-center justify-center rounded-md bg-primary">
                  <span className="text-sm font-bold text-white">T</span>
                </div>
                <div>
                  <div className="text-base font-bold tracking-tight text-ink">TRACE</div>
                  <div className="text-[10px] font-medium text-ink-muted">Multi-Camera Intelligence</div>
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="collapsed-logo"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="mx-auto"
              >
                <div className="flex h-8 w-8 items-center justify-center rounded-md bg-primary">
                  <span className="text-sm font-bold text-white">T</span>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
          {!collapsed && (
            <button
              onClick={onToggle}
              className="text-ink-muted transition-colors hover:text-ink"
            >
              <PanelLeftClose className="h-4 w-4" />
            </button>
          )}
        </div>

        {/* Navigation */}
        <nav className="flex-1 overflow-y-auto px-2 py-3">
          <div className="space-y-0.5">
            {NAV_ITEMS.map((item) => {
              const isActive =
                item.to === "/"
                  ? location.pathname === "/"
                  : location.pathname.startsWith(item.to);

              const link = (
                <NavLink
                  key={item.to}
                  to={item.to}
                  className={cn(
                    "group relative flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                    isActive
                      ? "text-ink"
                      : "text-ink-secondary hover:bg-surface-muted/60 hover:text-ink",
                    collapsed && "justify-center px-2",
                  )}
                >
                  {/* Active indicator bar */}
                  {isActive && (
                    <motion.div
                      layoutId="sidebar-active-bar"
                      className="absolute left-0 top-1/2 h-5 w-0.5 -translate-y-1/2 rounded-full bg-primary"
                      transition={{ type: "spring", stiffness: 380, damping: 30 }}
                    />
                  )}
                  <item.icon
                    className={cn(
                      "h-[18px] w-[18px] flex-shrink-0 transition-colors",
                      isActive ? "text-primary" : "text-ink-muted group-hover:text-ink",
                    )}
                  />
                  {!collapsed && <span className="truncate">{item.label}</span>}
                </NavLink>
              );

              if (collapsed) {
                return (
                  <Tooltip key={item.to}>
                    <TooltipTrigger asChild>{link}</TooltipTrigger>
                    <TooltipContent side="right" sideOffset={8}>
                      <div>
                        <div className="font-medium">{item.label}</div>
                        <div className="text-[10px] opacity-70">{item.description}</div>
                      </div>
                    </TooltipContent>
                  </Tooltip>
                );
              }
              return link;
            })}
          </div>
        </nav>

        {/* Footer: collapse toggle + WS status */}
        <div className="border-t border-border p-2">
          {collapsed && (
            <button
              onClick={onToggle}
              className="flex w-full items-center justify-center rounded-md py-2 text-ink-muted transition-colors hover:bg-surface-muted/60 hover:text-ink"
            >
              <PanelLeft className="h-4 w-4" />
            </button>
          )}
          <WsStatusIndicator collapsed={collapsed} />
        </div>
      </aside>
    </TooltipProvider>
  );
}

function WsStatusIndicator({ collapsed }: { collapsed: boolean }) {
  const [connected, setConnected] = React.useState(false);

  React.useEffect(() => {
    wsClient.connect();
    setConnected(wsClient.isConnected);
    const unsub = wsClient.onStatusChange(setConnected);
    return () => {
      unsub();
    };
  }, []);

  return (
    <div
      className={cn(
        "flex items-center gap-2 rounded-md px-3 py-2 text-xs font-medium",
        collapsed ? "justify-center px-2" : "",
      )}
    >
      <div
        className={cn(
          "h-2 w-2 flex-shrink-0 rounded-full transition-colors",
          connected ? "bg-success" : "bg-ink-faint animate-pulse",
        )}
      />
      {!collapsed && (
        <span className="text-ink-muted">
          {connected ? "Connected" : "Connecting…"}
        </span>
      )}
    </div>
  );
}
