import * as React from "react";
import { motion } from "motion/react";
import { useSearchParams } from "react-router-dom";
import { Route as RouteIcon, MapPin, Clock, ArrowRight, Info, GitBranch, MapPinned, Shuffle } from "lucide-react";
import { PageHeader, PageContainer } from "@/components/common/page-header";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { queriesApi } from "@/services/api";
import { useApi } from "@/hooks/use-api";
import type { QueryStatusOut, QueryRouteOut, RouteStepOut, SightingOut } from "@/types/api";
import { cn, cropUrl, formatTime } from "@/lib/utils";

const CAM_LAYOUT: Record<string, { x: number; y: number; loc: string }> = {
  C01: { x: 150, y: 170, loc: "Entrance Zone" },
  C02: { x: 400, y: 170, loc: "Corridor Hub" },
  C03: { x: 650, y: 170, loc: "Canteen Area" },
};

export default function RouteReconstruction() {
  const [searchParams, setSearchParams] = useSearchParams();
  const urlSessionId = searchParams.get("session_id") || searchParams.get("session") || "";

  const { data: sessions } = useApi<QueryStatusOut[]>(() => queriesApi.list(30));
  const doneSessions = sessions?.filter((s) => s.status === "done") ?? [];

  const [selectedSessionId, setSelectedSessionId] = React.useState<string>(urlSessionId);

  // Auto-select session 11 (or latest completed session) if none selected
  React.useEffect(() => {
    if (!selectedSessionId && doneSessions.length > 0) {
      const has11 = doneSessions.some((s) => s.id === 11);
      const targetId = has11 ? "11" : String(doneSessions[0].id);
      setSelectedSessionId(targetId);
      setSearchParams({ session_id: targetId }, { replace: true });
    }
  }, [doneSessions, selectedSessionId, setSearchParams]);

  const handleSessionChange = (id: string) => {
    setSelectedSessionId(id);
    setSearchParams(id ? { session_id: id } : {});
  };

  const { data: route, loading } = useApi<QueryRouteOut | null>(
    () => selectedSessionId ? queriesApi.route(Number(selectedSessionId)) : Promise.resolve(null),
    [selectedSessionId],
  );

  return (
    <PageContainer>
      <PageHeader title="Route Reconstruction" description="Spatial-temporal trajectory analysis across camera network">
        <Select value={selectedSessionId} onValueChange={handleSessionChange}>
          <SelectTrigger className="w-[260px]">
            <SelectValue placeholder="Select a session…" />
          </SelectTrigger>
          <SelectContent>
            {doneSessions.length === 0 ? (
              <SelectItem value="" disabled>No completed sessions</SelectItem>
            ) : (
              doneSessions.map((s) => (
                <SelectItem key={s.id} value={String(s.id)}>
                  #{s.id} — {formatTime(s.submitted_at)}
                </SelectItem>
              ))
            )}
          </SelectContent>
        </Select>
      </PageHeader>

      {!route && !loading ? (
        <Card>
          <CardContent className="py-16">
            <div className="flex flex-col items-center gap-3 text-center">
              <div className="flex h-12 w-12 items-center justify-center rounded-full bg-surface-muted">
                <RouteIcon className="h-6 w-6 text-ink-faint" />
              </div>
              <div>
                <p className="text-sm font-medium text-ink-secondary">No session selected</p>
                <p className="mt-1 text-xs text-ink-muted">
                  {doneSessions.length > 0
                    ? "Select a completed session above to view the reconstructed route."
                    : "Run a Re-ID search first to generate route data."}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      ) : loading ? (
        <div className="grid gap-4 lg:grid-cols-[1fr_360px]">
          <Skeleton className="h-[400px] w-full rounded-lg" />
          <Skeleton className="h-[400px] w-full rounded-lg" />
        </div>
      ) : route ? (
        <div className="space-y-4">
          <div className="grid gap-4 lg:grid-cols-[1fr_360px]">
            <RouteMap route={route} />
            <RouteSidebar route={route} />
          </div>
          <WhyThisRoute route={route} />
        </div>
      ) : null}
    </PageContainer>
  );
}

// ── Route Map (SVG) ────────────────────────────────────────────────────────────

function RouteMap({ route }: { route: QueryRouteOut }) {
  const [selectedCam, setSelectedCam] = React.useState<string | null>(null);
  const matchedCams = new Set(route.sightings.map((s) => s.camera_id));

  return (
    <Card className="overflow-hidden">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle>Camera Network Map</CardTitle>
          <div className="flex items-center gap-3 text-xs text-ink-muted">
            <span className="flex items-center gap-1">
              <span className="h-2 w-2 rounded-full bg-primary" /> Camera
            </span>
            <span className="flex items-center gap-1">
              <span className="h-2 w-2 rounded-full bg-success" /> Matched
            </span>
            <span className="flex items-center gap-1">
              <span className="h-0.5 w-4 bg-primary" /> Route
            </span>
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-0">
        <svg viewBox="0 0 800 360" className="w-full" style={{ maxHeight: 400 }}>
          {/* Background zones */}
          <g>
            <rect x="45" y="95" width="220" height="170" rx="8" fill="rgba(23,43,80,0.02)" stroke="#E5DED2" strokeWidth="1" strokeDasharray="4 3" />
            <rect x="290" y="95" width="220" height="170" rx="8" fill="rgba(23,43,80,0.02)" stroke="#E5DED2" strokeWidth="1" strokeDasharray="4 3" />
            <rect x="535" y="95" width="220" height="170" rx="8" fill="rgba(23,43,80,0.02)" stroke="#E5DED2" strokeWidth="1" strokeDasharray="4 3" />
            <text x="155" y="285" textAnchor="middle" fill="#A0A6B0" fontSize="9" fontWeight="600">ENTRANCE</text>
            <text x="400" y="285" textAnchor="middle" fill="#A0A6B0" fontSize="9" fontWeight="600">CORRIDOR</text>
            <text x="645" y="285" textAnchor="middle" fill="#A0A6B0" fontSize="9" fontWeight="600">CANTEEN</text>
          </g>

          {/* Route lines */}
          {route.steps.slice(1).map((_, i) => {
            const prev = route.steps[i];
            const curr = route.steps[i + 1];
            const pa = CAM_LAYOUT[prev.camera_id];
            const pb = CAM_LAYOUT[curr.camera_id];
            if (!pa || !pb) return null;
            return (
              <g key={i}>
                <motion.line
                  x1={pa.x + 60} y1={pa.y}
                  x2={pb.x - 60} y2={pb.y}
                  stroke="#8C263B" strokeWidth="2.5" strokeDasharray="8 5"
                  initial={{ pathLength: 0 }}
                  animate={{ pathLength: 1 }}
                  transition={{ delay: i * 0.15, duration: 0.3 }}
                />
                <circle cx={(pa.x + pb.x) / 2} cy={pa.y - 30} r="12" fill="#8C263B" />
                <text x={(pa.x + pb.x) / 2} y={(pa.y - 30) + 4} textAnchor="middle" fill="white" fontSize="10" fontWeight="700">
                  {i + 1}
                </text>
              </g>
            );
          })}

          {/* Camera nodes */}
          {Object.entries(CAM_LAYOUT).map(([id, pos]) => {
            const isMatched = matchedCams.has(id);
            const isSelected = selectedCam === id;
            return (
              <g
                key={id}
                className="cursor-pointer"
                onClick={() => setSelectedCam(id)}
              >
                {isMatched && (
                  <motion.circle
                    cx={pos.x} cy={pos.y} r="40"
                    fill="none" stroke="#2F7668" strokeWidth="1.5"
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: [1, 1.3, 1], opacity: [0.6, 0, 0.6] }}
                    transition={{ repeat: Infinity, duration: 2.5 }}
                  />
                )}
                <rect
                  x={pos.x - 58} y={pos.y - 38}
                  width="116" height="76" rx="10"
                  fill={isSelected ? "rgba(140,38,59,0.08)" : "#FFFFFF"}
                  stroke={isMatched ? "#2F7668" : isSelected ? "#8C263B" : "#E5DED2"}
                  strokeWidth={isMatched ? 2 : 1.5}
                />
                <text x={pos.x} y={pos.y - 8} textAnchor="middle" fill="#172B50" fontSize="13" fontWeight="700">{id}</text>
                <text x={pos.x} y={pos.y + 8} textAnchor="middle" fill="#7C8492" fontSize="9" fontWeight="500">{pos.loc}</text>
                <circle
                  cx={pos.x + 46} cy={pos.y - 28} r="4"
                  fill={isMatched ? "#2F7668" : "#A0A6B0"}
                />
              </g>
            );
          })}
        </svg>
      </CardContent>
    </Card>
  );
}

// ── Route Sidebar ───────────────────────────────────────────────────────────────

function RouteSidebar({ route }: { route: QueryRouteOut }) {
  const pct = (route.route_confidence * 100).toFixed(1);
  const colorClass = parseFloat(pct) >= 70 ? "text-success" : parseFloat(pct) >= 50 ? "text-warning" : "text-error";

  return (
    <div className="space-y-4">
      {/* Confidence */}
      <Card>
        <CardHeader><CardTitle>Route Confidence</CardTitle></CardHeader>
        <CardContent>
          <div className={cn("text-3xl font-bold", colorClass)}>{pct}%</div>
          <p className="mt-1 text-xs text-ink-muted">
            {route.total_cameras_matched} camera(s) matched
          </p>
          <div className="mt-3 h-2 overflow-hidden rounded-full bg-surface-muted">
            <div
              className={cn("h-full rounded-full", parseFloat(pct) >= 70 ? "bg-success" : parseFloat(pct) >= 50 ? "bg-warning" : "bg-error")}
              style={{ width: `${pct}%` }}
            />
          </div>
          <Badge
            variant={route.match_decision === "CONFIDENT_MATCH" ? "success" : "warning"}
            className="mt-3"
          >
            {route.match_decision.replace(/_/g, " ")}
          </Badge>
        </CardContent>
      </Card>

      {/* Steps */}
      <Card>
        <CardHeader><CardTitle>Sequence Steps</CardTitle></CardHeader>
        <CardContent className="space-y-2">
          {route.steps.map((step, i) => (
            <div key={i} className="flex items-center gap-3 rounded-md border border-border p-2.5">
              <div className="flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-full bg-primary text-xs font-bold text-white">
                {step.step_order + 1}
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-sm font-medium text-ink">{step.camera_id}</div>
                <div className="text-xs text-ink-muted">{step.camera_location}</div>
              </div>
              <div className="text-right">
                <div className="text-sm font-mono font-medium text-ink">{step.confidence.toFixed(1)}</div>
                <div className="text-[10px] text-ink-muted">{step.timestamp ? formatTime(step.timestamp) : "—"}</div>
              </div>
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  );
}

// ── Why This Route (Explainable AI) ─────────────────────────────────────────────

function WhyThisRoute({ route }: { route: QueryRouteOut }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Info className="h-4 w-4 text-primary" />
          Why This Route?
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <EvidenceCard
            icon={<MapPin className="h-4 w-4" />}
            title="Appearance Compatibility"
            value={route.sightings.length > 0 ? route.sightings[0].appearance_score : 0}
            available={route.sightings.length > 0}
          />
          <EvidenceCard
            icon={<GitBranch className="h-4 w-4" />}
            title="Camera Topology"
            value={route.steps.length > 1 ? 0.85 : 0}
            available={route.steps.length > 1}
            note={route.steps.length > 1 ? `${route.steps.length} connected cameras` : "Single camera"}
          />
          <EvidenceCard
            icon={<Clock className="h-4 w-4" />}
            title="Temporal Feasibility"
            value={route.sightings.length > 0 ? route.sightings[0].temporal_score : 0}
            available={route.sightings.length > 0}
          />
          <EvidenceCard
            icon={<MapPinned className="h-4 w-4" />}
            title="Geographic Consistency"
            value={route.steps.length / 3}
            available={route.steps.length > 0}
          />
        </div>

        {/* Status breakdown */}
        <div className="space-y-2">
          <h4 className="text-sm font-medium text-ink">Transition Status</h4>
          {route.steps.length > 0 ? (
            route.sightings.map((s) => (
              <div key={s.id} className="flex items-center gap-3 rounded-md border border-border p-2.5">
                <span className="text-sm font-medium text-ink">{s.camera_id}</span>
                <Badge
                  variant={
                    s.match_status === "CONFIDENT_MATCH" ? "success" :
                    s.match_status === "FACE_MISMATCH" ? "destructive" : "warning"
                  }
                  className="text-xs"
                >
                  {s.match_status.replace(/_/g, " ")}
                </Badge>
                {s.face_veto_applied && (
                  <span className="text-xs text-error">Face veto: {s.face_veto_reason}</span>
                )}
              </div>
            ))
          ) : (
            <p className="text-xs text-ink-muted">No sighting data available.</p>
          )}
        </div>

        {/* Alternatives */}
        {route.steps.length > 1 && (
          <div className="flex items-start gap-2 rounded-md bg-surface-muted p-3">
            <Shuffle className="mt-0.5 h-3.5 w-3.5 flex-shrink-0 text-ink-muted" />
            <p className="text-xs text-ink-muted">
              Alternative routes may exist if temporal gaps between cameras are ambiguous.
              Current route reflects the highest-confidence sequence based on available evidence.
            </p>
          </div>
        )}

        {/* Disclaimer */}
        <div className="flex items-start gap-2 rounded-md bg-primary/5 p-3">
          <Info className="mt-0.5 h-3.5 w-3.5 flex-shrink-0 text-primary" />
          <p className="text-xs text-ink-secondary">
            Route confidence is the mean fusion score across all route steps. It is not an identity probability.
            Visual verification is required before any conclusion.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}

function EvidenceCard({ icon, title, value, available, note }: { icon: React.ReactNode; title: string; value: number; available: boolean; note?: string }) {
  const pct = Math.min(value * 100, 100);
  return (
    <div className="rounded-lg border border-border p-3">
      <div className="flex items-center gap-2">
        <span className="text-ink-muted">{icon}</span>
        <span className="text-xs font-medium text-ink-secondary">{title}</span>
      </div>
      {available ? (
        <>
          <div className="mt-2 text-lg font-bold text-ink">{(value * 100).toFixed(1)}%</div>
          <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-surface-muted">
            <div className="h-full rounded-full bg-primary" style={{ width: `${pct}%` }} />
          </div>
          {note && <p className="mt-1 text-[10px] text-ink-faint">{note}</p>}
        </>
      ) : (
        <p className="mt-2 text-xs text-ink-muted">No evidence available</p>
      )}
    </div>
  );
}
