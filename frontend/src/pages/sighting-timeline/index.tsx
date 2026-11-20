import * as React from "react";
import { motion } from "motion/react";
import { useSearchParams } from "react-router-dom";
import {
  Clock,
  MapPin,
  User,
  ChevronRight,
  ShieldCheck,
  AlertTriangle,
  Layers,
  Activity,
  CheckCircle2,
  Calendar,
} from "lucide-react";
import { PageHeader, PageContainer } from "@/components/common/page-header";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { queriesApi, camerasApi } from "@/services/api";
import { useApi } from "@/hooks/use-api";
import type { QueryStatusOut, QueryRouteOut, SightingOut, CameraWithGraph } from "@/types/api";
import { cn, cropUrl, formatTime, parseTimeToSeconds, formatSecondsToTime } from "@/lib/utils";

export default function SightingTimeline() {
  const [searchParams, setSearchParams] = useSearchParams();
  const urlSessionId = searchParams.get("session_id") || searchParams.get("session") || "";

  const { data: sessions } = useApi<QueryStatusOut[]>(() => queriesApi.list(30));
  const { data: allCameras } = useApi<CameraWithGraph[]>(() => camerasApi.list());

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

  // Keep URL in sync with selected session
  const handleSessionChange = (id: string) => {
    setSelectedSessionId(id);
    setSearchParams(id ? { session_id: id } : {});
  };

  const { data: route, loading } = useApi<QueryRouteOut | null>(
    () => (selectedSessionId ? queriesApi.route(Number(selectedSessionId)) : Promise.resolve(null)),
    [selectedSessionId],
  );

  const [selectedSightingId, setSelectedSightingId] = React.useState<number | null>(null);

  // Auto-select first sighting when route data changes
  React.useEffect(() => {
    if (route?.sightings && route.sightings.length > 0) {
      setSelectedSightingId(route.sightings[0].id);
    } else {
      setSelectedSightingId(null);
    }
  }, [route?.session_id]);

  const selectedSighting =
    route?.sightings.find((s) => s.id === selectedSightingId) ?? route?.sightings[0] ?? null;

  return (
    <PageContainer>
      <PageHeader
        title="Sighting Timeline"
        description="Multi-camera temporal sequence and chronological trajectory of detection events"
      >
        <div className="flex items-center gap-2">
          <Select value={selectedSessionId} onValueChange={handleSessionChange}>
            <SelectTrigger className="w-[280px]">
              <SelectValue placeholder="Select a session…" />
            </SelectTrigger>
            <SelectContent>
              {doneSessions.length === 0 ? (
                <SelectItem value="none" disabled>
                  No completed sessions
                </SelectItem>
              ) : (
                doneSessions.map((s) => (
                  <SelectItem key={s.id} value={String(s.id)}>
                    #{s.id} — {s.submitted_at ? formatTime(s.submitted_at) : `Session ${s.id}`}
                  </SelectItem>
                ))
              )}
            </SelectContent>
          </Select>
        </div>
      </PageHeader>

      {!route && !loading ? (
        <Card>
          <CardContent className="py-16">
            <div className="flex flex-col items-center gap-3 text-center">
              <div className="flex h-12 w-12 items-center justify-center rounded-full bg-surface-muted">
                <Clock className="h-6 w-6 text-ink-faint" />
              </div>
              <div>
                <p className="text-sm font-medium text-ink-secondary">No session selected</p>
                <p className="mt-1 text-xs text-ink-muted">
                  {doneSessions.length > 0
                    ? "Select a completed session above to inspect its multi-camera sighting timeline."
                    : "Run a Re-ID search first to generate sighting and timeline data."}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      ) : loading ? (
        <div className="space-y-4">
          <Skeleton className="h-20 w-full rounded-lg" />
          <Skeleton className="h-[320px] w-full rounded-lg" />
          <Skeleton className="h-[220px] w-full rounded-lg" />
        </div>
      ) : route ? (
        <div className="space-y-5">
          {/* Session Overview Stats Banner */}
          <SessionOverview route={route} />

          {/* Temporal Swimlane Chart */}
          <TimelineSwimlane
            route={route}
            cameras={allCameras || []}
            selectedId={selectedSighting?.id ?? null}
            onSelect={(s) => setSelectedSightingId(s.id)}
          />

          {/* Sighting Detail & Chronological Stream */}
          <div className="grid gap-5 lg:grid-cols-[1fr_380px]">
            {/* Sequential List of Events */}
            <SightingsChronology
              sightings={route.sightings}
              cameras={allCameras || []}
              selectedId={selectedSighting?.id ?? null}
              onSelect={(s) => setSelectedSightingId(s.id)}
            />

            {/* Detailed Inspector for Selected Sighting */}
            {selectedSighting ? (
              <SightingInspector sighting={selectedSighting} cameras={allCameras || []} />
            ) : (
              <Card>
                <CardContent className="flex h-[320px] flex-col items-center justify-center p-6 text-center text-ink-muted">
                  <User className="mb-2 h-8 w-8 text-ink-faint" />
                  <p className="text-xs">Select a detection event on the timeline to inspect details.</p>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      ) : null}
    </PageContainer>
  );
}

// ── Session Overview Banner ──────────────────────────────────────────────────

function SessionOverview({ route }: { route: QueryRouteOut }) {
  const sightings = route.sightings;
  const parsed = sightings
    .map((s) => parseTimeToSeconds(s.first_seen))
    .filter((t): t is number => t !== null && !isNaN(t));

  const minSec = parsed.length > 0 ? Math.min(...parsed) : null;
  const maxSec = parsed.length > 0 ? Math.max(...parsed) : null;
  const durationSec = minSec !== null && maxSec !== null ? Math.round(maxSec - minSec) : null;

  const durationStr =
    durationSec !== null
      ? durationSec >= 60
        ? `${Math.floor(durationSec / 60)}m ${durationSec % 60}s`
        : `${durationSec}s`
      : "—";

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      <Card className="bg-surface">
        <CardContent className="p-3.5 flex items-center justify-between">
          <div>
            <p className="text-[11px] font-medium text-ink-muted">Match Decision</p>
            <p className="mt-0.5 text-sm font-semibold text-ink">
              {route.match_decision?.replace(/_/g, " ") || "COMPLETED"}
            </p>
          </div>
          <Badge
            variant={
              route.match_decision === "CONFIDENT_MATCH"
                ? "success"
                : route.match_decision === "POSSIBLE_MATCH_REVIEW"
                ? "warning"
                : "secondary"
            }
            className="text-[10px]"
          >
            {route.match_decision || "DONE"}
          </Badge>
        </CardContent>
      </Card>

      <Card className="bg-surface">
        <CardContent className="p-3.5 flex items-center justify-between">
          <div>
            <p className="text-[11px] font-medium text-ink-muted">Sightings Recorded</p>
            <p className="mt-0.5 text-sm font-semibold text-ink">{sightings.length} detections</p>
          </div>
          <div className="flex h-8 w-8 items-center justify-center rounded-md bg-primary/10 text-primary">
            <Activity className="h-4 w-4" />
          </div>
        </CardContent>
      </Card>

      <Card className="bg-surface">
        <CardContent className="p-3.5 flex items-center justify-between">
          <div>
            <p className="text-[11px] font-medium text-ink-muted">Cameras Matched</p>
            <p className="mt-0.5 text-sm font-semibold text-ink">
              {route.total_cameras_matched} active camera{route.total_cameras_matched === 1 ? "" : "s"}
            </p>
          </div>
          <div className="flex h-8 w-8 items-center justify-center rounded-md bg-surface-muted text-ink-secondary">
            <Layers className="h-4 w-4" />
          </div>
        </CardContent>
      </Card>

      <Card className="bg-surface">
        <CardContent className="p-3.5 flex items-center justify-between">
          <div>
            <p className="text-[11px] font-medium text-ink-muted">Elapsed Duration</p>
            <p className="mt-0.5 text-sm font-semibold text-ink">{durationStr}</p>
          </div>
          <div className="flex h-8 w-8 items-center justify-center rounded-md bg-surface-muted text-ink-secondary">
            <Clock className="h-4 w-4" />
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// ── Multi-Camera Swimlane Timeline ──────────────────────────────────────────

function TimelineSwimlane({
  route,
  cameras,
  selectedId,
  onSelect,
}: {
  route: QueryRouteOut;
  cameras: CameraWithGraph[];
  selectedId: number | null;
  onSelect: (s: SightingOut) => void;
}) {
  const sightings = route.sightings;

  // Build a camera map to lookup locations
  const camMap = React.useMemo(() => {
    const map = new Map<string, string>();
    cameras.forEach((c) => map.set(c.id, c.location));
    return map;
  }, [cameras]);

  // Determine which camera lanes to show: all registered cameras, or all present in sightings
  const cameraIds = React.useMemo(() => {
    const presentInSightings = Array.from(new Set(sightings.map((s) => s.camera_id)));
    if (cameras.length > 0) {
      const allKnown = cameras.map((c) => c.id);
      // Ensure all known cameras are present, followed by any unexpected ones
      return Array.from(new Set([...allKnown, ...presentInSightings]));
    }
    return presentInSightings.length > 0 ? presentInSightings : ["C01", "C02", "C03"];
  }, [cameras, sightings]);

  // Parse time stamps
  const parsedSightings = React.useMemo(() => {
    return sightings.map((s) => {
      const sec = parseTimeToSeconds(s.first_seen);
      return {
        ...s,
        sec,
      };
    });
  }, [sightings]);

  const validTimes = parsedSightings
    .map((s) => s.sec)
    .filter((t): t is number => t !== null && !isNaN(t));

  const hasTimestamps = validTimes.length > 0;
  const minTime = hasTimestamps ? Math.min(...validTimes) : 0;
  const maxTime = hasTimestamps ? Math.max(...validTimes) : 0;

  // Add a small safety padding to left and right so dots don't clip at edges
  const span = Math.max(maxTime - minTime, 10);
  const pad = Math.max(span * 0.08, 4);
  const startSec = Math.max(0, minTime - pad);
  const endSec = maxTime + pad;
  const totalRange = Math.max(endSec - startSec, 1);

  // Time ruler tick labels (0%, 25%, 50%, 75%, 100%)
  const ticks = [0, 0.25, 0.5, 0.75, 1.0].map((ratio) => {
    const sec = startSec + totalRange * ratio;
    return {
      ratio,
      label: formatSecondsToTime(sec),
    };
  });

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <CardTitle className="text-base flex items-center gap-2">
            <Clock className="h-4 w-4 text-primary" />
            Camera Swimlane Timeline
          </CardTitle>
          <div className="flex items-center gap-3 text-xs text-ink-muted">
            <span className="inline-flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded-full bg-emerald-500" />
              Confident Match
            </span>
            <span className="inline-flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded-full bg-amber-500" />
              Possible Match
            </span>
            <span className="inline-flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded-full bg-rose-500" />
              Face Mismatch
            </span>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-3">
        {/* Time Axis Ruler */}
        <div className="relative h-6 pl-[150px] pr-4">
          <div className="absolute inset-x-[150px] top-1/2 h-px bg-border -translate-y-1/2" />
          {ticks.map((t, idx) => (
            <div
              key={idx}
              className="absolute top-0 -translate-x-1/2 flex flex-col items-center"
              style={{ left: `calc(150px + (100% - 170px) * ${t.ratio})` }}
            >
              <div className="h-2 w-px bg-border-strong mb-0.5" />
              <span className="font-mono text-[10px] text-ink-muted select-none">{t.label}</span>
            </div>
          ))}
        </div>

        {/* Lanes for each Camera */}
        <div className="space-y-2.5 pt-1">
          {cameraIds.map((camId) => {
            const camSightings = parsedSightings.filter((s) => s.camera_id === camId);
            const locName = camMap.get(camId) || "Camera Zone";

            return (
              <div key={camId} className="flex items-center gap-3">
                {/* Lane Header */}
                <div className="w-[140px] shrink-0">
                  <div className="flex items-center gap-1.5">
                    <span className="text-xs font-semibold text-ink">{camId}</span>
                    {camSightings.length > 0 && (
                      <span className="rounded-full bg-primary/10 px-1.5 py-0.2 text-[10px] font-medium text-primary">
                        {camSightings.length}
                      </span>
                    )}
                  </div>
                  <div className="truncate text-[11px] text-ink-muted">{locName}</div>
                </div>

                {/* Lane Track */}
                <div className="relative flex-1 h-14 rounded-lg border border-border bg-surface-muted/30 overflow-hidden px-4">
                  {/* Gridline guidelines at 25%, 50%, 75% */}
                  <div className="absolute left-1/4 top-0 bottom-0 w-px bg-border/40" />
                  <div className="absolute left-2/4 top-0 bottom-0 w-px bg-border/40" />
                  <div className="absolute left-3/4 top-0 bottom-0 w-px bg-border/40" />

                  {/* Horizontal centerline */}
                  <div className="absolute top-1/2 left-0 right-0 h-px bg-border/60 -translate-y-1/2" />

                  {camSightings.length === 0 ? (
                    <div className="absolute inset-0 flex items-center justify-center">
                      <span className="text-[11px] text-ink-faint italic">No sightings detected</span>
                    </div>
                  ) : (
                    camSightings.map((s, idx) => {
                      const isSelected = selectedId === s.id;
                      let leftPct = 50;

                      if (hasTimestamps && s.sec !== null) {
                        leftPct = Math.max(3, Math.min(97, ((s.sec - startSec) / totalRange) * 100));
                      } else {
                        leftPct = ((idx + 1) / (camSightings.length + 1)) * 100;
                      }

                      const statusColor =
                        s.match_status === "CONFIDENT_MATCH"
                          ? "bg-emerald-500 border-emerald-600 text-white shadow-emerald-500/30"
                          : s.match_status === "FACE_MISMATCH"
                          ? "bg-rose-500 border-rose-600 text-white shadow-rose-500/30"
                          : "bg-amber-500 border-amber-600 text-white shadow-amber-500/30";

                      return (
                        <motion.button
                          key={s.id}
                          type="button"
                          initial={{ scale: 0, opacity: 0 }}
                          animate={{ scale: 1, opacity: 1 }}
                          transition={{ type: "spring", stiffness: 350, damping: 25 }}
                          onClick={() => onSelect(s)}
                          style={{ left: `${leftPct}%` }}
                          title={`${s.camera_id} · Track ${s.track_id} · ${formatTime(s.first_seen)} · ${s.best_confidence.toFixed(1)}%`}
                          className={cn(
                            "absolute top-1/2 -translate-x-1/2 -translate-y-1/2 cursor-pointer rounded-full border-2 shadow-sm transition-transform hover:scale-125 focus:outline-none",
                            statusColor,
                            isSelected
                              ? "h-8 w-8 ring-4 ring-primary/40 ring-offset-2 ring-offset-surface z-20 scale-110"
                              : "h-6 w-6 z-10",
                          )}
                        >
                          <span className="text-[10px] font-bold">
                            {s.track_id}
                          </span>
                        </motion.button>
                      );
                    })
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}

// ── Chronological Stream of Sightings ───────────────────────────────────────

function SightingsChronology({
  sightings,
  cameras,
  selectedId,
  onSelect,
}: {
  sightings: SightingOut[];
  cameras: CameraWithGraph[];
  selectedId: number | null;
  onSelect: (s: SightingOut) => void;
}) {
  const camMap = React.useMemo(() => {
    const map = new Map<string, string>();
    cameras.forEach((c) => map.set(c.id, c.location));
    return map;
  }, [cameras]);

  // Sort chronologically by first_seen
  const sorted = React.useMemo(() => {
    return [...sightings].sort((a, b) => {
      const ta = parseTimeToSeconds(a.first_seen) ?? 0;
      const tb = parseTimeToSeconds(b.first_seen) ?? 0;
      return ta - tb;
    });
  }, [sightings]);

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base flex items-center gap-2">
            <Activity className="h-4 w-4 text-primary" />
            Detection Event Stream ({sorted.length})
          </CardTitle>
          <span className="text-xs text-ink-muted">Sorted chronologically</span>
        </div>
      </CardHeader>
      <CardContent>
        {sorted.length === 0 ? (
          <div className="py-8 text-center text-sm text-ink-muted">
            No detection sightings recorded for this session.
          </div>
        ) : (
          <div className="space-y-2">
            {sorted.map((s, idx) => {
              const isSelected = selectedId === s.id;
              const url = cropUrl(s.crop_path);
              const loc = camMap.get(s.camera_id) || "Camera Zone";

              return (
                <div
                  key={s.id}
                  onClick={() => onSelect(s)}
                  className={cn(
                    "flex cursor-pointer items-center gap-3 rounded-lg border p-2.5 transition-all",
                    isSelected
                      ? "border-primary bg-primary/5 shadow-sm ring-1 ring-primary/20"
                      : "border-border bg-surface hover:bg-surface-muted/40",
                  )}
                >
                  <span className="w-5 text-center font-mono text-xs text-ink-faint">
                    #{idx + 1}
                  </span>

                  {/* Thumbnail crop */}
                  <div className="h-10 w-10 shrink-0 overflow-hidden rounded-md bg-surface-muted border border-border flex items-center justify-center">
                    {url ? (
                      <img src={url} alt="Crop" className="h-full w-full object-cover" />
                    ) : (
                      <User className="h-5 w-5 text-ink-faint" />
                    )}
                  </div>

                  {/* Camera & Time info */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-semibold text-ink">{s.camera_id}</span>
                      <span className="truncate text-xs text-ink-muted">— {loc}</span>
                      <span className="rounded bg-surface-muted px-1.5 py-0.2 text-[10px] font-mono text-ink-secondary">
                        Track {s.track_id}
                      </span>
                    </div>
                    <div className="mt-0.5 flex items-center gap-2 text-[11px] font-mono text-ink-muted">
                      <Clock className="h-3 w-3" />
                      <span>{formatTime(s.first_seen)}</span>
                      {s.last_seen && s.last_seen !== s.first_seen && (
                        <span>→ {formatTime(s.last_seen)}</span>
                      )}
                    </div>
                  </div>

                  {/* Confidence and status badge */}
                  <div className="flex flex-col items-end gap-1 shrink-0">
                    <Badge
                      variant={
                        s.match_status === "CONFIDENT_MATCH"
                          ? "success"
                          : s.match_status === "FACE_MISMATCH"
                          ? "destructive"
                          : "warning"
                      }
                      className="text-[10px]"
                    >
                      {s.match_status.replace(/_/g, " ")}
                    </Badge>
                    <span className="font-mono text-xs font-semibold text-ink">
                      {s.best_confidence.toFixed(1)}%
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// ── Sighting Detailed Inspector ──────────────────────────────────────────────

function SightingInspector({
  sighting,
  cameras,
}: {
  sighting: SightingOut;
  cameras: CameraWithGraph[];
}) {
  const url = cropUrl(sighting.crop_path);
  const cam = cameras.find((c) => c.id === sighting.camera_id);

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.15 }}
    >
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center justify-between">
            <span className="flex items-center gap-1.5">
              <MapPin className="h-4 w-4 text-primary" />
              {sighting.camera_id} · Track {sighting.track_id}
            </span>
            <Badge
              variant={
                sighting.match_status === "CONFIDENT_MATCH"
                  ? "success"
                  : sighting.match_status === "FACE_MISMATCH"
                  ? "destructive"
                  : "warning"
              }
              className="text-[10px]"
            >
              {sighting.match_status.replace(/_/g, " ")}
            </Badge>
          </CardTitle>
          <p className="text-xs text-ink-muted">{cam?.location || "Camera Location"}</p>
        </CardHeader>

        <CardContent className="space-y-4">
          {/* Crop image display */}
          <div className="flex justify-center rounded-lg border border-border bg-surface-muted/40 p-3">
            {url ? (
              <img
                src={url}
                alt="Detection Crop"
                className="max-h-48 rounded object-contain shadow-sm"
              />
            ) : (
              <div className="flex h-32 w-32 items-center justify-center rounded-md bg-surface-muted text-ink-faint">
                <User className="h-12 w-12" />
              </div>
            )}
          </div>

          {/* Telemetry Details */}
          <div className="space-y-2 rounded-lg border border-border bg-surface p-3 text-xs">
            <InspectorRow label="First Seen" value={formatTime(sighting.first_seen)} />
            <InspectorRow label="Last Seen" value={formatTime(sighting.last_seen)} />
            <InspectorRow label="Match Confidence" value={`${sighting.best_confidence.toFixed(1)}%`} />
            <InspectorRow label="Fusion Score" value={sighting.fusion_score.toFixed(4)} />
            <InspectorRow label="Appearance Similarity" value={sighting.appearance_score.toFixed(4)} />
            <InspectorRow label="Spatial Compatibility" value={sighting.spatial_score.toFixed(4)} />
            <InspectorRow label="Temporal Compatibility" value={sighting.temporal_score.toFixed(4)} />
            <InspectorRow label="Active Re-ID Backbone" value={sighting.active_backbone || "OSNet"} />
          </div>

          {/* Face veto alert if applied */}
          {sighting.face_veto_applied && (
            <div className="flex items-start gap-2 rounded-md border border-rose-500/20 bg-rose-500/10 p-2.5 text-xs text-rose-700 dark:text-rose-400">
              <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
              <div>
                <span className="font-semibold">Face Veto Triggered: </span>
                {sighting.face_veto_reason || "Face identity similarity fell below threshold."}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
}

function InspectorRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between py-0.5">
      <span className="text-ink-muted">{label}</span>
      <span className="font-mono font-medium text-ink">{value}</span>
    </div>
  );
}
