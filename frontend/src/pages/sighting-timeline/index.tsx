import * as React from "react";
import { motion } from "motion/react";
import { Clock, MapPin, User, ChevronRight } from "lucide-react";
import { PageHeader, PageContainer } from "@/components/common/page-header";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { queriesApi } from "@/services/api";
import { useApi } from "@/hooks/use-api";
import type { QueryStatusOut, QueryRouteOut, SightingOut } from "@/types/api";
import { cn, cropUrl, formatTime } from "@/lib/utils";

const CAMERA_TRACKS = [
  { id: "C01", loc: "Entrance Zone" },
  { id: "C02", loc: "Corridor Hub" },
  { id: "C03", loc: "Canteen Area" },
];

export default function SightingTimeline() {
  const { data: sessions } = useApi<QueryStatusOut[]>(() => queriesApi.list(30));
  const doneSessions = sessions?.filter((s) => s.status === "done") ?? [];

  const [selectedSessionId, setSelectedSessionId] = React.useState<string>("");
  const { data: route, loading } = useApi<QueryRouteOut | null>(
    () => selectedSessionId ? queriesApi.route(Number(selectedSessionId)) : Promise.resolve(null),
    [selectedSessionId],
  );

  const [selectedSighting, setSelectedSighting] = React.useState<SightingOut | null>(null);

  return (
    <PageContainer>
      <PageHeader title="Sighting Timeline" description="Multi-camera temporal sequence of detection events">
        <Select value={selectedSessionId} onValueChange={setSelectedSessionId}>
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
                <Clock className="h-6 w-6 text-ink-faint" />
              </div>
              <div>
                <p className="text-sm font-medium text-ink-secondary">No session selected</p>
                <p className="mt-1 text-xs text-ink-muted">
                  {doneSessions.length > 0
                    ? "Select a completed session above to view the sighting timeline."
                    : "Run a Re-ID search first to generate timeline data."}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      ) : loading ? (
        <Skeleton className="h-[300px] w-full rounded-lg" />
      ) : route ? (
        <div className="space-y-4">
          <TimelineView route={route} onSelect={setSelectedSighting} selected={selectedSighting} />
          {selectedSighting && <SightingDetail sighting={selectedSighting} />}
        </div>
      ) : null}
    </PageContainer>
  );
}

// ── Timeline View ──────────────────────────────────────────────────────────────

function TimelineView({
  route,
  onSelect,
  selected,
}: {
  route: QueryRouteOut;
  onSelect: (s: SightingOut) => void;
  selected: SightingOut | null;
}) {
  const sightings = route.sightings;

  // Calculate time range from real timestamps
  const times = sightings
    .map((s) => s.first_seen ? new Date(s.first_seen).getTime() : 0)
    .filter((t) => t > 0);
  const minTime = times.length ? Math.min(...times) : 0;
  const maxTime = times.length ? Math.max(...times) : 0;
  const range = maxTime - minTime || 1;

  // If no timestamps, show a placeholder
  if (times.length === 0) {
    return (
      <Card>
        <CardHeader><CardTitle>Timeline</CardTitle></CardHeader>
        <CardContent>
          <p className="py-8 text-center text-sm text-ink-muted">
            No timestamped sightings available for this session.
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Sighting Timeline</CardTitle>
        <Badge variant="secondary">{sightings.length} sightings</Badge>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Time axis */}
        <div className="flex items-center gap-2 pl-[120px]">
          <span className="text-xs font-mono text-ink-muted">{formatTime(new Date(minTime).toISOString())}</span>
          <div className="flex-1 border-t border-border" />
          <span className="text-xs font-mono text-ink-muted">{formatTime(new Date(maxTime).toISOString())}</span>
        </div>

        {/* Camera tracks */}
        {CAMERA_TRACKS.map((cam) => {
          const camSightings = sightings.filter((s) => s.camera_id === cam.id);
          return (
            <div key={cam.id} className="flex items-center gap-3">
              <div className="w-[110px] flex-shrink-0">
                <div className="text-sm font-semibold text-ink">{cam.id}</div>
                <div className="text-[10px] text-ink-muted">{cam.loc}</div>
              </div>
              <div className="relative flex-1 h-12 rounded-md border border-border bg-surface-muted/30">
                {/* Track line */}
                <div className="absolute top-1/2 left-0 right-0 h-px bg-border" />
                {camSightings.map((s) => {
                  const time = s.first_seen ? new Date(s.first_seen).getTime() : 0;
                  const leftPct = ((time - minTime) / range) * 100;
                  const isSelected = selected?.id === s.id;
                  const statusColor =
                    s.match_status === "CONFIDENT_MATCH" ? "bg-success border-success" :
                    s.match_status === "FACE_MISMATCH" ? "bg-error border-error" :
                    "bg-warning border-warning";

                  return (
                    <motion.button
                      key={s.id}
                      initial={{ scale: 0, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1 }}
                      transition={{ type: "spring", stiffness: 300, damping: 25 }}
                      className={cn(
                        "absolute top-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 transition-all",
                        statusColor,
                        isSelected ? "h-7 w-7 ring-2 ring-primary ring-offset-2 ring-offset-surface" : "h-5 w-5 hover:h-6 hover:w-6",
                      )}
                      style={{ left: `${leftPct}%` }}
                      onClick={() => onSelect(s)}
                      title={`${s.camera_id} · Track ${s.track_id} · ${formatTime(s.first_seen)} · ${s.best_confidence.toFixed(1)}`}
                    />
                  );
                })}
              </div>
            </div>
          );
        })}

        {sightings.length === 0 && (
          <div className="py-8 text-center text-sm text-ink-muted">
            No sightings recorded for this session.
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// ── Sighting Detail ────────────────────────────────────────────────────────────

function SightingDetail({ sighting }: { sighting: SightingOut }) {
  const url = cropUrl(sighting.crop_path);
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
    >
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <MapPin className="h-4 w-4 text-primary" />
            {sighting.camera_id} · Track {sighting.track_id}
            <Badge
              variant={
                sighting.match_status === "CONFIDENT_MATCH" ? "success" :
                sighting.match_status === "FACE_MISMATCH" ? "destructive" : "warning"
              }
              className="text-xs"
            >
              {sighting.match_status.replace(/_/g, " ")}
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex gap-4">
            <div className="h-24 w-18 flex-shrink-0 overflow-hidden rounded-md bg-surface-muted">
              {url ? (
                <img src={url} alt="Crop" className="h-full w-full object-cover" />
              ) : (
                <div className="flex h-full items-center justify-center">
                  <User className="h-8 w-8 text-ink-faint" />
                </div>
              )}
            </div>
            <div className="flex-1 space-y-2">
              <div className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm">
                <Detail label="First Seen" value={formatTime(sighting.first_seen)} />
                <Detail label="Last Seen" value={formatTime(sighting.last_seen)} />
                <Detail label="Match Score" value={`${sighting.best_confidence.toFixed(1)}%`} />
                <Detail label="Fusion Score" value={sighting.fusion_score.toFixed(4)} />
                <Detail label="Appearance" value={sighting.appearance_score.toFixed(4)} />
                <Detail label="Spatial" value={sighting.spatial_score.toFixed(4)} />
                <Detail label="Temporal" value={sighting.temporal_score.toFixed(4)} />
                <Detail label="Backbone" value={sighting.active_backbone} />
              </div>
              {sighting.face_veto_applied && (
                <div className="mt-2 rounded-md bg-error/5 p-2 text-xs text-error">
                  Face veto applied: {sighting.face_veto_reason || "different identity detected"}
                </div>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}

function Detail({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-xs text-ink-muted">{label}</span>
      <span className="text-xs font-mono font-medium text-ink">{value}</span>
    </div>
  );
}
