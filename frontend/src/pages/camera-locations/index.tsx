import * as React from "react";
import { motion } from "motion/react";
import { MapPin, Camera, Save, Search, ArrowRight } from "lucide-react";
import { PageHeader, PageContainer } from "@/components/common/page-header";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Separator } from "@/components/ui/separator";
import { camerasApi } from "@/services/api";
import { useApi } from "@/hooks/use-api";
import type { CameraWithGraph } from "@/types/api";
import { cn } from "@/lib/utils";

export default function CameraLocations() {
  const { data: cameras, loading, refetch } = useApi<CameraWithGraph[]>(() => camerasApi.list());
  const [selectedId, setSelectedId] = React.useState<string | null>(null);
  const [search, setSearch] = React.useState("");

  const filtered = cameras?.filter((c) =>
    c.id.toLowerCase().includes(search.toLowerCase()) ||
    c.location?.toLowerCase().includes(search.toLowerCase()),
  ) ?? [];

  const selected = cameras?.find((c) => c.id === selectedId) ?? null;

  return (
    <PageContainer>
      <PageHeader title="Camera Locations" description="Camera network management and configuration" />

      <div className="grid gap-4 lg:grid-cols-[320px_1fr]">
        {/* Camera list */}
        <div className="space-y-3">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-ink-faint" />
            <Input
              placeholder="Search cameras…"
              className="pl-9"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>

          <div className="space-y-2">
            {loading ? (
              [1, 2, 3].map((i) => <Skeleton key={i} className="h-16 w-full" />)
            ) : filtered.length === 0 ? (
              <div className="py-8 text-center text-sm text-ink-muted">No cameras found.</div>
            ) : (
              filtered.map((cam) => (
                <CameraListItem
                  key={cam.id}
                  camera={cam}
                  isSelected={selectedId === cam.id}
                  onSelect={() => setSelectedId(cam.id)}
                />
              ))
            )}
          </div>
        </div>

        {/* Detail panel */}
        {selected ? (
          <CameraDetail camera={selected} onUpdate={refetch} />
        ) : (
          <Card>
            <CardContent className="flex h-[400px] flex-col items-center justify-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-full bg-surface-muted">
                <MapPin className="h-6 w-6 text-ink-faint" />
              </div>
              <p className="text-sm text-ink-muted">Select a camera to inspect details</p>
            </CardContent>
          </Card>
        )}
      </div>
    </PageContainer>
  );
}

function CameraListItem({ camera, isSelected, onSelect }: { camera: CameraWithGraph; isSelected: boolean; onSelect: () => void }) {
  return (
    <motion.div
      initial={{ opacity: 0, x: 8 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.15 }}
      className={cn(
        "cursor-pointer rounded-lg border p-3 transition-colors",
        isSelected ? "border-primary bg-primary/5" : "border-border bg-surface hover:bg-surface-muted/50",
      )}
      onClick={onSelect}
    >
      <div className="flex items-center gap-3">
        <div className="flex h-9 w-9 items-center justify-center rounded-md bg-surface-muted">
          <Camera className="h-4 w-4 text-ink-secondary" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold text-ink">{camera.id}</span>
            <Badge variant={camera.is_active ? "success" : "secondary"} className="text-[10px]">
              {camera.is_active ? "Active" : "Inactive"}
            </Badge>
          </div>
          <div className="truncate text-xs text-ink-muted">{camera.location}</div>
        </div>
      </div>
    </motion.div>
  );
}

function CameraDetail({ camera, onUpdate }: { camera: CameraWithGraph; onUpdate: () => void }) {
  const [location, setLocation] = React.useState(camera.location);
  const [startTime, setStartTime] = React.useState(camera.start_time || "");
  const [saving, setSaving] = React.useState(false);
  const [transitEdits, setTransitEdits] = React.useState<Record<string, number>>({});

  const handleSave = async () => {
    setSaving(true);
    try {
      const updates: Record<string, number> = {};
      for (const [camId, secs] of Object.entries(transitEdits)) {
        updates[camId] = secs;
      }
      await camerasApi.patch(camera.id, {
        location: location !== camera.location ? location : undefined,
        start_time: startTime !== camera.start_time ? startTime : undefined,
        transit_updates: Object.keys(updates).length > 0 ? updates : undefined,
      });
      onUpdate();
    } catch (err) {
      // non-fatal
    } finally {
      setSaving(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
    >
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Camera className="h-4 w-4 text-primary" />
            {camera.id} — {camera.location}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-5">
          {/* Editable fields */}
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-1.5">
              <Label>Location Name</Label>
              <Input value={location} onChange={(e) => setLocation(e.target.value)} placeholder="e.g. Main Entrance" />
            </div>
            <div className="space-y-1.5">
              <Label>Recording Start Time (HH:MM:SS)</Label>
              <Input type="time" step="1" value={startTime} onChange={(e) => setStartTime(e.target.value)} />
            </div>
          </div>

          {/* Transit times */}
          {camera.connects_to.length > 0 && (
            <div className="space-y-2">
              <Label>Transit Times (seconds to adjacent cameras)</Label>
              {camera.connects_to.map((edge) => (
                <div key={edge.to_camera_id} className="flex items-center gap-3 rounded-md border border-border p-2.5">
                  <ArrowRight className="h-4 w-4 text-ink-muted" />
                  <span className="text-sm font-medium text-ink">{edge.to_camera_id}</span>
                  <div className="flex-1" />
                  <Input
                    type="number"
                    min="1"
                    max="3600"
                    className="w-24"
                    defaultValue={edge.avg_transit_sec}
                    onChange={(e) =>
                      setTransitEdits((prev) => ({
                        ...prev,
                        [edge.to_camera_id]: parseInt(e.target.value, 10) || 30,
                      }))
                    }
                  />
                  <span className="text-xs text-ink-muted">sec</span>
                </div>
              ))}
            </div>
          )}

          {/* Recent sightings summary */}
          <Separator />
          <div className="flex items-center justify-between">
            <span className="text-xs text-ink-muted">Description</span>
            <span className="text-sm text-ink">{camera.description || "—"}</span>
          </div>

          <Button onClick={handleSave} disabled={saving}>
            <Save className="h-4 w-4" />
            {saving ? "Saving…" : "Save Changes"}
          </Button>
        </CardContent>
      </Card>
    </motion.div>
  );
}
