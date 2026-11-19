import * as React from "react";
import { motion, AnimatePresence } from "motion/react";
import {
  Video, Upload, Camera, Save, Play, Loader2, CheckCircle2,
  XCircle, Clock, ArrowRight, ScanLine, Activity, Filter, Fingerprint,
  type LucideIcon,
} from "lucide-react";
import { PageHeader, PageContainer } from "@/components/common/page-header";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Separator } from "@/components/ui/separator";
import { camerasApi, uploadApi } from "@/services/api";
import { useApi } from "@/hooks/use-api";
import type { CameraWithGraph, UploadJobOut } from "@/types/api";
import { cn } from "@/lib/utils";

const PIPELINE_STEPS: { num: number; label: string; sub: string; icon: LucideIcon }[] = [
  { num: 1, label: "Detection", sub: "YOLOv8n", icon: ScanLine },
  { num: 2, label: "Tracking", sub: "ByteTrack", icon: Activity },
  { num: 3, label: "Crop Filter", sub: "Quality", icon: Filter },
  { num: 4, label: "Embeddings", sub: "OSNet Re-ID", icon: Fingerprint },
];

export default function PipelineVideoStudio() {
  const { data: cameras, refetch: refetchCameras } = useApi<CameraWithGraph[]>(() => camerasApi.list());
  const { data: jobs, refetch: refetchJobs } = useApi<UploadJobOut[]>(() => uploadApi.jobs());

  const [selectedCamera, setSelectedCamera] = React.useState<string>("C01");
  const [videoFile, setVideoFile] = React.useState<File | null>(null);
  const [uploading, setUploading] = React.useState(false);
  const [statusMsg, setStatusMsg] = React.useState("");

  // Poll jobs while any are processing
  React.useEffect(() => {
    const hasActive = jobs?.some((j) => j.status === "queued" || j.status === "processing");
    if (!hasActive) return;
    const interval = setInterval(() => refetchJobs(), 3000);
    return () => clearInterval(interval);
  }, [jobs, refetchJobs]);

  const handleUpload = async () => {
    if (!videoFile) return;
    setUploading(true);
    setStatusMsg("Uploading…");
    const fd = new FormData();
    fd.append("camera_id", selectedCamera);
    fd.append("video", videoFile);
    try {
      const job = await uploadApi.video(fd);
      setStatusMsg(`Job started (${job.job_id.slice(0, 8)}…)`);
      setVideoFile(null);
      refetchJobs();
    } catch (err) {
      setStatusMsg(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  return (
    <PageContainer>
      <PageHeader title="Pipeline Video Studio" description="Configure cameras, upload video, and run the AI processing pipeline" />

      <div className="space-y-4">
        {/* Step 1: Camera Configuration */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <span className="flex h-6 w-6 items-center justify-center rounded-full bg-primary text-xs font-bold text-white">1</span>
              Camera Configuration
            </CardTitle>
          </CardHeader>
          <CardContent>
            {cameras ? (
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                {cameras.map((cam) => (
                  <CameraConfigCard key={cam.id} camera={cam} onSave={refetchCameras} />
                ))}
              </div>
            ) : (
              <div className="grid gap-4 sm:grid-cols-3">
                {[1, 2, 3].map((i) => <Skeleton key={i} className="h-32 w-full" />)}
              </div>
            )}
          </CardContent>
        </Card>

        <div className="grid gap-4 lg:grid-cols-2">
          {/* Step 2: Upload */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <span className="flex h-6 w-6 items-center justify-center rounded-full bg-primary text-xs font-bold text-white">2</span>
                Upload & Launch Pipeline
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Pipeline steps */}
              <div className="grid grid-cols-4 gap-2 rounded-md border border-border bg-surface-muted/50 p-3">
                {PIPELINE_STEPS.map((step) => (
                  <div key={step.num} className="flex flex-col items-center gap-1 text-center">
                    <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/10">
                      <step.icon className="h-4 w-4 text-primary" />
                    </div>
                    <div>
                      <div className="text-xs font-medium text-ink">{step.label}</div>
                      <div className="text-[10px] text-ink-muted">{step.sub}</div>
                    </div>
                  </div>
                ))}
              </div>

              {/* Camera selector */}
              <div className="space-y-1.5">
                <Label>Target Camera</Label>
                <div className="grid grid-cols-3 gap-2">
                  {cameras?.map((cam) => (
                    <button
                      key={cam.id}
                      className={cn(
                        "flex flex-col items-center gap-1 rounded-md border-2 p-3 transition-colors",
                        selectedCamera === cam.id
                          ? "border-primary bg-primary/5"
                          : "border-border bg-surface hover:bg-surface-muted/50",
                      )}
                      onClick={() => setSelectedCamera(cam.id)}
                    >
                      <Camera className={cn("h-5 w-5", selectedCamera === cam.id ? "text-primary" : "text-ink-muted")} />
                      <span className="text-sm font-semibold text-ink">{cam.id}</span>
                      <span className="text-[10px] text-ink-muted">{cam.location}</span>
                    </button>
                  ))}
                </div>
              </div>

              {/* Video dropzone */}
              <div className="space-y-1.5">
                <Label>Surveillance Video</Label>
                <VideoDropzone
                  file={videoFile}
                  onFileSelect={setVideoFile}
                />
              </div>

              <Button className="w-full" size="lg" onClick={handleUpload} disabled={!videoFile || uploading}>
                {uploading ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Uploading…
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4" />
                    Launch Processing Job
                  </>
                )}
              </Button>
              {statusMsg && <p className="text-xs text-ink-muted">{statusMsg}</p>}
            </CardContent>
          </Card>

          {/* Step 3: Job Queue */}
          <Card>
            <CardHeader>
              <CardTitle>Job Queue</CardTitle>
              <Button variant="ghost" size="sm" onClick={refetchJobs}>
                <ArrowRight className="h-3 w-3" />
                Refresh
              </Button>
            </CardHeader>
            <CardContent>
              {jobs === null ? (
                <div className="space-y-2">
                  {[1, 2].map((i) => <Skeleton key={i} className="h-14 w-full" />)}
                </div>
              ) : jobs.length === 0 ? (
                <div className="py-8 text-center">
                  <div className="mx-auto flex h-10 w-10 items-center justify-center rounded-full bg-surface-muted">
                    <Video className="h-5 w-5 text-ink-faint" />
                  </div>
                  <p className="mt-2 text-sm text-ink-muted">No jobs yet</p>
                </div>
              ) : (
                <div className="space-y-2">
                  {jobs.map((job) => (
                    <JobRow key={job.job_id} job={job} />
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </PageContainer>
  );
}

// ── Camera Config Card ──────────────────────────────────────────────────────────

function CameraConfigCard({ camera, onSave }: { camera: CameraWithGraph; onSave: () => void }) {
  const [location, setLocation] = React.useState(camera.location);
  const [startTime, setStartTime] = React.useState(camera.start_time || "");
  const [saving, setSaving] = React.useState(false);
  const [saved, setSaved] = React.useState(false);

  const dirty = location !== camera.location || startTime !== (camera.start_time || "");

  const handleSave = async () => {
    if (!dirty) return;
    setSaving(true);
    try {
      await camerasApi.patch(camera.id, {
        location: location !== camera.location ? location : undefined,
        start_time: startTime !== camera.start_time ? startTime : undefined,
      });
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
      onSave();
    } catch {
      // non-fatal
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className={cn("rounded-lg border p-3 transition-colors", dirty ? "border-primary/40" : "border-border")}>
      <div className="mb-2 flex items-center gap-2">
        <span className="flex h-7 w-7 items-center justify-center rounded-md bg-primary/10 text-xs font-bold text-primary">{camera.id}</span>
        <span className="text-xs text-ink-muted">Camera Channel</span>
        {saved && <Badge variant="success" className="ml-auto text-[10px]">Saved</Badge>}
      </div>
      <div className="space-y-2">
        <div>
          <Label className="text-[10px]">LOCATION</Label>
          <Input value={location} onChange={(e) => setLocation(e.target.value)} className="h-8 text-sm" />
        </div>
        <div>
          <Label className="text-[10px]">START TIME</Label>
          <Input type="time" step="1" value={startTime} onChange={(e) => setStartTime(e.target.value)} className="h-8 text-sm" />
        </div>
      </div>
      <Button variant="outline" size="sm" className="mt-2 w-full text-xs" onClick={handleSave} disabled={!dirty || saving}>
        <Save className="h-3 w-3" />
        {saving ? "Saving…" : `Save ${camera.id}`}
      </Button>
    </div>
  );
}

// ── Video Dropzone ──────────────────────────────────────────────────────────────

function VideoDropzone({ file, onFileSelect }: { file: File | null; onFileSelect: (f: File) => void }) {
  const [dragOver, setDragOver] = React.useState(false);
  const inputRef = React.useRef<HTMLInputElement>(null);

  return (
    <div
      className={cn(
        "flex min-h-[120px] cursor-pointer flex-col items-center justify-center rounded-md border-2 border-dashed transition-colors",
        dragOver ? "border-primary bg-primary/5" : "border-border bg-surface-muted/50",
      )}
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        const f = e.dataTransfer.files[0];
        if (f && f.type.startsWith("video/")) onFileSelect(f);
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept="video/*"
        className="hidden"
        onChange={(e) => { const f = e.target.files?.[0]; if (f) onFileSelect(f); }}
      />
      {file ? (
        <div className="text-center">
          <Video className="mx-auto h-8 w-8 text-primary" />
          <p className="mt-1 text-sm font-medium text-ink">{file.name}</p>
          <p className="text-xs text-ink-muted">{(file.size / 1024 / 1024).toFixed(1)} MB</p>
        </div>
      ) : (
        <>
          <Upload className="h-7 w-7 text-ink-faint" />
          <p className="mt-2 text-sm text-ink-secondary">Drop video or browse</p>
          <p className="text-xs text-ink-muted">.mp4 · .avi · .mov · .mkv</p>
        </>
      )}
    </div>
  );
}

// ── Job Row ──────────────────────────────────────────────────────────────────────

function JobRow({ job }: { job: UploadJobOut }) {
  const statusConfig: Record<string, { icon: React.ReactNode; variant: "default" | "secondary" | "success" | "destructive" | "warning" }> = {
    queued: { icon: <Clock className="h-4 w-4" />, variant: "secondary" },
    processing: { icon: <Loader2 className="h-4 w-4 animate-spin" />, variant: "warning" },
    done: { icon: <CheckCircle2 className="h-4 w-4" />, variant: "success" },
    failed: { icon: <XCircle className="h-4 w-4" />, variant: "destructive" },
  };
  const cfg = statusConfig[job.status] ?? statusConfig.queued;

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex items-center gap-3 rounded-md border border-border p-3"
    >
      <div className={cn("flex h-9 w-9 items-center justify-center rounded-md", cfg.variant === "success" ? "bg-success/10" : cfg.variant === "destructive" ? "bg-error/10" : "bg-surface-muted")}>
        {cfg.icon}
      </div>
      <div className="flex-1 min-w-0">
        <div className="text-sm font-medium text-ink">
          {job.camera_id} — {job.filename}
        </div>
        {job.message && (
          <div className="truncate text-xs text-ink-muted">{job.message}</div>
        )}
      </div>
      <Badge variant={cfg.variant} className="text-[10px]">{job.status}</Badge>
    </motion.div>
  );
}
