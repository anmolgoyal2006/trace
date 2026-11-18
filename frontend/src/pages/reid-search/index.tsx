import * as React from "react";
import { motion, AnimatePresence } from "motion/react";
import {
  Search,
  Upload,
  User,
  ImageIcon,
  Fingerprint,
  Eye,
  ScanFace,
  AlertCircle,
  CheckCircle2,
  XCircle,
  Info,
  ArrowRight,
} from "lucide-react";
import { PageHeader, PageContainer } from "@/components/common/page-header";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { Separator } from "@/components/ui/separator";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { personsApi, queriesApi } from "@/services/api";
import { useApi } from "@/hooks/use-api";
import { useWsEvent } from "@/hooks/use-websocket";
import type { PersonOut, QueryRouteOut, CandidateTrackOut, WsEvent, MatchDecision } from "@/types/api";
import { cn, cropUrl, formatTime } from "@/lib/utils";

export default function ReIDSearch() {
  const { data: persons } = useApi<PersonOut[]>(() => personsApi.list());
  const enrolled = persons?.filter((p) => p.has_embedding) ?? [];

  const [photoFile, setPhotoFile] = React.useState<File | null>(null);
  const [photoPreview, setPhotoPreview] = React.useState<string | null>(null);
  const [selectedPersonId, setSelectedPersonId] = React.useState<string>("");
  const [submitting, setSubmitting] = React.useState(false);
  const [progress, setProgress] = React.useState(0);
  const [progressMsg, setProgressMsg] = React.useState("");
  const [sessionId, setSessionId] = React.useState<number | null>(null);
  const [route, setRoute] = React.useState<QueryRouteOut | null>(null);
  const [selectedCandidate, setSelectedCandidate] = React.useState<CandidateTrackOut | null>(null);

  // WebSocket event handling
  useWsEvent(
    (event: WsEvent) => {
      // Only events with session_id are relevant to this view
      if (event.type === "alert") return;
      if ("session_id" in event && event.session_id !== sessionId) return;
      switch (event.type) {
        case "query_progress":
          setProgress(event.progress_pct);
          setProgressMsg(event.message);
          break;
        case "sighting_found":
          setProgressMsg(`Candidate found on ${event.sighting.camera_id}`);
          break;
        case "route_complete":
          setProgress(100);
          setProgressMsg("Analysis complete");
          setRoute(event.route);
          setSubmitting(false);
          break;
        case "error":
          setProgressMsg(event.message);
          setSubmitting(false);
          break;
      }
    },
    [sessionId],
  );

  const handleSubmit = async () => {
    setSubmitting(true);
    setProgress(5);
    setProgressMsg("Submitting query…");
    setRoute(null);

    const fd = new FormData();
    const activeTab = photoFile ? "photo" : "person";

    if (activeTab === "person" && selectedPersonId) {
      fd.append("person_id", selectedPersonId);
    } else if (activeTab === "photo" && photoFile) {
      fd.append("image", photoFile);
    } else {
      setSubmitting(false);
      setProgress(0);
      setProgressMsg("");
      return;
    }

    try {
      const session = await queriesApi.submit(fd);
      setSessionId(session.id);
      setProgress(15);
      setProgressMsg(`Session #${session.id} — running OSNet Re-ID pipeline…`);
    } catch (err) {
      setProgressMsg(err instanceof Error ? err.message : "Failed to submit query");
      setSubmitting(false);
    }
  };

  const handlePhotoDrop = (file: File) => {
    setPhotoFile(file);
    const reader = new FileReader();
    reader.onload = (e) => setPhotoPreview(e.target?.result as string);
    reader.readAsDataURL(file);
  };

  const canSubmit = (selectedPersonId || photoFile) && !submitting;

  return (
    <PageContainer>
      <PageHeader
        title="Re-ID Search"
        description="Match a target identity across camera galleries using deep appearance features"
      />

      <div className="grid gap-5 lg:grid-cols-[400px_1fr]">
        {/* Left: Search configuration */}
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Target Input</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <Tabs defaultValue="person">
                <TabsList className="w-full">
                  <TabsTrigger value="person" className="flex-1">
                    <User className="mr-1.5 h-3.5 w-3.5" />
                    Registered Person
                  </TabsTrigger>
                  <TabsTrigger value="photo" className="flex-1">
                    <Upload className="mr-1.5 h-3.5 w-3.5" />
                    Reference Photo
                  </TabsTrigger>
                </TabsList>
                <TabsContent value="person" className="space-y-2">
                  <Label htmlFor="person-select">Select Registered Person</Label>
                  {!persons ? (
                    <Skeleton className="h-9 w-full" />
                  ) : enrolled.length === 0 ? (
                    <div className="rounded-md border border-border bg-surface-muted p-3 text-sm text-ink-muted">
                      No enrolled persons found. Register a person and upload a reference photo first.
                    </div>
                  ) : (
                    <Select value={selectedPersonId} onValueChange={setSelectedPersonId}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select a person…" />
                      </SelectTrigger>
                      <SelectContent>
                        {enrolled.map((p) => (
                          <SelectItem key={p.id} value={String(p.id)}>
                            {p.name}
                            {p.alias ? ` (${p.alias})` : ""} · {p.watchlist_status.toUpperCase()}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  )}
                  <p className="text-xs text-ink-muted">
                    Only persons with enrolled OSNet embeddings are available.
                  </p>
                </TabsContent>
                <TabsContent value="photo" className="space-y-2">
                  <Label>Reference Image</Label>
                  <PhotoDropzone onFileSelect={handlePhotoDrop} preview={photoPreview} />
                  <p className="text-xs text-ink-muted">
                    PNG or JPG — one person per photo.
                  </p>
                </TabsContent>
              </Tabs>

              {/* Submit button */}
              <Button
                className="w-full"
                size="lg"
                disabled={!canSubmit}
                onClick={handleSubmit}
              >
                {submitting ? (
                  <>
                    <div className="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                    Searching…
                  </>
                ) : (
                  <>
                    <Search className="h-4 w-4" />
                    Execute Re-ID Search
                  </>
                )}
              </Button>

              {/* Progress */}
              {submitting && (
                <div className="space-y-2">
                  <Progress value={progress} />
                  <p className="text-xs text-ink-muted">{progressMsg}</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Right: Results */}
        <div className="space-y-4">
          {route ? (
            <SearchResults
              route={route}
              selectedCandidate={selectedCandidate}
              onSelectCandidate={setSelectedCandidate}
            />
          ) : submitting ? (
            <Card>
              <CardContent className="py-16">
                <div className="flex flex-col items-center gap-4">
                  <div className="h-8 w-8 animate-spin rounded-full border-2 border-border border-t-primary" />
                  <p className="text-sm text-ink-muted">{progressMsg}</p>
                </div>
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardContent className="py-16">
                <div className="flex flex-col items-center gap-3 text-center">
                  <div className="flex h-12 w-12 items-center justify-center rounded-full bg-surface-muted">
                    <Search className="h-6 w-6 text-ink-faint" />
                  </div>
                  <div>
                    <p className="text-sm font-medium text-ink-secondary">No active search</p>
                    <p className="mt-1 text-xs text-ink-muted">
                      Select a registered person or upload a reference photo to begin.
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </PageContainer>
  );
}

// ── Photo Dropzone ────────────────────────────────────────────────────────────

function PhotoDropzone({
  onFileSelect,
  preview,
}: {
  onFileSelect: (file: File) => void;
  preview: string | null;
}) {
  const [dragOver, setDragOver] = React.useState(false);
  const inputRef = React.useRef<HTMLInputElement>(null);

  return (
    <div
      className={cn(
        "relative flex min-h-[160px] cursor-pointer flex-col items-center justify-center rounded-md border-2 border-dashed transition-colors",
        dragOver ? "border-primary bg-primary/5" : "border-border bg-surface-muted/50",
      )}
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith("image/")) onFileSelect(file);
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onFileSelect(file);
        }}
      />
      {preview ? (
        <img src={preview} alt="Preview" className="max-h-[180px] rounded-md object-contain" />
      ) : (
        <>
          <Upload className="h-8 w-8 text-ink-faint" />
          <p className="mt-2 text-sm text-ink-secondary">Drop photo or browse</p>
        </>
      )}
    </div>
  );
}

// ── Search Results ────────────────────────────────────────────────────────────

function SearchResults({
  route,
  selectedCandidate,
  onSelectCandidate,
}: {
  route: QueryRouteOut;
  selectedCandidate: CandidateTrackOut | null;
  onSelectCandidate: (c: CandidateTrackOut) => void;
}) {
  const decision = route.match_decision;
  const candidates = route.top_candidates || [];

  return (
    <>
      {/* Decision banner */}
      <DecisionBanner decision={decision} totalCameras={route.total_cameras_matched} confidence={route.route_confidence} />

      {/* Candidates grid */}
      {candidates.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Top Candidates</CardTitle>
            <Badge variant="secondary">{candidates.length} for review</Badge>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
              {candidates.map((candidate, idx) => (
                <CandidateCard
                  key={`${candidate.camera_id}-${candidate.track_id}`}
                  candidate={candidate}
                  index={idx}
                  isSelected={selectedCandidate?.camera_id === candidate.camera_id && selectedCandidate?.track_id === candidate.track_id}
                  onSelect={() => onSelectCandidate(candidate)}
                />
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Explainable AI: Why this match? */}
      {selectedCandidate && (
        <WhyThisMatch candidate={selectedCandidate} />
      )}

      {/* Sightings evidence */}
      {route.sightings.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Per-Camera Sighting Evidence</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {route.sightings.map((s) => (
                <SightingEvidence key={s.id} sighting={s} />
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </>
  );
}

// ── Decision Banner ────────────────────────────────────────────────────────────

function DecisionBanner({ decision, totalCameras, confidence }: { decision: MatchDecision; totalCameras: number; confidence: number }) {
  const configs: Record<MatchDecision, { label: string; desc: string; variant: "success" | "warning" | "destructive" }> = {
    CONFIDENT_MATCH: {
      label: "Candidate Found",
      desc: `${totalCameras} camera(s) matched. Match scores are similarity-derived — NOT identity probabilities. Visual verification required.`,
      variant: "success",
    },
    POSSIBLE_MATCH_REVIEW: {
      label: "Candidates for Review",
      desc: `No confident match. ${totalCameras > 0 ? `${totalCameras} candidate(s) below threshold.` : ""} Top candidates shown for human review.`,
      variant: "warning",
    },
    NO_CONFIDENT_MATCH: {
      label: "No Confident Match",
      desc: "Available evidence does not support identifying this person. Top candidates shown for manual review.",
      variant: "destructive",
    },
  };

  const config = configs[decision];
  const pct = (confidence * 100).toFixed(1);

  return (
    <Card className={cn(
      "border-l-4",
      config.variant === "success" && "border-l-success",
      config.variant === "warning" && "border-l-warning",
      config.variant === "destructive" && "border-l-error",
    )}>
      <CardContent className="flex items-center gap-4 py-4">
        {config.variant === "success" ? (
          <CheckCircle2 className="h-6 w-6 text-success" />
        ) : config.variant === "warning" ? (
          <AlertCircle className="h-6 w-6 text-warning" />
        ) : (
          <XCircle className="h-6 w-6 text-error" />
        )}
        <div className="flex-1">
          <p className="text-sm font-semibold text-ink">{config.label}</p>
          <p className="mt-0.5 text-xs text-ink-muted">{config.desc}</p>
        </div>
        {confidence > 0 && (
          <div className="text-right">
            <div className="text-xl font-bold text-ink">{pct}%</div>
            <div className="text-[10px] text-ink-muted">match score</div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// ── Candidate Card ────────────────────────────────────────────────────────────

function CandidateCard({
  candidate,
  index,
  isSelected,
  onSelect,
}: {
  candidate: CandidateTrackOut;
  index: number;
  isSelected: boolean;
  onSelect: () => void;
}) {
  const url = cropUrl(candidate.crop_path);
  const statusColor =
    candidate.match_status === "CONFIDENT_MATCH" ? "success" :
    candidate.match_status === "FACE_MISMATCH" ? "destructive" :
    candidate.match_status === "POSSIBLE_MATCH_REVIEW" ? "warning" : "secondary";

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05, duration: 0.2 }}
      className={cn(
        "cursor-pointer rounded-lg border-2 p-3 transition-colors",
        isSelected ? "border-primary bg-primary/5" : "border-border bg-surface hover:border-border-strong",
      )}
      onClick={onSelect}
    >
      <div className="mb-2 aspect-[3/4] overflow-hidden rounded-md bg-surface-muted">
        {url ? (
          <img src={url} alt="Candidate" className="h-full w-full object-cover" />
        ) : (
          <div className="flex h-full items-center justify-center">
            <User className="h-8 w-8 text-ink-faint" />
          </div>
        )}
      </div>
      <div className="flex items-center justify-between">
        <Badge variant={statusColor} className="text-[10px]">
          {candidate.match_status.replace(/_/g, " ")}
        </Badge>
        <span className="text-lg font-bold text-ink">
          {candidate.best_confidence.toFixed(1)}
        </span>
      </div>
      <div className="mt-1 text-xs text-ink-muted">
        {candidate.camera_id} · Track {candidate.track_id}
      </div>
      {candidate.first_seen && (
        <div className="text-[10px] text-ink-faint">{formatTime(candidate.first_seen)}</div>
      )}
    </motion.div>
  );
}

// ── Why This Match (Explainable AI) ────────────────────────────────────────────

function WhyThisMatch({ candidate }: { candidate: CandidateTrackOut }) {
  const [expanded, setExpanded] = React.useState(true);

  return (
    <Card>
      <CardHeader>
        <button
          onClick={() => setExpanded(!expanded)}
          className="flex items-center gap-2 text-left"
        >
          <Fingerprint className="h-4 w-4 text-primary" />
          <CardTitle>Why This Match?</CardTitle>
          <Badge variant="outline" className="ml-1 text-xs">
            {candidate.camera_id} · Track {candidate.track_id}
          </Badge>
        </button>
      </CardHeader>
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
          >
            <CardContent className="space-y-4">
              {/* Overall score */}
              <ScoreBar
                label="Overall Match Score"
                value={candidate.best_confidence}
                max={100}
                suffix="%"
                note="Similarity-derived, not a probability"
              />

              <Separator />

              {/* Body appearance */}
              <EvidenceSection
                icon={<Fingerprint className="h-4 w-4" />}
                title="Body Appearance"
                model={candidate.active_backbone}
                available={true}
              >
                <ScoreBar label="Appearance" value={candidate.appearance_score} max={1} />
                <ScoreBar label="Fused" value={candidate.fused_score} max={1} />
              </EvidenceSection>

              {/* Face similarity */}
              <EvidenceSection
                icon={<ScanFace className="h-4 w-4" />}
                title="Face Similarity"
                model="SCRFD + ArcFace"
                available={candidate.face_sim !== null && candidate.face_sim !== undefined}
              >
                <ScoreBar label="Face Similarity" value={(candidate.face_sim ?? 0) * 100} max={100} suffix="%" />
                {candidate.face_coverage !== null && (
                  <div className="text-xs text-ink-muted">
                    Coverage: {(candidate.face_coverage * 100).toFixed(0)}%
                  </div>
                )}
                {candidate.face_veto_applied && (
                  <div className="flex items-center gap-1.5 text-xs text-error">
                    <XCircle className="h-3.5 w-3.5" />
                    Face veto applied — different identity
                  </div>
                )}
              </EvidenceSection>

              {/* KPR part-aware */}
              <EvidenceSection
                icon={<Eye className="h-4 w-4" />}
                title="KPR Part-Aware"
                model="KPR (ECCV 2024)"
                available={candidate.kpr_sim !== null && candidate.kpr_sim !== undefined}
              >
                <ScoreBar label="KPR Similarity" value={(candidate.kpr_sim ?? 0) * 100} max={100} suffix="%" />
              </EvidenceSection>

              {/* Disclaimer */}
              <div className="flex items-start gap-2 rounded-md bg-surface-muted p-3">
                <Info className="mt-0.5 h-3.5 w-3.5 flex-shrink-0 text-ink-muted" />
                <p className="text-xs text-ink-muted">
                  Evidence is produced by the matching pipeline. Scores are similarity-derived values, not calibrated identity probabilities.
                  Visual verification by a human operator is required before any identity conclusion.
                </p>
              </div>
            </CardContent>
          </motion.div>
        )}
      </AnimatePresence>
    </Card>
  );
}

function EvidenceSection({
  icon,
  title,
  model,
  available,
  children,
}: {
  icon: React.ReactNode;
  title: string;
  model: string;
  available: boolean;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <span className="text-ink-secondary">{icon}</span>
        <span className="text-sm font-medium text-ink">{title}</span>
        <Badge variant="outline" className="text-[10px]">{model}</Badge>
        {!available && (
          <Badge variant="secondary" className="text-[10px]">Unavailable</Badge>
        )}
      </div>
      {available ? children : (
        <p className="text-xs text-ink-muted pl-6">No evidence available for this signal.</p>
      )}
    </div>
  );
}

function ScoreBar({ label, value, max, suffix, note }: { label: string; value: number; max: number; suffix?: string; note?: string }) {
  const pct = Math.min((value / max) * 100, 100);
  const color = pct >= 70 ? "bg-success" : pct >= 50 ? "bg-warning" : "bg-error";
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs">
        <span className="text-ink-secondary">{label}</span>
        <span className="font-mono font-medium text-ink">
          {value.toFixed(max === 100 ? 1 : 4)}{suffix}
        </span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-surface-muted">
        <div className={cn("h-full rounded-full transition-all duration-500", color)} style={{ width: `${pct}%` }} />
      </div>
      {note && <p className="text-[10px] text-ink-faint">{note}</p>}
    </div>
  );
}

// ── Sighting Evidence ────────────────────────────────────────────────────────

function SightingEvidence({ sighting }: { sighting: import("@/types/api").SightingOut }) {
  const url = cropUrl(sighting.crop_path);
  return (
    <div className="flex items-center gap-4 rounded-md border border-border p-3">
      <div className="h-16 w-12 flex-shrink-0 overflow-hidden rounded bg-surface-muted">
        {url ? (
          <img src={url} alt="Crop" className="h-full w-full object-cover" />
        ) : (
          <div className="flex h-full items-center justify-center">
            <User className="h-5 w-5 text-ink-faint" />
          </div>
        )}
      </div>
      <div className="flex-1">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-ink">{sighting.camera_id}</span>
          <span className="text-xs text-ink-muted">Track {sighting.track_id}</span>
          <Badge variant={sighting.match_status === "CONFIDENT_MATCH" ? "success" : "secondary"} className="text-[10px]">
            {sighting.match_status.replace(/_/g, " ")}
          </Badge>
        </div>
        <div className="mt-1 text-xs text-ink-muted">
          {formatTime(sighting.first_seen)} — {formatTime(sighting.last_seen)}
        </div>
        <div className="mt-1 text-xs text-ink-secondary">
          Score: {sighting.best_confidence.toFixed(1)} · Fusion: {sighting.fusion_score.toFixed(4)}
        </div>
        {sighting.face_veto_applied && (
          <div className="mt-1 flex items-center gap-1 text-xs text-error">
            <XCircle className="h-3 w-3" />
            Face veto: {sighting.face_veto_reason || "different identity"}
          </div>
        )}
      </div>
    </div>
  );
}
