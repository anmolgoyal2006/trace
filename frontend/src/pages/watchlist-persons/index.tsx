import * as React from "react";
import { motion, AnimatePresence } from "motion/react";
import {
  Users, User, Plus, Trash2, Upload, Search, ShieldAlert, ShieldCheck,
  type LucideIcon,
} from "lucide-react";
import { PageHeader, PageContainer } from "@/components/common/page-header";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Separator } from "@/components/ui/separator";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter, DialogClose,
} from "@/components/ui/dialog";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { personsApi } from "@/services/api";
import { useApi } from "@/hooks/use-api";
import type { PersonOut } from "@/types/api";
import { cn, refPhotoUrl, formatDateTime } from "@/lib/utils";

const WATCHLIST_VARIANTS: Record<string, { label: string; variant: "default" | "destructive" | "warning" | "secondary" }> = {
  none: { label: "Standard", variant: "secondary" },
  missing: { label: "Missing", variant: "destructive" },
  suspect: { label: "Suspect", variant: "warning" },
  poi: { label: "POI", variant: "default" },
};

export default function WatchlistPersons() {
  const { data: persons, loading, refetch } = useApi<PersonOut[]>(() => personsApi.list());
  const [search, setSearch] = React.useState("");
  const [showRegister, setShowRegister] = React.useState(false);
  const [deleteId, setDeleteId] = React.useState<number | null>(null);

  const filtered = persons?.filter((p) =>
    p.name.toLowerCase().includes(search.toLowerCase()) ||
    p.alias?.toLowerCase().includes(search.toLowerCase()),
  ) ?? [];

  return (
    <PageContainer>
      <PageHeader title="Watchlist & Persons" description="Registered target profiles and reference embeddings">
        <Button onClick={() => setShowRegister(true)}>
          <Plus className="h-4 w-4" />
          Register Person
        </Button>
      </PageHeader>

      {/* Search */}
      <div className="relative max-w-md">
        <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-ink-faint" />
        <Input
          placeholder="Search persons…"
          className="pl-9"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </div>

      {/* Person grid */}
      {loading ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <Skeleton key={i} className="h-56 w-full" />
          ))}
        </div>
      ) : filtered.length === 0 ? (
        <Card>
          <CardContent className="py-16">
            <div className="flex flex-col items-center gap-3 text-center">
              <div className="flex h-12 w-12 items-center justify-center rounded-full bg-surface-muted">
                <Users className="h-6 w-6 text-ink-faint" />
              </div>
              <div>
                <p className="text-sm font-medium text-ink-secondary">No persons registered</p>
                <p className="mt-1 text-xs text-ink-muted">
                  Register a person and upload a reference photo to begin.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {filtered.map((person, idx) => (
            <PersonCard
              key={person.id}
              person={person}
              index={idx}
              onEnroll={refetch}
              onDelete={() => setDeleteId(person.id)}
            />
          ))}
        </div>
      )}

      {/* Register dialog */}
      <RegisterDialog open={showRegister} onOpenChange={setShowRegister} onRegistered={refetch} />

      {/* Delete confirmation */}
      <DeleteDialog
        personId={deleteId}
        persons={persons}
        onClose={() => setDeleteId(null)}
        onDeleted={refetch}
      />
    </PageContainer>
  );
}

// ── Person Card ────────────────────────────────────────────────────────────────

function PersonCard({
  person, index, onEnroll, onDelete,
}: {
  person: PersonOut;
  index: number;
  onEnroll: () => void;
  onDelete: () => void;
}) {
  const url = refPhotoUrl(person.reference_image_path);
  const wlVariant = WATCHLIST_VARIANTS[person.watchlist_status] ?? WATCHLIST_VARIANTS.none;
  const enrollRef = React.useRef<HTMLInputElement>(null);

  const handleEnroll = async (file: File) => {
    const fd = new FormData();
    fd.append("photo", file);
    try {
      await personsApi.enroll(person.id, fd);
      onEnroll();
    } catch {
      // non-fatal
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.04, duration: 0.2 }}
    >
      <Card className="overflow-hidden">
        <CardContent className="p-4">
          <div className="flex items-start gap-3">
            <div className="h-16 w-16 flex-shrink-0 overflow-hidden rounded-full border-2 border-border bg-surface-muted">
              {url ? (
                <img src={url} alt={person.name} className="h-full w-full object-cover" />
              ) : (
                <div className="flex h-full items-center justify-center">
                  <User className="h-7 w-7 text-ink-faint" />
                </div>
              )}
            </div>
            <div className="flex-1 min-w-0">
              <h3 className="truncate text-sm font-semibold text-ink">{person.name}</h3>
              {person.alias && (
                <p className="truncate text-xs text-ink-muted">{person.alias}</p>
              )}
              <Badge variant={wlVariant.variant} className="mt-1.5 text-[10px]">
                {wlVariant.label}
              </Badge>
            </div>
          </div>

          {person.description && (
            <p className="mt-3 line-clamp-2 text-xs text-ink-secondary">{person.description}</p>
          )}

          <div className="mt-3 flex items-center gap-2 text-xs">
            {person.has_embedding ? (
              <Badge variant="success" className="text-[10px]">
                <ShieldCheck className="mr-1 h-3 w-3" />
                Enrolled
              </Badge>
            ) : (
              <Badge variant="secondary" className="text-[10px]">
                <ShieldAlert className="mr-1 h-3 w-3" />
                No embedding
              </Badge>
            )}
            <span className="text-ink-muted">{formatDateTime(person.created_at)}</span>
          </div>

          {/* Actions */}
          <div className="mt-3 flex gap-2">
            <Button
              variant="outline"
              size="sm"
              className="flex-1 text-xs"
              onClick={() => enrollRef.current?.click()}
            >
              <Upload className="h-3 w-3" />
              Enroll Photo
            </Button>
            <input
              ref={enrollRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleEnroll(file);
              }}
            />
            <Button variant="ghost" size="icon" className="h-8 w-8 text-error" onClick={onDelete}>
              <Trash2 className="h-3.5 w-3.5" />
            </Button>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}

// ── Register Dialog ────────────────────────────────────────────────────────────

function RegisterDialog({
  open, onOpenChange, onRegistered,
}: {
  open: boolean;
  onOpenChange: (v: boolean) => void;
  onRegistered: () => void;
}) {
  const [name, setName] = React.useState("");
  const [alias, setAlias] = React.useState("");
  const [description, setDescription] = React.useState("");
  const [watchlistStatus, setWatchlistStatus] = React.useState<string>("none");
  const [photoFile, setPhotoFile] = React.useState<File | null>(null);
  const [photoPreview, setPhotoPreview] = React.useState<string | null>(null);
  const [saving, setSaving] = React.useState(false);
  const [error, setError] = React.useState("");
  const photoRef = React.useRef<HTMLInputElement>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) { setError("Name is required"); return; }
    setSaving(true);
    setError("");
    try {
      const fd = new FormData();
      fd.append("name", name);
      fd.append("alias", alias);
      fd.append("description", description);
      fd.append("watchlist_status", watchlistStatus);
      if (photoFile) fd.append("photo", photoFile);
      await personsApi.create(fd);
      onRegistered();
      onOpenChange(false);
      // Reset form
      setName(""); setAlias(""); setDescription(""); setWatchlistStatus("none");
      setPhotoFile(null); setPhotoPreview(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to register person");
    } finally {
      setSaving(false);
    }
  };

  const handlePhoto = (file: File) => {
    setPhotoFile(file);
    const reader = new FileReader();
    reader.onload = (e) => setPhotoPreview(e.target?.result as string);
    reader.readAsDataURL(file);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Register New Person</DialogTitle>
          <DialogDescription>Add a new target profile with optional reference photo.</DialogDescription>
        </DialogHeader>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-1.5">
            <Label>Full Name *</Label>
            <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. John Doe" required />
          </div>
          <div className="space-y-1.5">
            <Label>Alias / Codename</Label>
            <Input value={alias} onChange={(e) => setAlias(e.target.value)} placeholder="e.g. Subject-Alpha" />
          </div>
          <div className="space-y-1.5">
            <Label>Description / Notes</Label>
            <Textarea value={description} onChange={(e) => setDescription(e.target.value)} placeholder="Physical features, attire notes…" />
          </div>
          <div className="space-y-1.5">
            <Label>Watchlist Status</Label>
            <Select value={watchlistStatus} onValueChange={setWatchlistStatus}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="none">Standard</SelectItem>
                <SelectItem value="missing">Missing</SelectItem>
                <SelectItem value="suspect">Suspect</SelectItem>
                <SelectItem value="poi">POI</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-1.5">
            <Label>Reference Photo (optional)</Label>
            <div
              className="flex min-h-[100px] cursor-pointer flex-col items-center justify-center rounded-md border-2 border-dashed border-border bg-surface-muted/50 transition-colors hover:border-primary/40"
              onClick={() => photoRef.current?.click()}
            >
              <input ref={photoRef} type="file" accept="image/*" className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if (f) handlePhoto(f); }} />
              {photoPreview ? (
                <img src={photoPreview} alt="Preview" className="max-h-[120px] rounded-md object-contain" />
              ) : (
                <>
                  <Upload className="h-6 w-6 text-ink-faint" />
                  <span className="mt-1 text-xs text-ink-muted">Upload reference photo</span>
                </>
              )}
            </div>
          </div>
          {error && <p className="text-sm text-error">{error}</p>}
          <DialogFooter>
            <Button type="submit" disabled={saving}>
              {saving ? "Saving…" : "Save Profile"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

// ── Delete Dialog ──────────────────────────────────────────────────────────────

function DeleteDialog({
  personId, persons, onClose, onDeleted,
}: {
  personId: number | null;
  persons: PersonOut[] | null;
  onClose: () => void;
  onDeleted: () => void;
}) {
  const [deleting, setDeleting] = React.useState(false);
  const person = persons?.find((p) => p.id === personId);

  const handleDelete = async () => {
    if (!personId) return;
    setDeleting(true);
    try {
      await personsApi.delete(personId);
      onDeleted();
      onClose();
    } catch {
      // non-fatal
    } finally {
      setDeleting(false);
    }
  };

  return (
    <Dialog open={personId !== null} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Delete Person</DialogTitle>
          <DialogDescription>
            This action is permanent. All reference embeddings will be lost.
          </DialogDescription>
        </DialogHeader>
        <p className="text-sm text-ink-secondary">
          Are you sure you want to delete{" "}
          <span className="font-semibold text-ink">{person?.name}</span>?
        </p>
        <DialogFooter>
          <DialogClose asChild>
            <Button variant="outline">Cancel</Button>
          </DialogClose>
          <Button variant="destructive" onClick={handleDelete} disabled={deleting}>
            <Trash2 className="h-4 w-4" />
            {deleting ? "Deleting…" : "Delete"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
