import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/** Merge Tailwind classes intelligently. */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** Format ISO datetime or time-of-day string to display time string (HH:MM:SS). */
export function formatTime(isoStr: string | null | undefined): string {
  if (!isoStr) return "—";
  // Handle direct time strings like "10:05:12.57" or "10:05:12"
  const timeMatch = isoStr.match(/^(\d{1,2}:\d{2}(?::\d{2})?)/);
  if (timeMatch) {
    return timeMatch[1];
  }
  try {
    const d = new Date(isoStr);
    if (isNaN(d.getTime())) return isoStr;
    return d.toLocaleTimeString("en-GB", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  } catch {
    return isoStr;
  }
}

/** Parse any timestamp (ISO string, datetime, or HH:MM:SS.ms) into seconds. */
export function parseTimeToSeconds(timeStr: string | null | undefined): number | null {
  if (!timeStr) return null;
  const match = timeStr.match(/^(\d{1,2}):(\d{2})(?::(\d{2})(?:\.(\d+))?)?$/);
  if (match) {
    const h = parseInt(match[1], 10);
    const m = parseInt(match[2], 10);
    const s = parseInt(match[3] || "0", 10);
    const ms = match[4] ? parseFloat("0." + match[4]) : 0;
    return h * 3600 + m * 60 + s + ms;
  }
  const d = new Date(timeStr);
  if (!isNaN(d.getTime())) {
    return d.getTime() / 1000;
  }
  return null;
}

/** Format seconds into HH:MM:SS wall-clock format. */
export function formatSecondsToTime(seconds: number): string {
  const h = Math.floor(seconds / 3600) % 24;
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

/** Format ISO datetime to date + time. */
export function formatDateTime(isoStr: string | null | undefined): string {
  if (!isoStr) return "—";
  try {
    return new Date(isoStr).toLocaleString("en-GB", {
      day: "2-digit",
      month: "short",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return isoStr;
  }
}

/** Escape HTML to prevent XSS. */
export function escapeHtml(str: string | null | undefined): string {
  if (!str) return "";
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/** Build a crop image URL from a crop_path. */
export function cropUrl(cropPath: string | null | undefined): string | null {
  if (!cropPath) return null;
  const clean = cropPath.replace(/\\/g, "/");
  const filename = clean.split("/crops/").pop();
  return filename ? `/crops/${filename}` : null;
}

/** Build a reference photo URL from a path. */
export function refPhotoUrl(imagePath: string | null | undefined): string | null {
  if (!imagePath) return null;
  const clean = imagePath.replace(/\\/g, "/");
  const filename = clean.split("/reference_photos/").pop();
  return filename ? `/uploads/reference_photos/${filename}` : null;
}

/** Format bytes to human readable. */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

/** Sleep helper. */
export function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}
