import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/** Merge Tailwind classes intelligently. */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** Format ISO datetime to time string. */
export function formatTime(isoStr: string | null | undefined): string {
  if (!isoStr) return "—";
  try {
    return new Date(isoStr).toLocaleTimeString("en-GB", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  } catch {
    return isoStr;
  }
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
