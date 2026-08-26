"""
per_track_sanity.py — Phase 3.6
Per-track sanity check for the final crop dataset.

Cross-references:
  - tracks_C01.json         → all unique track IDs + frame records
  - crop_quality_report_phase3_final.json → rejection details
  - crops_metadata_phase3_final.json      → accepted crops

Produces a table:
  track_id | sampled | accepted | rejected | status
and lists any tracks with zero valid crops with full rejection details.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

TRACKS_PATH  = REPO_ROOT / "dataset" / "tracks_C01.json"
REPORT_PATH  = REPO_ROOT / "dataset" / "crop_quality_report_phase3_final.json"
META_PATH    = REPO_ROOT / "dataset" / "crops_metadata_phase3_final.json"

CROPS_PER_TRACK = 5   # K — matches config.yaml

def load_json(p: Path) -> object:
    with open(p) as f:
        return json.load(f)

def main() -> None:
    # ------------------------------------------------------------------ #
    # 1. Load all data
    # ------------------------------------------------------------------ #
    all_records = load_json(TRACKS_PATH)
    quality_report = load_json(REPORT_PATH)
    metadata = load_json(META_PATH)

    # ------------------------------------------------------------------ #
    # 2. Unique track IDs from tracks file (ground truth)
    # ------------------------------------------------------------------ #
    all_track_ids = sorted({r["track_id"] for r in all_records})
    n_unique_tracks = len(all_track_ids)

    # ------------------------------------------------------------------ #
    # 3. Per-track sampled frame count
    #    sampling.py picks up to K evenly-spaced frames per track.
    #    We reconstruct from the quality report + metadata combined.
    # ------------------------------------------------------------------ #
    # Sampled = accepted + rejected (anything that entered quality eval)
    # We count per-track candidates from the report summary + metadata.

    # Accepted per track (from metadata)
    accepted_per_track: dict[int, int] = defaultdict(int)
    for rec in metadata:
        accepted_per_track[rec["track_id"]] += 1

    # Rejected per track + details (from quality report)
    rejected_per_track: dict[int, list[dict]] = defaultdict(list)
    for rej in quality_report["rejections"]:
        rejected_per_track[rej["track_id"]].append(rej)

    # ------------------------------------------------------------------ #
    # 4. Build per-track table
    # ------------------------------------------------------------------ #
    W_ID  = 10
    W_SAM = 9
    W_ACC = 10
    W_REJ = 10
    W_STA = 18

    header = (
        f"{'track_id':>{W_ID}} | "
        f"{'sampled':>{W_SAM}} | "
        f"{'accepted':>{W_ACC}} | "
        f"{'rejected':>{W_REJ}} | "
        f"{'status':<{W_STA}}"
    )
    sep = "-" * len(header)

    print()
    print("=" * len(header))
    print("TRACE Phase 3.6 — Per-Track Sanity Check")
    print("=" * len(header))
    print(header)
    print(sep)

    zero_crop_tracks: list[dict] = []
    total_sampled  = 0
    total_accepted = 0
    total_rejected = 0

    for tid in all_track_ids:
        acc = accepted_per_track.get(tid, 0)
        rej = len(rejected_per_track.get(tid, []))
        sam = acc + rej
        total_sampled  += sam
        total_accepted += acc
        total_rejected += rej

        if acc == 0:
            status = "ZERO VALID CROPS"
        else:
            status = "OK"

        print(
            f"{tid:>{W_ID}} | "
            f"{sam:>{W_SAM}} | "
            f"{acc:>{W_ACC}} | "
            f"{rej:>{W_REJ}} | "
            f"{status:<{W_STA}}"
        )

        if acc == 0:
            zero_crop_tracks.append({
                "track_id": tid,
                "sampled":  sam,
                "accepted": acc,
                "rejected": rej,
                "rejections": rejected_per_track.get(tid, []),
            })

    print(sep)
    print(
        f"{'TOTAL':>{W_ID}} | "
        f"{total_sampled:>{W_SAM}} | "
        f"{total_accepted:>{W_ACC}} | "
        f"{total_rejected:>{W_REJ}} | "
        f"{'':18}"
    )
    print("=" * len(header))

    # ------------------------------------------------------------------ #
    # 5. Summary
    # ------------------------------------------------------------------ #
    print()
    print(f"Unique track IDs      : {n_unique_tracks}")
    print(f"Total sampled         : {total_sampled}")
    print(f"Total accepted        : {total_accepted}")
    print(f"Total rejected        : {total_rejected}")
    print(f"Tracks with 0 crops   : {len(zero_crop_tracks)}")
    print()

    # ------------------------------------------------------------------ #
    # 6. Zero-crop track details
    # ------------------------------------------------------------------ #
    if zero_crop_tracks:
        print("=" * len(header))
        print("Tracks with ZERO valid crops — full rejection details")
        print("=" * len(header))
        for entry in zero_crop_tracks:
            tid = entry["track_id"]
            print(f"\nTrack {tid}:")
            print(f"  Sampled candidates : {entry['sampled']}")
            print(f"  Accepted           : {entry['accepted']}")
            print(f"  Rejected           : {entry['rejected']}")
            for rej in entry["rejections"]:
                print(
                    f"    frame={rej['frame']:>5}  "
                    f"reason={rej['reason']}  "
                    f"details={rej['details']}"
                )
    else:
        print("No tracks with zero valid crops — every track produced at least one valid crop.")

    print()

    # ------------------------------------------------------------------ #
    # 7. Verify totals match quality report summary
    # ------------------------------------------------------------------ #
    qsummary = quality_report["summary"]
    ok = True
    if qsummary["selected"] != total_sampled:
        print(f"[WARNING] Report selected={qsummary['selected']} vs table total_sampled={total_sampled}")
        ok = False
    if qsummary["accepted"] != total_accepted:
        print(f"[WARNING] Report accepted={qsummary['accepted']} vs table total_accepted={total_accepted}")
        ok = False
    if qsummary["rejected"] != total_rejected:
        print(f"[WARNING] Report rejected={qsummary['rejected']} vs table total_rejected={total_rejected}")
        ok = False
    if ok:
        print("[OK] Per-track totals match quality report summary exactly.")
    print()


if __name__ == "__main__":
    main()
