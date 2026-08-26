# Filming Plan — Trace Dataset

> **Film during Phase 7.** This document is written now so nothing gets forgotten on the day.
> Complete every checklist item before marking a scenario as done.

---

## Camera Setup

| Camera | Location | Mount | Approx. Resolution | FPS |
|---|---|---|---|---|
| C01 | Main Entrance | Fixed tripod / ceiling bracket | 1080p | 25–30 |
| C02 | Main Corridor | Fixed tripod / ceiling bracket | 1080p | 25–30 |
| C03 | Canteen / Common Area | Fixed tripod / ceiling bracket | 1080p | 25–30 |

**Pre-filming checklist (do once per session):**
- [ ] All three cameras are positioned and fixed — no moving them mid-session.
- [ ] All cameras are time-synced to the same clock (or a sync event is filmed — e.g., clap in front of each camera at session start).
- [ ] A `metadata.csv` row is created for each clip before filming starts (camera ID, start time, scenario tag).
- [ ] Camera angles and heights match `camera_graph.json` descriptions.
- [ ] Background is as empty as possible before adding subjects (record 30 s of empty scene for background subtraction baselines).

---

## Subjects

- **Minimum 3 subjects** for MVP (enough to test cross-camera confusion between similar-looking people).
- Each subject gets an assigned **subject ID** (S01, S02, S03 …) logged in `metadata.csv`.
- Ground-truth route for each subject is manually recorded on the day (which camera, entry/exit time).

---

## Scenario Checklist

Each scenario is a separate recording session. Film each scenario at least once; film starred (★) scenarios twice for redundancy.

### Scenario 1 — Baseline (Clean) ★
- [ ] Single subject, no occlusion, clear lighting, no clothing changes.
- [ ] Subject walks C01 → C02 → C03 at normal pace.
- [ ] Record ground-truth timestamps at each camera transition.
- **Purpose:** Sanity-check that the pipeline works at all before adding difficulty.

### Scenario 2 — Multiple Subjects (Crowd Baseline) ★
- [ ] 3 subjects walking simultaneously through all three cameras.
- [ ] Subjects should sometimes be in the same frame at the same time.
- [ ] Vary walking pace (slow, normal, fast).
- **Purpose:** Test cross-camera matching under identity confusion pressure.

### Scenario 3 — Clothing Change
- [ ] One subject changes an outer layer (e.g., removes jacket) between C01 and C02.
- [ ] Other subjects remain unchanged as distractors.
- **Purpose:** Stress-test appearance-based re-ID — the key hard case for the system.

### Scenario 4 — Occlusion
- [ ] One subject is briefly occluded mid-corridor (e.g., by another person, a pillar, or a bag).
- [ ] Occlusion should last at least 3–5 seconds.
- **Purpose:** Test ByteTrack's ability to recover track ID after occlusion within a camera.

### Scenario 5 — Lighting Variation
- [ ] Film the same C01 → C02 → C03 route at two different times of day if possible (morning vs afternoon), or simulate by adjusting overhead lights.
- [ ] At minimum: one clip with bright even lighting, one with uneven / dimmer lighting.
- **Purpose:** Re-ID embeddings are sensitive to illumination — this checks robustness.

### Scenario 6 — Backtrack / Non-Linear Route
- [ ] Subject walks C01 → C02, then turns back to C01, then returns to C02 → C03.
- **Purpose:** Verify the route reconstruction handles backtracking and non-linear paths correctly.

### Scenario 7 — Blind Test (Hold-Out) ★
- [ ] Film on a **different day** from all other scenarios.
- [ ] **Do not use this footage during development.** Only use it for the final demo evaluation.
- [ ] Include at least one subject from Scenarios 1–6 and one new subject.
- [ ] Mix of scenario types (some clean, some occlusion, one clothing change).
- [ ] Record full ground-truth route for all subjects — this is the evaluation ground truth.
- **Purpose:** Unbiased final demo evaluation. If the system was overfit to development footage, it will fail here.

---

## Metadata CSV Schema

File: `dataset/metadata.csv`

```
clip_id, camera_id, subject_id, scenario_tag, start_time_iso, end_time_iso, notes
```

Example rows:
```
clip_001, C01, S01, baseline, 2025-09-01T10:00:00, 2025-09-01T10:10:00, clean walk no occlusion
clip_002, C02, S01, baseline, 2025-09-01T10:00:30, 2025-09-01T10:10:30, same session as clip_001
clip_007, C01, S02, clothing_change, 2025-09-02T14:00:00, 2025-09-02T14:10:00, removed jacket before C02
```

---

## Ground-Truth Route Format

For each subject in each scenario, record:

```
subject_id, scenario_tag, ground_truth_route
S01, baseline, [{"camera": "C01", "entry": "10:00:05", "exit": "10:00:52"}, {"camera": "C02", ...}, ...]
```

Store this in `dataset/ground_truth.json` after filming. This is what the final demo evaluation compares against.

---

## Day-of Checklist

- [ ] All cameras set up and time-synced.
- [ ] Subjects briefed on each scenario (no improv — follow the script).
- [ ] Someone is designated to log ground-truth timestamps in real time.
- [ ] `metadata.csv` rows filled in for each clip as it's filmed.
- [ ] All clips transferred to `dataset/raw_videos/` and named `<clip_id>_<camera_id>.mp4`.
- [ ] Blind test footage stored separately and not opened until demo day.
- [ ] Quick sanity check: play back one clip per camera to confirm correct framing and no corruption.

---

## What Not to Film (Avoid)

- Faces as the primary identifying feature — Trace relies on body appearance, not face recognition. Faces can be present but should not be the only distinguishing feature.
- Scenarios where the same person never appears in more than one camera — useless for cross-camera matching testing.
- Clips shorter than 5 minutes — not enough data for realistic tracking.
