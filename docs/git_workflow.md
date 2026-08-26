# Git Workflow & Team Task Split

> Read this before you push your first commit.

---

## 1. Branch Strategy

We use a lightweight **feature-branch workflow** — no one pushes directly to `main`.

```
main                  ← stable, demo-ready only
└── dev               ← integration branch; all features merge here first
    ├── feature/detection
    ├── feature/tracking
    ├── feature/reid
    ├── feature/matching
    ├── feature/route-reconstruction
    ├── feature/backend-api
    ├── feature/database-schema
    └── feature/frontend-dashboard
```

### Rules

| Rule | Detail |
|---|---|
| Never push to `main` directly | Only `dev` gets merged into `main` — and only when the feature is demo-ready. |
| Branch off `dev`, not `main` | `git checkout dev && git pull && git checkout -b feature/your-feature` |
| Keep branches short-lived | A branch should cover one coherent chunk of work, not weeks of commits. |
| PR to `dev` requires one review | At minimum, one other team member reads the diff before merge. |
| Merge `dev` → `main` as a group | Do this together at agreed checkpoints (end of each phase), not solo. |
| Delete merged branches | Keep the branch list clean. GitHub does this automatically if you enable "delete branch on merge". |

### Branch Naming

```
feature/<short-name>      # new functionality
fix/<short-name>          # bug fix
chore/<short-name>        # config, docs, tooling (no code logic change)
experiment/<short-name>   # notebook / exploratory work — never merges to dev
```

---

## 2. Commit Message Convention

Use the format: `<type>: <short description>`

```
feat: add ByteTrack integration for within-camera tracking
fix: correct embedding normalization in reid module
chore: update .gitignore for raw video files
docs: add filming plan checklist
refactor: extract fusion scorer into its own class
test: add unit tests for route reconstruction graph traversal
```

Keep the subject line under 72 characters. Add a body if the why isn't obvious.

---

## 3. Ownership Map

One person owns each area — meaning they're the primary author and the reviewer for PRs touching that area. Everyone can contribute everywhere, but the owner has final say and merges.

| Area | Branch(es) | Owner | Notes |
|---|---|---|---|
| **AI Pipeline — Detection & Tracking** | `feature/detection`, `feature/tracking` | Person A | YOLOv8 inference, ByteTrack integration, per-camera sighting extraction |
| **AI Pipeline — Re-ID & Matching** | `feature/reid`, `feature/matching` | Person B | Embedding extraction, cross-camera fusion scorer, route reconstruction |
| **Backend API & Database** | `feature/backend-api`, `feature/database-schema` | Person C | FastAPI routes, SQLAlchemy models, SQLite schema, query session management |
| **Frontend Dashboard** | `feature/frontend-dashboard` | Person A or C (TBD) | React map view, timeline view, query form |
| **Dataset & Evaluation** | `chore/dataset-*` | Person B | `camera_graph.json`, `metadata.csv`, ground-truth logging, blind test |
| **Docs & Tooling** | `chore/docs`, `chore/ci` | Rotating | Whoever touches it owns the PR |

> Fill in real names above before the first sprint starts.

---

## 4. Raw Video Storage — Google Drive, Not Git

**Do not commit video files to git.** GitHub has a 100 MB file size limit and a 1 GB soft cap per repo. A 10-minute 1080p clip is ~1.5 GB.

### Where videos live

```
Google Drive: Trace Project / dataset / raw_videos /
    clip_001_C01.mp4
    clip_002_C02.mp4
    ...
    blind_test/          ← access restricted to demo day only
```

### How to get the videos locally

1. Get access to the shared Drive folder (ask the project lead).
2. Download the clips you need into `dataset/raw_videos/` locally.
3. That folder is in `.gitignore` — git will never see these files.

### Naming convention

```
<clip_id>_<camera_id>.mp4
e.g., clip_001_C01.mp4, clip_007_C02.mp4
```

Match the `clip_id` exactly to the row in `dataset/metadata.csv`.

> **Drive folder link:** _(add link here once the folder is created)_

---

## 5. What Gets Committed vs What Doesn't

| Committed to git | Not committed (Drive or regenerated) |
|---|---|
| All source code (`ai_pipeline/`, `backend/`, `frontend/`) | Raw video files (`*.mp4`, `*.avi`, etc.) |
| Config files (`config.yaml`, `camera_graph.json`) | SQLite database (`*.db`) |
| `metadata.csv`, `ground_truth.json` (text, small) | Model weights (`*.pt`, `*.pth`, `*.onnx`) |
| `requirements.txt` files | Virtual environment (`venv/`, `.venv/`) |
| Docs and planning docs | Processed embeddings / intermediate outputs |
| `.gitignore` | Notebook checkpoints (`.ipynb_checkpoints/`) |

---

## 6. First Commit Checklist

Before the first real feature work starts, make sure this baseline is in place:

- [ ] Repo cloned by all team members.
- [ ] `dev` branch created and pushed: `git checkout -b dev && git push -u origin dev`.
- [ ] Each person creates their first feature branch off `dev`.
- [ ] Google Drive folder created and link added to this doc (section 4 above).
- [ ] Everyone has confirmed they can see `.gitignore` is working: `git status` should not show any `.mp4` or `.db` files.
- [ ] Branch protection on `main` enabled on GitHub (Settings → Branches → require PR + 1 review).
- [ ] Names filled into the ownership table in section 3.

---

## 7. Resolving Merge Conflicts

The most likely conflict zones are `config.yaml` and `requirements.txt`. When a conflict happens:

1. Don't resolve it alone if it touches someone else's area — ping the owner first.
2. For `requirements.txt`: keep all dependencies, don't delete someone else's package.
3. For `config.yaml`: talk through the conflicting values; don't just pick yours.
4. Mark the resolution clearly in the commit message: `fix: resolve merge conflict in config.yaml — keep reid batch_size=32`.

---

## 8. Phase Checkpoints (Planned Merge Gates)

| Phase | What merges to `main` | Target |
|---|---|---|
| Phase 1–2 | Detection + Tracking pipeline runnable end-to-end on one clip | Week 2 |
| Phase 3–4 | Re-ID + Matching producing cross-camera sightings | Week 4 |
| Phase 5–6 | Backend API + DB storing and serving route results | Week 6 |
| Phase 7 | Frontend dashboard rendering map + timeline | Week 7 |
| Phase 8 | Blind test integration, evaluation scripts | Week 8 (demo) |

Each checkpoint = a PR from `dev` → `main`, reviewed together as a team.
