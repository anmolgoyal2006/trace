"""
generate_pdf.py — Generates Trace_SE_Lab_Assignment_2.pdf
from SRS.md and rendered diagram PNGs using xhtml2pdf.
"""

import markdown
import os
import base64
from pathlib import Path
from xhtml2pdf import pisa

DOCS_DIR = Path(__file__).parent
RENDERED_DIR = DOCS_DIR / "rendered"
OUTPUT = DOCS_DIR.parent.parent / "Trace_SE_Lab_Assignment_2.pdf"


def read_file(name: str) -> str:
    p = DOCS_DIR / name
    return p.read_text(encoding="utf-8") if p.exists() else f"[File not found: {name}]"


def img_data_uri(name: str) -> str:
    p = RENDERED_DIR / name
    if not p.exists():
        return ""
    data = base64.b64encode(p.read_bytes()).decode()
    return f"data:image/png;base64,{data}"


srs_md = read_file("SRS.md")
traceability_md = read_file("traceability.md")

srs_html = markdown.markdown(srs_md, extensions=['tables', 'fenced_code', 'toc'])
traceability_html = markdown.markdown(traceability_md, extensions=['tables', 'fenced_code'])

# Load diagram images as base64
dfd0_img = img_data_uri("DFD_Level_0.png")
dfd1_img = img_data_uri("DFD_Level_1.png")
dfd2_video_img = img_data_uri("DFD_Level_2_Video.png")
dfd2_match_img = img_data_uri("DFD_Level_2_Matching.png")
dfd2_route_img = img_data_uri("DFD_Level_2_Route.png")
uc_img = img_data_uri("UML_Use_Case.png")
act_img = img_data_uri("UML_Activity.png")
seq_img = img_data_uri("UML_Sequence.png")
comp_img = img_data_uri("UML_Component.png")
cls_img = img_data_uri("UML_Class.png")

html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
    @page {{
        size: A4;
        margin: 1.8cm 1.5cm;
        @frame footer {{
            -pdf-frame-content: footerContent;
            bottom: 0.5cm;
            margin-left: 1.5cm;
            margin-right: 1.5cm;
            height: 1cm;
        }}
    }}

    body {{
        font-family: Helvetica, Arial, sans-serif;
        font-size: 10pt;
        line-height: 1.5;
        color: #1e293b;
    }}

    h1 {{
        font-size: 18pt;
        color: #0f172a;
        border-bottom: 2px solid #3b82f6;
        padding-bottom: 6px;
        margin-top: 28px;
        page-break-after: avoid;
    }}

    h2 {{
        font-size: 14pt;
        color: #1e40af;
        border-bottom: 1px solid #93c5fd;
        padding-bottom: 4px;
        margin-top: 22px;
        page-break-after: avoid;
    }}

    h3 {{
        font-size: 12pt;
        color: #1e3a5f;
        margin-top: 16px;
        page-break-after: avoid;
    }}

    h4 {{
        font-size: 11pt;
        color: #334155;
        margin-top: 12px;
    }}

    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 10px 0;
        font-size: 8pt;
    }}

    th {{
        background-color: #1e40af;
        color: white;
        padding: 5px 6px;
        text-align: left;
        font-weight: bold;
    }}

    td {{
        padding: 4px 6px;
        border: 1px solid #cbd5e1;
    }}

    tr:nth-child(even) td {{
        background-color: #f1f5f9;
    }}

    code {{
        font-family: Courier, monospace;
        background-color: #f1f5f9;
        padding: 1px 3px;
        font-size: 8pt;
    }}

    pre {{
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 8px;
        font-size: 7pt;
        line-height: 1.3;
    }}

    .cover-page {{
        text-align: center;
        padding-top: 100px;
        page-break-after: always;
    }}

    .figure-container {{
        text-align: center;
        margin: 15px 0;
        page-break-inside: avoid;
    }}

    .figure-container img {{
        max-width: 100%;
    }}

    .figure-caption {{
        font-size: 9pt;
        color: #1e40af;
        font-weight: bold;
        margin-bottom: 6px;
    }}

    .figure-explanation {{
        font-size: 8.5pt;
        color: #334155;
        text-align: left;
        margin-top: 8px;
        padding: 0 10px;
    }}

    .section-break {{
        page-break-before: always;
    }}

    .toc-item {{
        margin: 4px 0;
        font-size: 11pt;
    }}

    .toc-sub {{
        margin-left: 20px;
        font-size: 10pt;
    }}

    hr {{
        border: none;
        border-top: 1px solid #e2e8f0;
        margin: 15px 0;
    }}

    ul, ol {{
        margin: 6px 0;
        padding-left: 22px;
    }}

    li {{
        margin: 2px 0;
    }}

    blockquote {{
        border-left: 3px solid #3b82f6;
        margin: 8px 0;
        padding: 4px 12px;
        background-color: #f0f9ff;
    }}
</style>
</head>
<body>

<!-- COVER PAGE -->
<div class="cover-page">
    <h1 style="font-size: 26pt; border: none; color: #0f172a;">TRACE</h1>
    <h2 style="font-size: 15pt; border: none; color: #475569;">AI-Powered Multi-Camera Person Tracking &amp; Re-Identification</h2>
    <hr/>
    <p style="font-size: 13pt; margin-top: 35px;">Software Engineering Laboratory</p>
    <p style="font-size: 13pt;">Assignment 2</p>
    <hr/>
    <p style="margin-top: 50px; font-size: 11pt;">SRS Document | DFD Diagrams | Class Diagram | UML Diagrams</p>
    <p style="margin-top: 35px; font-size: 10pt; color: #64748b;">
        Student Name: _________________________<br/><br/>
        Roll Number: _________________________<br/><br/>
        Date: 2 October 2026
    </p>
</div>

<!-- TABLE OF CONTENTS -->
<h1>Table of Contents</h1>
<div class="toc-item">1. Software Requirements Specification (SRS)</div>
<div class="toc-sub">1.1 Introduction | 1.2 Overall Description | 1.3 Functional Requirements</div>
<div class="toc-sub">1.4 Non-Functional Requirements | 1.5 External Interface Requirements</div>
<div class="toc-sub">1.6 Data Requirements | 1.7 System Constraints | 1.8 Assumptions | 1.9 Future Scope</div>
<div class="toc-item">2. DFD Diagrams</div>
<div class="toc-sub">2.1 Level 0 (Context) | 2.2 Level 1 | 2.3 Level 2: Video Processing</div>
<div class="toc-sub">2.4 Level 2: Matching | 2.5 Level 2: Route Reconstruction</div>
<div class="toc-item">3. UML Diagrams</div>
<div class="toc-sub">3.1 Use Case | 3.2 Activity | 3.3 Sequence | 3.4 Component</div>
<div class="toc-item">4. Class Diagram</div>
<div class="toc-item">5. Traceability Matrix</div>
<div class="toc-item">6. References</div>

<!-- SRS -->
<div class="section-break">
{srs_html}
</div>

<!-- DFD DIAGRAMS -->
<div class="section-break">
<h1>2. Data Flow Diagrams (DFD)</h1>
<p>These DFD diagrams model data flow through the Trace system using standard notation: rectangles for external entities, circles for processes, open-ended rectangles for data stores, and arrows for data flows. All diagrams are derived from the actual implemented architecture.</p>

<h2>2.1 DFD Level 0 - Context Diagram</h2>
<div class="figure-container">
    <div class="figure-caption">Figure 1: DFD Level 0 - Context Diagram</div>
    <img src="{dfd0_img}" width="520"/>
    <div class="figure-explanation">
    The context diagram shows Trace as a single process (Process 0) interacting with two external entities: the Operator (who submits search queries, reference photos, and camera configuration) and CCTV Camera Video Files (uploaded for processing). Outputs include reconstructed routes with timestamps, watchlist alerts, and dashboard analytics.
    </div>
</div>

<h2>2.2 DFD Level 1 - System Decomposition</h2>
<div class="figure-container">
    <div class="figure-caption">Figure 2: DFD Level 1 - Major Process Decomposition</div>
    <img src="{dfd1_img}" width="540"/>
    <div class="figure-explanation">
    Level 1 decomposes Trace into seven processes: (1) Video Upload and Validation, (2) Detection and Tracking, (3) Crop Extraction and Embedding, (4) Query Submission and Matching, (5) Route Reconstruction, (6) Persistence and Alerts, and (7) Dashboard and Result Delivery. Data stores: D1 (raw videos), D2 (tracks JSON), D4 (embedding galleries), D5 (SQLite DB), D6 (camera graph).
    </div>
</div>

<h2>2.3 DFD Level 2 - Video Processing Pipeline</h2>
<div class="figure-container">
    <div class="figure-caption">Figure 3: DFD Level 2 - Video Processing (Processes 1.0-3.0)</div>
    <img src="{dfd2_video_img}" width="540"/>
    <div class="figure-explanation">
    Expands the video processing pipeline. Process 1.1 validates camera_id (C01/C02/C03) and format (MP4/AVI/MOV/MKV). Process 2.1 runs YOLOv8 detection; 2.2 runs BoT-SORT tracking. Process 3.1 samples frames per track; 3.2 applies quality filtering (min 40x100px, edge truncation); 3.3 extracts crop JPEGs; 3.4 generates OSNet embeddings (always); 3.5/3.6 generate face/KPR embeddings (optional, weight-dependent).
    </div>
</div>

<h2>2.4 DFD Level 2 - Person Re-ID / Matching</h2>
<div class="figure-container">
    <div class="figure-caption">Figure 4: DFD Level 2 - Query and Matching (Process 4.0)</div>
    <img src="{dfd2_match_img}" width="540"/>
    <div class="figure-explanation">
    Expands the matching process. The query image is auto-cropped to the largest person (4.1), body embedding via OSNet (4.2), optional face (4.2F) and KPR (4.2K) embeddings. Galleries are loaded per camera (4.3), cosine similarity computed per crop (4.4), triple fusion scoring applied (4.5), and threshold decision at 0.74 filters tracks (4.6). Tracks below threshold produce "no confident match."
    </div>
</div>

<h2>2.5 DFD Level 2 - Route Reconstruction</h2>
<div class="figure-container">
    <div class="figure-caption">Figure 5: DFD Level 2 - Route Reconstruction (Process 5.0)</div>
    <img src="{dfd2_route_img}" width="540"/>
    <div class="figure-explanation">
    Route reconstruction builds camera adjacency from camera_graph.json (5.1), selects anchor camera (5.2), computes spatial score (1-hop=1.0, 2-hop=0.4, else 0.0) (5.3), temporal score via Gaussian centred on expected transit time (5.4), and fuses: 0.60 x appearance + 0.25 x spatial + 0.15 x temporal (5.5). Greedy extension iteratively picks the best camera-track pair (5.6). Route confidence is the mean fusion score (5.7).
    </div>
</div>
</div>

<!-- UML DIAGRAMS -->
<div class="section-break">
<h1>3. UML Diagrams</h1>

<h2>3.1 Use Case Diagram</h2>
<div class="figure-container">
    <div class="figure-caption">Figure 6: UML Use Case Diagram</div>
    <img src="{uc_img}" width="540"/>
    <div class="figure-explanation">
    One primary actor: the Operator. Use cases span five areas: Video Processing (upload, validate, poll), Person Management (register, enroll, update, delete, search by image), Re-ID Query (submit, progress, route, crops), Monitoring (dashboard, alerts, acknowledge, sightings), and Configuration (camera settings). No admin role exists in the implementation - all operations are available to the single Operator actor.
    </div>
</div>

<h2>3.2 Activity Diagram</h2>
<div class="figure-container">
    <div class="figure-caption">Figure 7: UML Activity Diagram - Person Search and Route Reconstruction</div>
    <img src="{act_img}" width="420"/>
    <div class="figure-explanation">
    Shows the complete search workflow. Decision points: input type (image vs. registered person), optional signal availability (face, KPR), the 0.74 threshold check, and whether any camera produces candidates. The no-confident-match branch is explicitly modelled. Route reconstruction uses greedy extension with spatial-temporal fusion. Watchlist alert generation is conditional on the person's watchlist_status.
    </div>
</div>

<h2>3.3 Sequence Diagram</h2>
<div class="figure-container">
    <div class="figure-caption">Figure 8: UML Sequence Diagram - Search Person Across Cameras</div>
    <img src="{seq_img}" width="540"/>
    <div class="figure-explanation">
    Traces the search workflow through actual components: Operator, Frontend (app.js), FastAPI (queries.py), PipelineService, MatchingService, RouteService, SQLite DB, and WebSocketManager. Key interactions: 202 async response, background task execution, per-camera matching loop with triple fusion, greedy route reconstruction, DB persistence of sightings and route steps, and WebSocket broadcast of route_complete event.
    </div>
</div>

<h2>3.4 Component Diagram</h2>
<div class="figure-container">
    <div class="figure-caption">Figure 9: UML Component Diagram - System Architecture</div>
    <img src="{comp_img}" width="540"/>
    <div class="figure-explanation">
    Shows physical architecture: Frontend (vanilla JS SPA), Backend API (FastAPI with 5 routers + WebSocket), Service Layer (Pipeline, Matching, Route, Embedding, Detection, Tracker, Crop), AI Pipeline scripts, Data Storage (SQLite, JSON galleries, camera graph, crops/videos), and Model Weights (YOLOv8, OSNet, SCRFD/ArcFace ONNX, KPR optional).
    </div>
</div>
</div>

<!-- CLASS DIAGRAM -->
<div class="section-break">
<h1>4. Class Diagram</h1>
<div class="figure-container">
    <div class="figure-caption">Figure 10: UML Class Diagram - Backend and AI Pipeline</div>
    <img src="{cls_img}" width="540"/>
    <div class="figure-explanation">
    Database entities (Person, Camera, QuerySession, Sighting, RouteStep, Alert) are SQLAlchemy ORM models with PK/FK relationships. Service classes implement business logic: PipelineService orchestrates the query session; MatchingService performs triple-fusion similarity search; RouteService does spatial-temporal route reconstruction; EmbeddingService wraps OSNet inference; WebSocketManager broadcasts events. Data classes CameraMatch and FusedSighting carry intermediate results. Settings centralises configuration. Key relationships: QuerySession composes Sightings and RouteSteps (cascade delete); Sighting references Camera; Person optionally links to QuerySession and Alert.
    </div>
</div>
</div>

<!-- TRACEABILITY -->
<div class="section-break">
<h1>5. Traceability and Consistency Table</h1>
{traceability_html}
</div>

<!-- REFERENCES -->
<div class="section-break">
<h1>6. References</h1>
<ol>
    <li>Trace Repository Source Code: backend/app/, ai_pipeline/, frontend/</li>
    <li>README.md - Project documentation and tech stack</li>
    <li>ai_pipeline/config.yaml - AI pipeline configuration</li>
    <li>dataset/camera_graph.json - Camera topology and transit times</li>
    <li>backend/app/config.py - Application settings (Settings class)</li>
    <li>backend/app/models/orm.py - SQLAlchemy ORM definitions</li>
    <li>backend/app/models/schemas.py - Pydantic request/response schemas</li>
    <li>YOLOv8 (Ultralytics) - Person detection</li>
    <li>BoT-SORT - Multi-object tracking (via Ultralytics)</li>
    <li>OSNet x1_0 (torchreid) - Body Re-ID embedding (512-dim)</li>
    <li>SCRFD 10G + ArcFace ResNet-50 - Face detection/recognition (ONNX)</li>
    <li>KPR Swin-Small (ECCV 2024) - Part-based Re-ID</li>
    <li>SOLIDER Swin-Small - Cloth-change robust embedding (768-dim)</li>
</ol>
</div>

<!-- FOOTER -->
<div id="footerContent">
    <p style="text-align: center; font-size: 8pt; color: #94a3b8;">
        Trace - SE Lab Assignment 2 | Page <pdf:pagenumber/>
    </p>
</div>

</body>
</html>"""

output_path = str(OUTPUT)
with open(output_path, "wb") as f:
    status = pisa.CreatePDF(html, dest=f)

if status.err:
    print(f"ERROR: PDF generation failed with {status.err} errors")
else:
    size_kb = os.path.getsize(output_path) / 1024
    print(f"PDF generated successfully: {output_path}")
    print(f"Size: {size_kb:.1f} KB")