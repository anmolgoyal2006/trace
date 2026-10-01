"""
render_diagrams.py — Renders all DFD and UML diagrams as PNG images
using matplotlib. These will be embedded into the final PDF.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse, Rectangle
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "rendered"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Style constants ─────────────────────────────────────────────────────────
BG_COLOR = 'white'
ENTITY_COLOR = '#dbeafe'
PROCESS_COLOR = '#fef3c7'
STORE_COLOR = '#d1fae5'
FLOW_COLOR = '#334155'
FONT_SIZE = 9
TITLE_SIZE = 14


def draw_entity(ax, x, y, label, w=1.8, h=0.7):
    """External entity — rectangle"""
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.05",
                          facecolor=ENTITY_COLOR, edgecolor='#1e40af', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x, y, label, ha='center', va='center', fontsize=FONT_SIZE,
            fontweight='bold', color='#1e3a5f')


def draw_process(ax, x, y, label, r=0.55):
    """Process — circle/rounded"""
    circle = Circle((x, y), r, facecolor=PROCESS_COLOR,
                    edgecolor='#92400e', linewidth=1.5)
    ax.add_patch(circle)
    ax.text(x, y, label, ha='center', va='center', fontsize=FONT_SIZE-1,
            fontweight='bold', color='#78350f', wrap=True)


def draw_store(ax, x, y, label, w=2.0, h=0.6):
    """Data store — open-ended rectangle"""
    rect = Rectangle((x - w/2, y - h/2), w, h,
                     facecolor=STORE_COLOR, edgecolor='#065f46', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x, y, label, ha='center', va='center', fontsize=FONT_SIZE-1,
            color='#064e3b', fontweight='bold')


def draw_arrow(ax, x1, y1, x2, y2, label='', color=FLOW_COLOR):
    """Data flow arrow"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my + 0.12, label, ha='center', va='bottom',
                fontsize=7, color='#475569', style='italic')


def new_figure(title, figsize=(11.7, 8.3)):
    """A4 landscape figure"""
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=150)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold',
                 color='#1e293b', pad=10)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    return fig, ax


def save_fig(fig, name):
    path = OUTPUT_DIR / name
    fig.savefig(str(path), bbox_inches='tight', facecolor='white',
                dpi=150, pad_inches=0.3)
    plt.close(fig)
    print(f"  Saved: {name}")


# ═══════════════════════════════════════════════════════════════════════════
# DFD LEVEL 0 — CONTEXT DIAGRAM
# ═══════════════════════════════════════════════════════════════════════════

def render_dfd_level_0():
    fig, ax = new_figure("DFD Level 0 — Context Diagram: Trace System")

    # External entities
    draw_entity(ax, 1.5, 5.5, "Operator", w=1.6, h=0.6)
    draw_entity(ax, 1.5, 1.5, "CCTV Camera\nVideo Files", w=1.8, h=0.7)

    # Central process
    draw_process(ax, 5.0, 3.5, "0\nTrace\nSystem", r=0.9)

    # Output entity (same operator receives outputs)
    draw_entity(ax, 8.5, 3.5, "Operator\n(Results)", w=1.8, h=0.7)

    # Flows in
    draw_arrow(ax, 2.3, 5.5, 4.1, 4.0, "Reference photo /\nPerson ID")
    draw_arrow(ax, 2.3, 5.2, 4.1, 3.7, "Search request")
    draw_arrow(ax, 2.4, 1.5, 4.1, 3.0, "Video files\n(MP4/AVI/MOV)")
    draw_arrow(ax, 2.3, 1.8, 4.1, 3.3, "Camera config")

    # Flows out
    draw_arrow(ax, 5.9, 3.8, 7.6, 3.7, "Route + timestamps")
    draw_arrow(ax, 5.9, 3.5, 7.6, 3.5, "Watchlist alerts")
    draw_arrow(ax, 5.9, 3.2, 7.6, 3.3, "Dashboard analytics")

    save_fig(fig, "DFD_Level_0.png")


# ═══════════════════════════════════════════════════════════════════════════
# DFD LEVEL 1
# ═══════════════════════════════════════════════════════════════════════════

def render_dfd_level_1():
    fig, ax = new_figure("DFD Level 1 — Trace System Decomposition", figsize=(11.7, 8.3))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)

    # External entity
    draw_entity(ax, 1.0, 7.5, "Operator", w=1.5, h=0.6)

    # Processes
    draw_process(ax, 3.0, 7.5, "1.0\nVideo Upload\n& Validation", r=0.7)
    draw_process(ax, 3.0, 5.0, "2.0\nDetection &\nTracking", r=0.7)
    draw_process(ax, 3.0, 2.5, "3.0\nCrop &\nEmbedding", r=0.7)
    draw_process(ax, 7.5, 7.5, "4.0\nQuery &\nMatching", r=0.7)
    draw_process(ax, 7.5, 5.0, "5.0\nRoute\nReconstruction", r=0.7)
    draw_process(ax, 7.5, 2.5, "6.0\nPersistence\n& Alerts", r=0.7)
    draw_process(ax, 10.5, 5.0, "7.0\nDashboard &\nDelivery", r=0.7)

    # Data stores
    draw_store(ax, 5.5, 8.5, "D1: Raw Videos", w=1.8, h=0.45)
    draw_store(ax, 5.5, 6.2, "D2: Tracks JSON", w=1.8, h=0.45)
    draw_store(ax, 5.5, 4.0, "D4: Galleries", w=1.8, h=0.45)
    draw_store(ax, 10.5, 8.5, "D5: SQLite DB", w=1.8, h=0.45)
    draw_store(ax, 10.5, 2.0, "D6: Camera Graph", w=1.9, h=0.45)

    # Flows
    draw_arrow(ax, 1.8, 7.5, 2.3, 7.5, "video + camera_id")
    draw_arrow(ax, 1.5, 7.2, 6.8, 7.5, "reference photo / person_id")

    draw_arrow(ax, 3.0, 6.8, 3.0, 5.7, "video path")
    draw_arrow(ax, 3.0, 4.3, 3.0, 3.2, "tracks")

    draw_arrow(ax, 3.7, 7.5, 4.6, 8.5, "")
    draw_arrow(ax, 3.7, 5.0, 4.6, 6.2, "track records")
    draw_arrow(ax, 3.7, 2.5, 4.6, 4.0, "embeddings")

    draw_arrow(ax, 6.4, 4.0, 6.8, 7.2, "galleries")
    draw_arrow(ax, 8.2, 7.5, 8.2, 5.7, "candidates")
    draw_arrow(ax, 8.2, 4.3, 8.2, 3.2, "fused route")

    draw_arrow(ax, 8.2, 7.5, 9.6, 8.5, "persist")
    draw_arrow(ax, 9.6, 8.5, 10.5, 5.7, "")
    draw_arrow(ax, 10.5, 5.7, 10.5, 5.7, "")

    draw_arrow(ax, 10.5, 2.0, 8.2, 4.3, "adjacency")
    draw_arrow(ax, 8.2, 5.0, 9.8, 5.0, "WS events")

    save_fig(fig, "DFD_Level_1.png")


# ═══════════════════════════════════════════════════════════════════════════
# DFD LEVEL 2 — VIDEO PROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def render_dfd_level_2_video():
    fig, ax = new_figure("DFD Level 2 — Video Processing Pipeline (Processes 1.0–3.0)", figsize=(11.7, 8.3))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)

    draw_entity(ax, 1.0, 7.0, "Operator", w=1.5, h=0.6)

    draw_process(ax, 3.0, 7.0, "1.1\nValidate\nCamera &\nFormat", r=0.6)
    draw_process(ax, 3.0, 5.0, "1.2\nSave Video\n& Create Job", r=0.6)
    draw_process(ax, 3.0, 3.0, "2.1\nYOLOv8\nDetection", r=0.6)
    draw_process(ax, 3.0, 1.2, "2.2\nBoT-SORT\nTracking", r=0.6)

    draw_process(ax, 6.5, 6.0, "3.1\nFrame\nSampling", r=0.6)
    draw_process(ax, 6.5, 4.0, "3.2\nQuality\nFilter", r=0.6)
    draw_process(ax, 6.5, 2.0, "3.3\nCrop\nExtraction", r=0.6)

    draw_process(ax, 9.5, 6.0, "3.4\nOSNet\nEmbedding", r=0.6)
    draw_process(ax, 9.5, 4.0, "3.5\nFace\nEmbedding\n(optional)", r=0.6)
    draw_process(ax, 9.5, 2.0, "3.6\nKPR\nEmbedding\n(optional)", r=0.6)

    draw_store(ax, 5.0, 7.5, "D1: Raw Videos", w=1.7, h=0.4)
    draw_store(ax, 5.0, 0.7, "D2: Tracks JSON", w=1.7, h=0.4)
    draw_store(ax, 8.0, 0.7, "D3: Crops", w=1.5, h=0.4)
    draw_store(ax, 11.0, 6.0, "D4: Body Gallery", w=1.8, h=0.4)
    draw_store(ax, 11.0, 4.0, "D4F: Face Gallery", w=1.8, h=0.4)
    draw_store(ax, 11.0, 2.0, "D4K: KPR Gallery", w=1.8, h=0.4)

    draw_arrow(ax, 1.8, 7.0, 2.4, 7.0, "video + cam_id")
    draw_arrow(ax, 3.0, 6.4, 3.0, 5.6, "valid")
    draw_arrow(ax, 3.0, 4.4, 3.0, 3.6, "")
    draw_arrow(ax, 3.0, 2.4, 3.0, 1.8, "detections")

    draw_arrow(ax, 3.6, 5.0, 4.1, 7.5, "")
    draw_arrow(ax, 3.6, 1.2, 4.1, 0.7, "tracks")

    draw_arrow(ax, 5.9, 0.7, 5.9, 5.4, "track boxes")
    draw_arrow(ax, 6.5, 5.4, 6.5, 4.6, "samples")
    draw_arrow(ax, 6.5, 3.4, 6.5, 2.6, "accepted")
    draw_arrow(ax, 7.1, 2.0, 7.2, 0.7, "crops")

    draw_arrow(ax, 7.1, 6.0, 8.9, 6.0, "metadata + crops")
    draw_arrow(ax, 7.1, 4.0, 8.9, 4.0, "")
    draw_arrow(ax, 7.1, 2.0, 8.9, 2.0, "")

    draw_arrow(ax, 10.1, 6.0, 10.1, 6.0, "")
    draw_arrow(ax, 10.1, 4.0, 10.1, 4.0, "")
    draw_arrow(ax, 10.1, 2.0, 10.1, 2.0, "")

    save_fig(fig, "DFD_Level_2_Video.png")


# ═══════════════════════════════════════════════════════════════════════════
# DFD LEVEL 2 — MATCHING
# ═══════════════════════════════════════════════════════════════════════════

def render_dfd_level_2_matching():
    fig, ax = new_figure("DFD Level 2 — Person Re-ID / Matching (Process 4.0)", figsize=(11.7, 8.3))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)

    draw_entity(ax, 1.0, 6.5, "Operator", w=1.5, h=0.6)

    draw_process(ax, 3.0, 6.5, "4.1\nQuery\nAuto-Crop", r=0.6)
    draw_process(ax, 3.0, 4.5, "4.2\nBody\nEmbedding\n(OSNet)", r=0.6)
    draw_process(ax, 3.0, 2.5, "4.2F\nFace\nEmbedding\n(optional)", r=0.6)
    draw_process(ax, 3.0, 0.8, "4.2K\nKPR\nEmbedding\n(optional)", r=0.6)

    draw_process(ax, 6.5, 5.5, "4.3\nLoad\nCamera\nGalleries", r=0.6)
    draw_process(ax, 6.5, 3.0, "4.4\nPer-Camera\nSimilarity\nSearch", r=0.6)
    draw_process(ax, 9.5, 4.5, "4.5\nTriple\nFusion\nScoring", r=0.6)
    draw_process(ax, 9.5, 2.0, "4.6\nThreshold\nDecision\n(≥ 0.74)", r=0.6)

    draw_store(ax, 5.5, 7.5, "D4: Body Gallery", w=1.8, h=0.4)
    draw_store(ax, 5.5, 1.0, "D4F: Face Gallery", w=1.8, h=0.4)
    draw_store(ax, 8.5, 7.0, "D4K: KPR Gallery", w=1.8, h=0.4)
    draw_store(ax, 1.0, 4.5, "D5: SQLite DB", w=1.6, h=0.4)

    draw_entity(ax, 11.0, 0.8, "Route\nReconstruction", w=1.8, h=0.6)

    draw_arrow(ax, 1.8, 6.5, 2.4, 6.5, "query image")
    draw_arrow(ax, 3.0, 5.9, 3.0, 5.1, "cropped img")
    draw_arrow(ax, 3.0, 3.9, 3.0, 3.1, "")
    draw_arrow(ax, 3.0, 1.9, 3.0, 1.4, "")

    draw_arrow(ax, 1.0, 4.3, 2.4, 4.5, "enrolled emb")
    draw_arrow(ax, 3.6, 4.5, 5.9, 3.2, "query vector")

    draw_arrow(ax, 5.5, 7.3, 5.9, 5.8, "body embeddings")
    draw_arrow(ax, 5.5, 1.2, 5.9, 2.7, "face embeddings")
    draw_arrow(ax, 8.5, 6.8, 7.1, 5.8, "KPR embeddings")

    draw_arrow(ax, 6.5, 4.9, 6.5, 3.6, "galleries")
    draw_arrow(ax, 7.1, 3.0, 8.9, 4.2, "crop sims")
    draw_arrow(ax, 9.5, 3.9, 9.5, 2.6, "fused scores")
    draw_arrow(ax, 9.5, 1.4, 10.1, 1.0, "qualified tracks")

    save_fig(fig, "DFD_Level_2_Matching.png")


# ═══════════════════════════════════════════════════════════════════════════
# DFD LEVEL 2 — ROUTE RECONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════

def render_dfd_level_2_route():
    fig, ax = new_figure("DFD Level 2 — Route Reconstruction (Process 5.0)", figsize=(11.7, 8.3))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)

    draw_process(ax, 2.0, 6.5, "5.1\nBuild\nAdjacency", r=0.6)
    draw_process(ax, 2.0, 4.5, "5.2\nAnchor\nCamera\nSelection", r=0.6)
    draw_process(ax, 5.0, 6.5, "5.3\nSpatial\nScore", r=0.6)
    draw_process(ax, 5.0, 4.5, "5.4\nTemporal\nScore", r=0.6)
    draw_process(ax, 5.0, 2.5, "5.5\nFusion\nScore", r=0.6)
    draw_process(ax, 8.0, 4.5, "5.6\nGreedy\nRoute\nExtension", r=0.6)
    draw_process(ax, 8.0, 2.0, "5.7\nRoute\nConfidence", r=0.6)

    draw_store(ax, 1.0, 7.8, "D6: Camera Graph", w=1.8, h=0.4)
    draw_store(ax, 10.5, 2.0, "D5: SQLite DB", w=1.6, h=0.4)
    draw_entity(ax, 10.5, 6.5, "WebSocket\nroute_complete", w=1.8, h=0.6)
    draw_entity(ax, 10.5, 4.5, "Ordered Route\n+ Timestamps", w=1.8, h=0.6)

    # Input
    ax.text(0.5, 3.5, "Per-camera\ncandidate\nmatches", ha='center', va='center',
            fontsize=8, color='#475569',
            bbox=dict(boxstyle='round', facecolor='#e2e8f0', edgecolor='#64748b'))

    draw_arrow(ax, 1.0, 7.6, 1.5, 7.0, "adjacency + transit")
    draw_arrow(ax, 1.2, 3.5, 1.5, 4.5, "candidates")

    draw_arrow(ax, 2.0, 5.9, 2.0, 5.1, "")
    draw_arrow(ax, 2.6, 6.5, 4.4, 6.5, "adjacency map")
    draw_arrow(ax, 2.6, 4.5, 4.4, 4.5, "timestamps")

    draw_arrow(ax, 5.0, 5.9, 5.0, 5.1, "spatial [0,1]")
    draw_arrow(ax, 5.0, 3.9, 5.0, 3.1, "temporal [0,1]")

    draw_arrow(ax, 5.6, 2.5, 7.4, 4.0, "fusion = 0.60×app + 0.25×spa + 0.15×tem")

    draw_arrow(ax, 8.0, 3.9, 8.0, 2.6, "ordered route")
    draw_arrow(ax, 8.6, 4.5, 9.6, 4.5, "route result")
    draw_arrow(ax, 8.6, 5.0, 9.6, 6.3, "WS event")
    draw_arrow(ax, 8.0, 1.4, 9.7, 2.0, "persist")

    save_fig(fig, "DFD_Level_2_Route.png")


# ═══════════════════════════════════════════════════════════════════════════
# UML USE CASE DIAGRAM
# ═══════════════════════════════════════════════════════════════════════════

def render_use_case():
    fig, ax = new_figure("UML Use Case Diagram — Trace System", figsize=(11.7, 8.3))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)

    # Actor
    ax.plot(1.0, 5.0, 'o', markersize=12, color='#1e40af')
    ax.plot([1.0, 1.0], [4.7, 4.0], '-', color='#1e40af', lw=2)
    ax.plot([0.6, 1.4], [4.4, 4.4], '-', color='#1e40af', lw=2)
    ax.plot([1.0, 0.7], [4.0, 3.4], '-', color='#1e40af', lw=2)
    ax.plot([1.0, 1.3], [4.0, 3.4], '-', color='#1e40af', lw=2)
    ax.text(1.0, 5.4, "Operator", ha='center', fontsize=FONT_SIZE, fontweight='bold')

    # System boundary
    rect = FancyBboxPatch((2.5, 0.3), 9.0, 8.4,
                          boxstyle="round,pad=0.1",
                          facecolor='#f8fafc', edgecolor='#334155', linewidth=1.5,
                          linestyle='--')
    ax.add_patch(rect)
    ax.text(7.0, 8.9, "Trace — Multi-Camera Person Tracking & Re-ID",
            ha='center', fontsize=11, fontweight='bold', color='#1e293b')

    # Use cases as ellipses
    use_cases = [
        (4.5, 8.0, "Upload CCTV Video"),
        (4.5, 7.0, "Register Person"),
        (4.5, 6.0, "Enroll Reference Photo"),
        (4.5, 5.0, "Submit Search Query"),
        (4.5, 4.0, "View Route Result"),
        (4.5, 3.0, "View Match Crops"),
        (4.5, 2.0, "View Dashboard Analytics"),
        (4.5, 1.0, "View Watchlist Alerts"),
        (8.5, 8.0, "Poll Upload Job Status"),
        (8.5, 7.0, "Configure Camera"),
        (8.5, 6.0, "Search Person by Image"),
        (8.5, 5.0, "View Query Progress"),
        (8.5, 4.0, "Acknowledge Alert"),
        (8.5, 3.0, "View Camera Sightings"),
    ]

    for x, y, label in use_cases:
        ellipse = Ellipse((x, y), 2.8, 0.7,
                         facecolor='#eff6ff', edgecolor='#3b82f6', linewidth=1.2)
        ax.add_patch(ellipse)
        ax.text(x, y, label, ha='center', va='center', fontsize=8, color='#1e3a5f')
        # Draw line from actor to use case
        ax.plot([1.4, x-1.4], [4.5, y], '-', color='#94a3b8', lw=0.5, alpha=0.6)

    save_fig(fig, "UML_Use_Case.png")


# ═══════════════════════════════════════════════════════════════════════════
# UML ACTIVITY DIAGRAM
# ═══════════════════════════════════════════════════════════════════════════

def render_activity():
    fig, ax = new_figure("UML Activity Diagram — Person Search & Route Reconstruction", figsize=(8.3, 11.7))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 14)

    def activity_box(x, y, text, w=3.5, h=0.55, color='#e0f2fe', edge='#0284c7'):
        rect = FancyBboxPatch((x-w/2, y-h/2), w, h,
                              boxstyle="round,pad=0.08",
                              facecolor=color, edgecolor=edge, linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=7.5, color='#1e293b')

    def diamond(x, y, text, s=0.4):
        d = plt.Polygon([(x, y+s), (x+s*1.5, y), (x, y-s), (x-s*1.5, y)],
                       facecolor='#fef3c7', edgecolor='#d97706', linewidth=1.2)
        ax.add_patch(d)
        ax.text(x, y, text, ha='center', va='center', fontsize=6.5, color='#78350f')

    def arrow_down(x, y1, y2):
        ax.annotate('', xy=(x, y2), xytext=(x, y1),
                   arrowprops=dict(arrowstyle='->', color='#475569', lw=1.2))

    # Start
    ax.plot(4, 13.5, 'o', markersize=8, color='#1e293b')

    arrow_down(4, 13.3, 12.9)
    activity_box(4, 12.6, "Submit reference image or select registered person")

    arrow_down(4, 12.3, 11.9)
    diamond(4, 11.6, "Input type?")

    # Left branch: reference image
    ax.text(2.0, 11.6, "image", fontsize=7, color='#475569')
    activity_box(2.0, 10.8, "Auto-crop (YOLOv8)\n+ OSNet embedding", w=2.8)

    # Right branch: registered person
    ax.text(6.0, 11.6, "person", fontsize=7, color='#475569')
    activity_box(6.0, 10.8, "Load enrolled\nembedding from DB", w=2.5)

    # Merge
    arrow_down(2.0, 10.5, 10.0)
    arrow_down(6.0, 10.5, 10.0)
    ax.plot([2.0, 6.0], [10.0, 10.0], '-', color='#475569', lw=1)
    arrow_down(4, 10.0, 9.6)

    activity_box(4, 9.3, "Optional: Face embedding (SCRFD + ArcFace)")
    arrow_down(4, 9.0, 8.6)
    activity_box(4, 8.3, "Optional: KPR part embedding (subprocess)")
    arrow_down(4, 8.0, 7.6)
    activity_box(4, 7.3, "Load camera galleries (C01, C02, C03)")
    arrow_down(4, 7.0, 6.6)
    activity_box(4, 6.3, "Per-camera cosine similarity + track aggregation")
    arrow_down(4, 6.0, 5.6)
    activity_box(4, 5.3, "Triple fusion scoring (body + face + KPR)")
    arrow_down(4, 5.0, 4.6)

    diamond(4, 4.3, "Score ≥ 0.74?")

    # Yes branch
    ax.text(5.5, 4.3, "yes", fontsize=7, color='#16a34a')
    arrow_down(4, 3.9, 3.5)
    activity_box(4, 3.2, "Route reconstruction (spatial + temporal fusion)")
    arrow_down(4, 2.9, 2.5)
    activity_box(4, 2.2, "Persist sightings + route steps to SQLite")
    arrow_down(4, 1.9, 1.5)
    activity_box(4, 1.2, "Push route_complete via WebSocket + display")

    # No branch
    ax.text(1.5, 4.3, "no", fontsize=7, color='#dc2626')
    ax.plot([2.5, 1.5], [4.3, 4.3], '-', color='#dc2626', lw=1)
    activity_box(1.5, 3.5, "No confident\nmatch found", w=2.0, h=0.5, color='#fee2e2', edge='#dc2626')

    # End
    arrow_down(4, 0.9, 0.5)
    circle = Circle((4, 0.3), 0.15, facecolor='#1e293b', edgecolor='#1e293b')
    ax.add_patch(circle)

    save_fig(fig, "UML_Activity.png")


# ═══════════════════════════════════════════════════════════════════════════
# UML SEQUENCE DIAGRAM
# ═══════════════════════════════════════════════════════════════════════════

def render_sequence():
    fig, ax = new_figure("UML Sequence Diagram — Search Person Across Cameras", figsize=(11.7, 8.3))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)

    # Lifelines
    actors = [
        (1.0, "Operator"),
        (2.8, "Frontend\n(app.js)"),
        (4.6, "FastAPI\n(queries.py)"),
        (6.4, "Pipeline\nService"),
        (8.0, "Matching\nService"),
        (9.5, "Route\nService"),
        (11.0, "SQLite DB"),
    ]

    top_y = 8.5
    bottom_y = 0.5

    for x, label in actors:
        # Lifeline
        ax.plot([x, x], [top_y - 0.5, bottom_y], '--', color='#94a3b8', lw=0.8)
        # Box
        rect = FancyBboxPatch((x-0.6, top_y-0.5), 1.2, 0.5,
                              boxstyle="round,pad=0.05",
                              facecolor='#e2e8f0', edgecolor='#475569', linewidth=1)
        ax.add_patch(rect)
        ax.text(x, top_y - 0.25, label, ha='center', va='center', fontsize=7,
                fontweight='bold', color='#1e293b')

    def msg(x1, x2, y, label, dashed=False):
        style = '->' if not dashed else '<-'
        ls = '-' if not dashed else '--'
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                   arrowprops=dict(arrowstyle='->', color='#334155', lw=1,
                                  linestyle=ls))
        mid = (x1 + x2) / 2
        ax.text(mid, y + 0.1, label, ha='center', fontsize=6.5, color='#334155')

    # Messages
    msg(1.0, 2.8, 7.5, "Upload photo / select person")
    msg(2.8, 4.6, 7.0, "POST /api/queries")
    msg(4.6, 11.0, 6.5, "Create QuerySession")
    msg(4.6, 2.8, 6.0, "202 {session_id}")
    msg(4.6, 6.4, 5.5, "BackgroundTask: run_session()")
    msg(6.4, 6.4, 5.0, "Auto-crop + embed query")
    msg(6.4, 8.0, 4.5, "search_camera_top_k()")
    msg(8.0, 8.0, 4.0, "Cosine sim + fusion + threshold")
    msg(8.0, 6.4, 3.5, "list[CameraMatch]")
    msg(6.4, 9.5, 3.0, "reconstruct_route()")
    msg(9.5, 9.5, 2.5, "Greedy walk (spatial+temporal)")
    msg(9.5, 6.4, 2.0, "list[FusedSighting]")
    msg(6.4, 11.0, 1.5, "Persist sightings + route")
    msg(6.4, 2.8, 1.0, "WS: route_complete")

    save_fig(fig, "UML_Sequence.png")


# ═══════════════════════════════════════════════════════════════════════════
# UML COMPONENT DIAGRAM
# ═══════════════════════════════════════════════════════════════════════════

def render_component():
    fig, ax = new_figure("UML Component Diagram — Trace System Architecture", figsize=(11.7, 8.3))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)

    def component(x, y, label, w=2.2, h=0.7, color='#f0f9ff', edge='#0369a1'):
        rect = FancyBboxPatch((x-w/2, y-h/2), w, h,
                              boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor=edge, linewidth=1.2)
        ax.add_patch(rect)
        # Component icon
        ax.add_patch(Rectangle((x+w/2-0.3, y+h/2-0.25), 0.25, 0.15,
                              facecolor='white', edgecolor=edge, linewidth=0.8))
        ax.add_patch(Rectangle((x+w/2-0.3, y+h/2-0.5), 0.25, 0.15,
                              facecolor='white', edgecolor=edge, linewidth=0.8))
        ax.text(x, y, label, ha='center', va='center', fontsize=7.5, color='#1e293b')

    def package(x, y, w, h, label):
        rect = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.05",
                              facecolor='none', edgecolor='#64748b', linewidth=1.2,
                              linestyle='--')
        ax.add_patch(rect)
        ax.text(x + 0.2, y + h - 0.2, label, ha='left', va='top',
                fontsize=9, fontweight='bold', color='#475569')

    # Packages
    package(0.3, 6.0, 3.0, 2.7, "Frontend (Vanilla JS)")
    package(4.0, 6.0, 3.5, 2.7, "Backend API (FastAPI)")
    package(8.0, 6.0, 3.5, 2.7, "AI Pipeline")
    package(0.3, 1.0, 3.0, 4.0, "Service Layer")
    package(4.0, 1.0, 3.5, 4.0, "Data Storage")
    package(8.0, 1.0, 3.5, 4.0, "Model Weights")

    # Frontend components
    component(1.8, 8.0, "index.html\n(SPA Dashboard)", w=2.2, h=0.6)
    component(1.8, 7.0, "app.js\n(Dashboard Logic)", w=2.2, h=0.6)
    component(1.8, 6.3, "api.js\n(REST + WS Client)", w=2.2, h=0.6)

    # Backend components
    component(5.7, 8.2, "upload.py", w=1.6, h=0.5)
    component(5.7, 7.5, "queries.py", w=1.6, h=0.5)
    component(5.7, 6.8, "persons.py", w=1.6, h=0.5)
    component(5.7, 6.3, "analytics.py", w=1.6, h=0.5)

    # AI Pipeline
    component(9.7, 8.2, "detect.py\n(YOLOv8)", w=2.0, h=0.6)
    component(9.7, 7.4, "track.py\n(BoT-SORT)", w=2.0, h=0.6)
    component(9.7, 6.6, "crop_extractor.py", w=2.0, h=0.5)
    component(9.7, 6.1, "embed.py\n(OSNet)", w=2.0, h=0.5)

    # Service layer
    component(1.8, 4.5, "PipelineService", w=2.2, h=0.5)
    component(1.8, 3.7, "MatchingService", w=2.2, h=0.5)
    component(1.8, 2.9, "RouteService", w=2.2, h=0.5)
    component(1.8, 2.1, "EmbeddingService", w=2.2, h=0.5)
    component(1.8, 1.4, "DetectionService\nTrackerService\nCropService", w=2.2, h=0.8)

    # Data storage
    component(5.7, 4.5, "SQLite DB\n(trace.db)", w=2.0, h=0.6, color='#ecfdf5', edge='#065f46')
    component(5.7, 3.5, "embeddings_*.json\n(Galleries)", w=2.2, h=0.6, color='#ecfdf5', edge='#065f46')
    component(5.7, 2.5, "camera_graph.json", w=2.0, h=0.5, color='#ecfdf5', edge='#065f46')
    component(5.7, 1.6, "crops/ + videos/", w=2.0, h=0.5, color='#ecfdf5', edge='#065f46')

    # Model weights
    component(9.7, 4.5, "yolov8n.pt", w=1.8, h=0.5, color='#fef3c7', edge='#92400e')
    component(9.7, 3.7, "OSNet weights", w=1.8, h=0.5, color='#fef3c7', edge='#92400e')
    component(9.7, 2.9, "SCRFD + ArcFace\n(ONNX)", w=2.0, h=0.6, color='#fef3c7', edge='#92400e')
    component(9.7, 2.0, "KPR weights\n(optional)", w=2.0, h=0.6, color='#fef3c7', edge='#92400e')

    save_fig(fig, "UML_Component.png")


# ═══════════════════════════════════════════════════════════════════════════
# CLASS DIAGRAM
# ═══════════════════════════════════════════════════════════════════════════

def render_class_diagram():
    fig, ax = new_figure("UML Class Diagram — Trace Backend & AI Pipeline", figsize=(11.7, 8.3))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)

    def class_box(x, y, name, attrs, methods, w=2.8, h=None, color='#f0f9ff', edge='#0369a1'):
        n_lines = 1 + len(attrs) + len(methods) + 1
        if h is None:
            h = 0.25 + n_lines * 0.18
        rect = FancyBboxPatch((x, y-h), w, h,
                              boxstyle="round,pad=0.03",
                              facecolor=color, edgecolor=edge, linewidth=1.2)
        ax.add_patch(rect)
        # Class name
        ax.text(x + w/2, y - 0.12, name, ha='center', va='center',
                fontsize=8, fontweight='bold', color='#1e293b')
        ax.plot([x+0.1, x+w-0.1], [y-0.22, y-0.22], '-', color=edge, lw=0.6)
        # Attributes
        for i, attr in enumerate(attrs):
            ax.text(x + 0.1, y - 0.35 - i*0.17, attr, ha='left', va='center',
                    fontsize=5.5, color='#374151', family='monospace')
        # Separator
        sep_y = y - 0.35 - len(attrs)*0.17
        ax.plot([x+0.1, x+w-0.1], [sep_y, sep_y], '-', color=edge, lw=0.5)
        # Methods
        for i, meth in enumerate(methods):
            ax.text(x + 0.1, sep_y - 0.13 - i*0.17, meth, ha='left', va='center',
                    fontsize=5.5, color='#374151', family='monospace')
        return (x, y, w, h)

    # ORM Entities
    class_box(0.3, 9.5, "Person",
              ["+id: int <<PK>>", "+name: str", "+watchlist_status: str",
               "+reference_embedding: str?"],
              ["+embedding_vector: list[float]"],
              color='#dbeafe', edge='#1d4ed8')

    class_box(3.5, 9.5, "Camera",
              ["+id: str <<PK>>", "+location: str", "+start_time: str?",
               "+is_active: bool"],
              [],
              color='#dbeafe', edge='#1d4ed8')

    class_box(6.7, 9.5, "QuerySession",
              ["+id: int <<PK>>", "+person_id: int? <<FK>>", "+status: str",
               "+progress_pct: int", "+query_image_path: str?"],
              [],
              color='#dbeafe', edge='#1d4ed8')

    class_box(10.0, 9.5, "Sighting",
              ["+id: int <<PK>>", "+session_id: int <<FK>>", "+camera_id: str <<FK>>",
               "+track_id: int", "+best_confidence: float",
               "+appearance_score: float", "+fusion_score: float"],
              [],
              color='#dbeafe', edge='#1d4ed8')

    class_box(0.3, 6.5, "RouteStep",
              ["+id: int <<PK>>", "+session_id: int <<FK>>", "+step_order: int",
               "+sighting_id: int <<FK>>"],
              [],
              color='#dbeafe', edge='#1d4ed8')

    class_box(3.5, 6.5, "Alert",
              ["+id: int <<PK>>", "+session_id: int <<FK>>", "+person_id: int? <<FK>>",
               "+severity: str", "+title: str", "+acknowledged: bool"],
              [],
              color='#dbeafe', edge='#1d4ed8')

    # Services
    class_box(6.7, 6.5, "PipelineService",
              ["-_ws: WebSocketManager"],
              ["+run_session(session_id)", "-_execute(session, db)",
               "-_persist_sighting(...)", "-_fire_alert(...)"],
              color='#fef3c7', edge='#92400e')

    class_box(10.0, 6.5, "MatchingService",
              ["-_threshold: float", "-_body_weight: float",
               "-_face_weight: float", "-_kpr_weight: float"],
              ["+search_camera_top_k(...)", "+search_camera(...)",
               "-_compute_crop_similarities(...)"],
              color='#fef3c7', edge='#92400e')

    class_box(0.3, 4.0, "RouteService",
              ["-_adjacency: dict", "-_w_app: float",
               "-_w_spa: float", "-_w_tem: float"],
              ["+reconstruct_route(candidates)", "+compute_route_confidence(route)",
               "-_fuse(app, spa, tem)"],
              color='#fef3c7', edge='#92400e')

    class_box(3.5, 4.0, "EmbeddingService",
              ["-_model: Any?", "-_device: Any?"],
              ["+load_model()", "+embed_single(image_path)",
               "+batch_embed(metadata, crops, out)", "+load_gallery(path)"],
              color='#fef3c7', edge='#92400e')

    class_box(6.7, 4.0, "WebSocketManager",
              ["-_connections: list[WS]"],
              ["+connect(websocket)", "+disconnect(websocket)",
               "+broadcast(data)"],
              color='#fef3c7', edge='#92400e')

    class_box(10.0, 4.0, "Settings",
              ["+host: str", "+port: int", "+no_match_threshold: float",
               "+fusion_appearance_weight: float", "+yolo_model: str",
               "+camera_graph_path: Path"],
              [],
              color='#fef3c7', edge='#92400e')

    # Data classes
    class_box(0.3, 2.0, "CameraMatch",
              ["+camera_id: str", "+track_id: int", "+appearance_score: float",
               "+best_confidence: float", "+first_seen: str?",
               "+face_sim: float?", "+kpr_sim: float?"],
              [],
              color='#d1fae5', edge='#065f46')

    class_box(3.5, 2.0, "FusedSighting",
              ["+match: CameraMatch", "+spatial: float",
               "+temporal: float", "+fusion: float"],
              [],
              color='#d1fae5', edge='#065f46')

    # AI Pipeline
    class_box(6.7, 2.0, "DetectionService",
              [],
              ["+run(video, camera_id)", "+load_detections(path)"],
              color='#fce7f3', edge='#9d174d')

    class_box(10.0, 2.0, "TrackerService",
              [],
              ["+run(video, camera_id)", "+load_tracks(path)",
               "+group_by_track(tracks)"],
              color='#fce7f3', edge='#9d174d')

    # Relationships
    ax.annotate('', xy=(6.7, 9.0), xytext=(6.3, 9.0),
                arrowprops=dict(arrowstyle='-|>', color='#64748b', lw=1))
    ax.text(6.5, 9.15, "1..*", fontsize=6, color='#64748b')

    ax.annotate('', xy=(10.0, 9.0), xytext=(9.5, 9.0),
                arrowprops=dict(arrowstyle='-|>', color='#64748b', lw=1))

    save_fig(fig, "UML_Class.png")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Rendering diagrams...")
    render_dfd_level_0()
    render_dfd_level_1()
    render_dfd_level_2_video()
    render_dfd_level_2_matching()
    render_dfd_level_2_route()
    render_use_case()
    render_activity()
    render_sequence()
    render_component()
    render_class_diagram()
    print(f"\nAll diagrams saved to: {OUTPUT_DIR}")
    print(f"Total: {len(list(OUTPUT_DIR.glob('*.png')))} PNG files")