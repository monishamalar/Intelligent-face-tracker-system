# Intelligent Face Tracker with Auto-Registration and Visitor Counting

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Framework: YOLOv8 + InsightFace + ByteTrack](https://img.shields.io/badge/Pipeline-YOLOv8%20%2B%20InsightFace%20%2B%20ByteTrack-orange.svg)]()

> A production-grade, AI-driven unique visitor counter and face tracking system. Detects, tracks, and recognizes faces in real-time from video files or live RTSP camera feeds, auto-registers new identities on first sight, logs exactly-once entry and exit events with cropped snapshots, and maintains an unalterable database of unique visitors.

---

## 1. Problem Statement & Objective

Traditional visitor counting systems rely on basic tripwires, break-beams, or non-re-identifying person detectors. They suffer from critical flaws:
* **Double Counting:** Visitors pacing near boundaries or walking back and forth are repeatedly counted.
* **Tracker ID Loss:** Standard tracker IDs reset or swap when people turn away or get occluded.
* **Lack of Audit Proof:** No visual record of who entered or exited.

### Objective
Build a modular, robust AI system that:
1. Ingests video from a file or live RTSP camera stream.
2. Detects faces using a YOLO-based detector (configurable frame skip $N$).
3. Generates 512-dimensional normalized embeddings using InsightFace/ArcFace.
4. Auto-registers unseen faces with unique identifiers (`FACE_000001`, `FACE_000002`...).
5. Separates spatial `track_id` (ByteTrack) from permanent `face_id`.
6. Enforces an exactly-once State Machine: exactly one `ENTRY` event upon arrival, and exactly one `EXIT` event after an absence grace timeout.
7. Saves timestamped cropped face images into a structured filesystem (`logs/entries/YYYY-MM-DD/`, `logs/exits/YYYY-MM-DD/`).
8. Records audit metadata in SQLite database and logs all events in `logs/events.log`.
9. Exposes `get_unique_visitor_count()` guaranteed against duplicate counting.

---

## 2. System Architecture

```mermaid
flowchart TD
    subgraph Ingestion ["1. Video Ingestion Layer"]
        A[File Source: sample.mp4] --> C[Threaded VideoSource]
        B[Live RTSP Camera Stream] --> C
        C --> D{Frame Decimator / Skip N}
    end

    subgraph Perception ["2. Perception & Extraction Layer"]
        D -- "Every Nth Frame" --> E[YOLOv8-Face Detector]
        D -- "Intervening Frames" --> F[Kalman Motion Prediction]
        E --> G[Face Quality & Size Filter]
        G --> H[InsightFace / ArcFace Extractor 512-D]
    end

    subgraph Tracking_ID ["3. Association & Tracking Engine"]
        E --> I[ByteTrack IoU Matching]
        F --> I
        H --> J[Identity Manager & Cosine Metric]
        I <--> J
        J -- "Sim < Threshold" --> K[Auto-Register New FACE_ID]
        J -- "Sim >= Threshold" --> L[Bind to Existing FACE_ID]
    end

    subgraph StateMachine ["4. Lifecycle & Event State Machine"]
        K --> M[Visitor State Machine]
        L --> M
        M -- "New Presence" --> N[Emit Exactly-Once ENTRY]
        M -- "Active" --> O[Update Presence]
        M -- "Missing > Timeout" --> P[Emit Exactly-Once EXIT]
    end

    subgraph Persistence ["5. Audit, Database & Storage"]
        N --> Q[(SQLite Database)]
        P --> Q
        N --> R[Disk: logs/entries/YYYY-MM-DD/]
        P --> R[Disk: logs/exits/YYYY-MM-DD/]
        K --> R[Disk: logs/registrations/YYYY-MM-DD/]
        M --> S[System Log: logs/events.log]
    end
```

---

## 3. Technology Stack

| Module | Technology | Justification |
|---|---|---|
| **Language** | Python 3.10+ | Standard for computer vision and deep learning pipelines. |
| **Face Detection** | YOLOv8-Face / Ultralytics | Real-time single-shot face detector (~10ms CPU, ~2ms GPU). |
| **Face Recognition** | InsightFace (ArcFace / `buffalo_l`) | SOTA additive angular margin loss yielding 512-D L2-normalized hyperspherical vectors. |
| **Object Tracking** | ByteTrack | Associates low- and high-confidence boxes using Kalman filters without heavy ReID compute. |
| **Database** | SQLite + SQLAlchemy | Embedded, self-contained relational DB with binary BLOB vector storage and foreign keys. |
| **Filesystem Storage** | Python `Pathlib` / OpenCV | Structured date-partitioned folder hierarchy (`logs/<type>/YYYY-MM-DD/`). |
| **Configuration** | JSON (`config.json`) | Centralized, validated schema for thresholds, frame skip, and filepaths. |
| **Logging** | Python `logging` | Rotating file handler generating structured `logs/events.log` with millisecond precision. |
| **Input Sources** | OpenCV `VideoCapture` + Threaded RTSP | Abstracted interface supporting MP4, AVI, Webcams, and RTSP stream reconnects. |

---

## 4. Project Directory Structure

```text
intelligent-face-tracker/
│
├── config.json                     # System configuration file
├── requirements.txt                # Pinned production dependencies
├── README.md                       # Comprehensive guide and documentation
├── .gitignore                      # Git exclusion rules
│
├── app/
│   ├── __init__.py
│   ├── main.py                     # CLI entry point
│   ├── config.py                   # Pydantic / dataclass config parser & validator
│   │
│   ├── video/
│   │   ├── base.py                 # Abstract BaseVideoSource
│   │   ├── file_source.py          # FileVideoSource (local MP4 files)
│   │   └── rtsp_source.py          # RTSPVideoSource (Threaded, non-blocking, auto-reconnect)
│   │
│   ├── detection/
│   │   ├── detector.py             # YOLO Face Detector with simulation fallback
│   │   └── face_filter.py          # Face quality filter (min size, confidence)
│   │
│   ├── recognition/
│   │   ├── recognizer.py           # InsightFace 512-D embedding extractor
│   │   └── distance.py             # Vectorized Cosine Similarity & L2 normalization
│   │
│   ├── tracking/
│   │   └── tracker.py              # ByteTrack / IoU association engine
│   │
│   ├── identity/
│   │   └── identity_manager.py     # Gallery matching, FACE_XXXXXX ID generation
│   │
│   ├── events/
│   │   ├── state_machine.py        # Exactly-once State Machine (NEW, ACTIVE, MISSING, EXITED)
│   │   └── event_manager.py        # Event dispatcher for DB, disk, and logs
│   │
│   ├── storage/
│   │   └── image_store.py          # Atomic cropped face saver (YYYY-MM-DD structure)
│   │
│   ├── database/
│   │   ├── connection.py           # SQLite connection manager with WAL mode
│   │   ├── models.py               # Database schemas & records
│   │   └── repository.py           # Atomic CRUD operations & visitor counting
│   │
│   ├── pipeline/
│   │   └── pipeline.py             # Master pipeline orchestrator
│   │
│   └── utils/
│       ├── logging_config.py       # Rotating file logger setup (events.log)
│       └── benchmark.py            # Latency, FPS, CPU %, and RAM profiler
│
├── tests/
│   ├── test_config.py              # Configuration validation tests
│   ├── test_database.py            # SQLite schema, unique count, and event tests
│   ├── test_env.py                 # Environment smoke tests
│   ├── test_identity.py            # Cosine similarity and embedding serialization tests
│   └── test_state_machine.py       # Simulation test for enter -> occlude -> return -> exit
│
├── docs/
│   └── AI_PLANNING.md              # AI workflow documentation, prompt log, and trade-offs
│
├── logs/                           # Runtime logs and cropped face output directory
│   ├── events.log
│   ├── entries/
│   ├── exits/
│   └── registrations/
│
└── models/                         # Pre-trained model weights directory
```

---

## 5. Configuration (`config.json`)

All operational parameters are centralized in `config.json` and validated by `app/config.py`:

```json
{
  "system": {
    "app_name": "Intelligent Face Tracker",
    "version": "1.0.0",
    "device": "auto"
  },
  "input": {
    "source": "sample/sample_walkthrough.mp4",
    "source_type": "video",
    "rtsp_reconnect_interval_sec": 5.0,
    "max_reconnect_attempts": 10
  },
  "detection": {
    "model_name": "yolov8n-face",
    "confidence_threshold": 0.55,
    "frame_skip": 5,
    "min_face_size": [60, 60],
    "iou_threshold": 0.45
  },
  "recognition": {
    "model_name": "buffalo_l",
    "embedding_dimension": 512,
    "similarity_threshold": 0.45,
    "distance_metric": "cosine",
    "max_gallery_embeddings_per_face": 5
  },
  "tracking": {
    "tracker": "bytetrack",
    "track_thresh": 0.5,
    "track_buffer": 30,
    "match_thresh": 0.8,
    "exit_timeout_frames": 30,
    "exit_timeout_seconds": 2.0
  },
  "storage": {
    "base_dir": "logs",
    "entries_dir": "logs/entries",
    "exits_dir": "logs/exits",
    "registrations_dir": "logs/registrations",
    "save_crop_margin_pct": 0.15
  },
  "database": {
    "sqlite_path": "visitor_tracker.db",
    "echo_sql": false
  },
  "logging": {
    "log_file": "logs/events.log",
    "log_level": "INFO",
    "console_output": true,
    "max_bytes": 10485760,
    "backup_count": 5
  }
}
```

---

## 6. Installation & Quick Start

### 6.1 Requirements
* Python 3.10 or higher.
* Linux / macOS / Windows.

### 6.2 Setup
```bash
# 1. Clone repository
git clone <your-repo-url>
cd intelligent-face-tracker

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 6.3 Run with Sample Video File
```bash
python3 app/main.py --source sample/sample_walkthrough.mp4
```

### 6.4 Run with Live RTSP Camera Feed
```bash
python3 app/main.py --source "rtsp://username:password@192.168.1.100:554/stream1" --rtsp
```

### 6.5 Run Unit & Simulation Tests
```bash
python3 -m unittest discover -s tests -v
```

---

## 7. Exactly-Once State Machine & Event Logging

The system models each person's visit lifecycle using a finite state machine:

```
[NEW DETECTED]
      │
      ▼
[UNREGISTERED] ──(Match < Threshold)──► [AUTO-REGISTER NEW FACE_ID]
      │                                             │
      ├────────────────(Match >= Threshold)─────────┘
      │
      ▼
   [ACTIVE] ───────────────► Emit exactly ONE "ENTRY" Event & Save Crop
      │        ▲
      │        │ (Returned before timeout)
      ▼        │
  [MISSING] ───┘
      │
      ▼ (Absent > exit_timeout_frames)
   [EXITED] ───────────────► Emit exactly ONE "EXIT" Event & Save Crop
```

* **Idempotency Guarantee:** Once an `ENTRY` event is emitted for a session, `entry_emitted` is set to `True`. No subsequent frames can re-emit `ENTRY`.
* **Grace Timeout:** When a visitor is temporarily occluded (e.g. by another person or turning their back), they enter `MISSING` state. If they reappear before `exit_timeout_frames` (default 30 frames / 1-2 sec), the session resumes as `ACTIVE` without a duplicate entry event.
* **Exit Finalization:** When absence exceeds `exit_timeout_frames`, exactly one `EXIT` event is logged with the last known snapshot crop, the active DB track session is marked `COMPLETED`, and the session is closed.

---

## 8. Database Schema & Queries

SQLite database located at `visitor_tracker.db`:

```sql
-- Registered human identities
CREATE TABLE faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    face_id TEXT UNIQUE NOT NULL,
    first_seen_at TIMESTAMP NOT NULL,
    last_seen_at TIMESTAMP NOT NULL,
    registration_image_path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 512-dimensional facial embedding vectors
CREATE TABLE face_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    face_id TEXT NOT NULL,
    embedding BLOB NOT NULL,                 -- 512-float32 binary representation
    quality_score REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(face_id) REFERENCES faces(face_id) ON DELETE CASCADE
);

-- Audit log of ENTRY and EXIT events
CREATE TABLE events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    face_id TEXT NOT NULL,
    event_type TEXT NOT NULL CHECK(event_type IN ('ENTRY', 'EXIT')),
    timestamp TIMESTAMP NOT NULL,
    image_path TEXT NOT NULL,
    tracker_id INTEGER,
    confidence REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(face_id) REFERENCES faces(face_id)
);

-- Physical tracking sessions
CREATE TABLE tracks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    face_id TEXT NOT NULL,
    tracker_id INTEGER NOT NULL,
    started_at TIMESTAMP NOT NULL,
    last_seen_at TIMESTAMP NOT NULL,
    ended_at TIMESTAMP,
    status TEXT NOT NULL CHECK(status IN ('ACTIVE', 'MISSING', 'COMPLETED')),
    FOREIGN KEY(face_id) REFERENCES faces(face_id)
);
```

### Verification SQL Queries
```sql
-- 1. Unique Visitor Count
SELECT COUNT(DISTINCT face_id) FROM faces;

-- 2. Entry vs Exit Totals
SELECT event_type, COUNT(*) FROM events GROUP BY event_type;

-- 3. Audit trail for a specific visitor
SELECT event_type, timestamp, image_path, tracker_id FROM events WHERE face_id = 'FACE_000001' ORDER BY id ASC;
```

---

## 9. Performance & Compute Load Consumption

Estimated computational load across hardware configurations:

| Component | CPU-Only (Intel i7 / Ryzen 7, 8 Cores) | GPU (NVIDIA RTX 3060 / T4) |
|---|---|---|
| **YOLOv8-Face Detection** | 12 - 20 ms / inference | 2 - 4 ms / inference |
| **InsightFace 512-D Embedding** | 25 - 40 ms / face | 4 - 7 ms / face |
| **ByteTrack Spatial Update** | 1 - 2 ms / frame | 1 ms / frame |
| **Cosine Vector Search** | < 0.2 ms / query | < 0.2 ms / query |
| **Disk Crop Write (Asynchronous)** | 2 - 5 ms / event | 2 - 5 ms / event |
| **System RAM Usage** | ~1.2 GB | ~1.5 GB |
| **GPU VRAM Usage** | N/A | ~1.1 GB |
| **Effective Pipeline Speed (Skip=5)** | **25 - 32 FPS** | **60+ FPS** |

* **Effect of Frame Skip:** Running face detection every $N=5$ frames reduces detection neural network load by 80%, while ByteTrack maintains continuous spatial trajectories on intervening frames.

---

## 10. Sample Output Demonstration

### Sample `logs/events.log`
```text
2026-10-02 05:59:34 [INFO] [face_tracker.main:62] Initializing Intelligent Face Tracker v1.0.0
2026-10-02 05:59:34 [INFO] [face_tracker.identity.associate_or_register:105] Auto-registered new identity: FACE_000001 (Track #1)
2026-10-02 05:59:34 [INFO] [face_tracker.events.handle_active_face:72] ENTRY EVENT: FACE_000001 (Track #1) logged. Saved crop: logs/entries/2026-10-02/FACE_000001_055934_587513.jpg
2026-10-02 05:59:38 [INFO] [face_tracker.events.handle_frame_completion:95] EXIT EVENT: FACE_000001 (Track #1) exited frame after grace period. Saved crop: logs/exits/2026-10-02/FACE_000001_055938_102941.jpg
```

### Structured Filesystem
```text
logs/
├── events.log
├── entries/
│   └── 2026-10-02/
│       ├── FACE_000001_055934_587513.jpg
│       └── FACE_000002_060112_982144.jpg
└── exits/
    └── 2026-10-02/
        ├── FACE_000001_055938_102941.jpg
        └── FACE_000002_060245_192837.jpg
```

---

## 11. Demonstration Video & Hackathon Submission

* **Loom / YouTube Demonstration:** https://youtu.be/P0u_jupBVY4

---

This project is a part of a hackathon run by https://katomaran.com




🚀 Terminal Cheat Sheet: How to Run the Project in VS Code
Keep this handy whenever you open VS Code to work on or demonstrate this project.
📌 Daily Startup Routine (Every Time You Open VS Code)
Step 1: Open the Project Folder
Open VS Code, then go to:
File > Open Folder... and select:
C:\Users\pmoni\Downloads\intelligent-face-tracker-visitor-counter
Step 2: Open Terminal & Activate the Virtual Environment
Open the VS Code terminal (Ctrl + ~ or top menu: Terminal > New Terminal).
Run this command to activate your Python environment:
code
Powershell
.\venv\Scripts\Activate.ps1
(If PowerShell asks about execution policy, run:)
code
Powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
(You will see (venv) appear on the left of your terminal prompt).
Step 3: Start the Web Dashboard
In the terminal, run:
code
Powershell
npm run dev
You will see:
code
Code
Intelligent Face Tracker backend running at http://localhost:3000
Now open your browser to:
👉 http://localhost:3000
🧪 Testing & Verification Commands
Whenever you want to test your models, database, and logic, open a terminal with (venv) active:
1. Run All 14 Automated Unit Tests
code
Powershell
python -m unittest discover -s tests -v
Output: Ran 14 tests in 0.3s ... OK
What it tests: Database consistency, state machine, ArcFace vector cosine distance, and configuration validation.
2. Run the Computer Vision Pipeline (CLI Video Simulation)
code
Powershell
python app/main.py --max-frames 30
Output: Processes 30 video frames at ~30 FPS, detects faces, auto-registers visitors, logs ENTRY events, and prints the summary report.
3. Run Pipeline with a Custom Video File
code
Powershell
python app/main.py --source "path/to/your/video.mp4" --max-frames 100
4. Run Pipeline Directly with Your Webcam
code
Powershell
python app/main.py --source 0
(Press Ctrl + C in the terminal to stop processing anytime).
🛠️ Handy Maintenance Commands
Check SQLite Database Contents from Terminal
code
Powershell
python -c "from app.database.connection import DatabaseManager; from app.database.repository import VisitorRepository; r = VisitorRepository(DatabaseManager()); print('Unique Visitors:', r.get_unique_visitor_count(), '| Total Entries:', r.get_total_events_count('ENTRY'), '| Total Exits:', r.get_total_events_count('EXIT'))"
Stop the Server
In the terminal where npm run dev is running, press:
code
Code
Ctrl + C
Then type Y and press Enter.