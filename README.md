# Intelligent Fitness Recommendation & Coaching System
### Hệ thống gợi ý bài tập cá nhân hóa & huấn luyện thông minh — YOLO · RAG · Computer Vision

> **Trạng thái:** 🟢 Phase 1 — Thiết kế & Thu thập dữ liệu  
> **Môi trường:** Google Colab A100 · Google Drive · FastAPI · React

---

## Chúng ta đang xây dựng gì?

> **Một câu:** Hệ thống AI tự động nhận diện và phân tích tư thế tập luyện qua ảnh/video, gợi ý kế hoạch tập cá nhân hóa theo mục tiêu người dùng, và trả lời câu hỏi fitness trong thời gian thực dựa trên knowledge base chuyên sâu đa ngôn ngữ.

**Sản phẩm cuối cùng trông như thế nào:**
- Một **web app** cho phép upload ảnh/video tập gym → hệ thống tự động nhận diện bài tập, phát hiện lỗi tư thế và hiển thị skeleton overlay
- Giao diện **Recommendation** gợi ý kế hoạch tập 7 ngày cá nhân hóa theo thể trạng & mục tiêu từng người
- **Fitness Chatbot** trả lời câu hỏi tiếng Việt & tiếng Anh dựa trên knowledge base từ sách chuyên ngành + nghiên cứu PubMed
- **Progress Tracking** theo dõi lịch sử session, form errors, và tiến trình theo thời gian

---

## Định nghĩa "Thành công" (Definition of Done)

Dự án coi là **hoàn thành** khi đáp ứng đủ các tiêu chí sau:

| # | Tiêu chí |
|---|---|
| 1 | YOLOv8 classification đạt **Accuracy ≥ 85%** trên tập test (squat / deadlift / bench) |
| 2 | Pose estimation pipeline chạy với **latency < 150ms/frame** |
| 3 | Heuristic form checker phát hiện đúng lỗi tư thế với **≥ 3 rule** |
| 4 | Recommendation engine sinh được **workout plan 7 ngày** cá nhân hóa |
| 5 | RAG system trả lời đúng **≥ 85%** câu hỏi fitness theo ngữ cảnh (tiếng Việt + tiếng Anh) |
| 6 | Web app end-to-end **chạy được demo** với người dùng thực tế |

---

## Mỗi người đóng góp gì vào sản phẩm?

```
Member 1 (CV Data Engineer)   →  Thu thập MPII · Leeds · YouTube · tự chụp gym
                                  Annotation Roboflow/CVAT → YOLO pose .txt
                                  Tổ chức toàn bộ data pipeline trên Google Drive

Member 2 (ML Engineer – CV)   →  Fine-tune YOLOv8-cls + YOLOv8-pose trên Colab A100
                                  Viết inference module + heuristic angle engine
                                  Form error report + keypoint visualization

Member 3 (ML Engineer – Rec)  →  User profiling + exercise schema design
                                  Rule-based → Content-based → Hybrid recommender
                                  Workout Plan Generator 7 ngày + progressive overload

Member 4 (NLP / RAG Engineer) →  Knowledge base từ Tier 1–3 (sách PDF · PubMed · Blog)
                                  LangChain + ChromaDB (BGE-M3) + Gemini/GPT-4o-mini
                                  Myth-busting · Safe fallback · Citations panel

Member 5 (Full-Stack)         →  FastAPI backend 5 endpoints · PostgreSQL · Docker
                                  React web app 4 trang · Tích hợp 3 module AI
                                  Demo hoàn chỉnh + deployment
```

---

## Kiến trúc hệ thống

```
[Upload ảnh / Video .mp4]
         │
         ▼
  Stage 1 — YOLOv8-cls
  Exercise Classification
  label: squat | deadlift | bench_press
  Confidence > 0.7? ──► [REJECT — yêu cầu upload lại]
         │
         ▼
  Stage 2 — YOLOv8-pose
  Human Pose Extraction
  17 keypoints (x, y, conf) → normalize
         │
         ▼
  Stage 3 — Form Feedback (Heuristics)
  Tính góc đầu gối / cột sống / cùi chỏ → so threshold
  ├── Form Error Report (JSON)
  ├── Keypoint Visualization (skeleton overlay)
  └── RAG Chatbot trigger
         │
  User Profile + Session Log (PostgreSQL)
         │
  ┌──────────────────────────┐
  │  Recommendation Engine   │
  │  Phase 1: Rule-based     │
  │  Phase 2: Content-based  │
  │  Phase 3: Hybrid         │
  │  → Workout Plan 7 ngày   │
  └──────────┬───────────────┘
             │
  ┌──────────▼───────────────┐
  │  RAG Fitness Chatbot     │
  │  LangChain + ChromaDB    │
  │  BGE-M3 · Gemini API     │
  │  Myth-busting · Fallback │
  └──────────┬───────────────┘
             │
     FastAPI Backend
             │
     React Frontend  ← người dùng tương tác ở đây
```

> **Compute:** Google Colab A100 (training) · Local inference via FastAPI  
> **Storage:** Google Drive (datasets + model weights) · PostgreSQL (user data) · ChromaDB (vector store)

---

## Data Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CV Model Data (Member 1 + 2)                     │
├───────────────────┬─────────────────────────────────────────────────┤
│ MPII dataset      │ YouTube crawl     │ Leeds Sport  │ Self-captured │
│ ~25K ảnh .mat     │ yt-dlp @10fps     │ 2K+ poses    │ 4+ người gym │
└───────────────────┴───────────────────┴──────────────┴──────────────┘
         │                    Raw data pool — merge + dedup
         ▼
  Preprocessing (Google Colab A100)
  Resize 640×640 · normalize · Albumentations augment
         │
  Annotation — Roboflow / CVAT
  17 keypoints + class label → YOLO pose .txt
         │
  Train / Val / Test — 70 / 15 / 15
  └── Train → YOLOv8 fine-tune
  └── Val   → monitor mAP / AP loss
  └── Test  → final benchmark

┌─────────────────────────────────────────────────────────────────────┐
│                  RAG Knowledge Base (Member 4)                      │
├───────────────────┬─────────────────────────────────────────────────┤
│ Sách PDF          │ PubMed API        │ Blog crawl   │ ACSM PDF     │
│ Helms·McGill·     │ Abstract + XML    │ RP Strength  │ Official     │
│ Rippetoe          │ Hypertrophy·MPS   │ SBS·gym.vn   │ guidelines   │
└───────────────────┴───────────────────┴──────────────┴──────────────┘
         │          Document parsing + cleaning (PyMuPDF · BeautifulSoup)
         ▼
  Chunking — 500–1000 tokens, overlap 100
         │
  Embedding — BGE-M3 (multilingual)
         │
  ChromaDB Vector Store — HNSW index
```

---

## Tài liệu chi tiết

| Tài liệu | Nội dung |
|---|---|
| [`README.md`](README.md) | File này — Tổng quan dự án, kiến trúc, quick start |
| [`TECH_STACK.md`](TECH_STACK.md) | Tech stack chi tiết từng module, lý do chọn, code mẫu, requirements |
| [`TEAM_ASSIGNMENTS.md`](TEAM_ASSIGNMENTS.md) | Phân công tasks theo Phase cho 5 thành viên, ma trận phụ thuộc, handoff guide |
| [`TEAM_GUIDE.md`](TEAM_GUIDE.md) | Git workflow, coding conventions, PR process, Colab/Drive conventions |

---

## Lộ trình 4 Phases

| Phase | Mục tiêu cần đạt được |
|---|---|
| **Phase 1** | Annotated CV dataset sẵn sàng trên Drive · ChromaDB KB embedded · FastAPI skeleton chạy |
| **Phase 2** | YOLOv8 Accuracy ≥ 85% · Hybrid recommender hoạt động · RAG trả lời đúng |
| **Phase 3** | End-to-end: Upload → Form feedback → Recommend → Chatbot |
| **Phase 4** | Demo hoàn chỉnh · Báo cáo đầy đủ · Benchmark latency + accuracy |

---

## Cấu trúc thư mục

```
fitness-recommendation-system/
├── src/
│   ├── cv/
│   │   ├── inference.py           # YOLO inference, load weights từ Drive
│   │   ├── heuristics.py          # Angle computation (knee/spine/elbow)
│   │   ├── visualizer.py          # Keypoint skeleton overlay
│   │   └── form_reporter.py       # Form error JSON output
│   ├── recsys/
│   │   ├── feature_engineering.py
│   │   ├── rule_recommender.py    # Phase 1 baseline
│   │   ├── content_based.py       # Phase 2
│   │   ├── hybrid_recommender.py  # Phase 3 (chính)
│   │   ├── plan_generator.py      # 7-day workout plan
│   │   └── progressive_overload.py
│   └── rag/
│       ├── document_loader.py     # PyMuPDF + BeautifulSoup
│       ├── chunker.py             # 500-1000 tokens, overlap 100
│       ├── embedder.py            # BGE-M3
│       ├── vector_store.py        # ChromaDB HNSW
│       ├── chain.py               # LangChain RAG chain
│       ├── llm_connector.py       # Gemini / GPT-4o-mini API
│       ├── myth_buster.py
│       └── guardrail.py
├── app/
│   ├── main.py
│   ├── routers/
│   │   ├── pose.py                # POST /predict
│   │   ├── recommend.py           # POST /recommend
│   │   ├── qa.py                  # POST /ask
│   │   └── user.py                # CRUD /user/profile
│   ├── services/
│   │   └── session_service.py
│   └── middleware/                # CORS · JWT · logging
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── FormCheck.jsx      # Upload + skeleton overlay + error report
│   │   │   ├── Recommendation.jsx # Workout plan 7 ngày
│   │   │   ├── Chatbot.jsx        # Chat + citations panel
│   │   │   └── Progress.jsx       # Progress tracking
│   │   └── services/
│   │       └── api.js
│   └── package.json
├── db/
│   └── schema.sql
├── schemas/
│   ├── exercise_schema.json       # M1 → M3, M5
│   ├── user_schema.json           # M3
│   └── workout_plan_schema.json   # M3 → M5
├── notebooks/                     # Colab notebooks (committed sau training)
│   ├── cv_preprocess.ipynb
│   ├── yolo_cls_training.ipynb
│   ├── yolo_pose_training.ipynb
│   └── rag_ingest.ipynb
├── reports/
├── benchmarks/
├── tests/
│   └── test_qa.py
├── docker-compose.yml
├── requirements.txt
├── README.md
├── TECH_STACK.md
├── TEAM_ASSIGNMENTS.md
├── TEAM_GUIDE.md
└── HANDOFF_LOG.md
```

---

## Quick Start

```bash
# 1. Clone repo
git clone https://github.com/<org>/fitness-recommendation-system.git
cd fitness-recommendation-system

# 2. Đọc tài liệu theo thứ tự
#    README.md (file này) → TEAM_ASSIGNMENTS.md → TECH_STACK.md → TEAM_GUIDE.md

# 3. Tạo file .env từ template
cp .env.example .env
# Điền: GOOGLE_DRIVE_PATH, GEMINI_API_KEY hoặc OPENAI_API_KEY, DATABASE_URL

# 4. Setup môi trường local
conda create -n fitness-ai python=3.10
conda activate fitness-ai
pip install -r requirements.txt

# 5. Khởi động services (PostgreSQL + ChromaDB)
docker-compose up -d

# 6. Chạy RAG ingestion (lần đầu)
python src/rag/document_loader.py
python src/rag/embedder.py

# 7. Chạy API server
uvicorn app.main:app --reload

# 8. Chạy frontend
cd frontend
npm install && npm start
```

### Chạy từng module riêng lẻ

```bash
# CV Inference (cần model weights từ Drive)
python src/cv/inference.py --input path/to/image.jpg

# RAG Chatbot (standalone test)
python -c "from src.rag.chain import ask; print(ask('Squat đúng kỹ thuật như thế nào?'))"

# Recommendation (standalone test)
python -c "from src.recsys.hybrid_recommender import recommend; print(recommend(goal='fat_loss', level='beginner'))"
```

### Colab Training (Member 2)

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone repo & install
!git clone https://github.com/<org>/fitness-recommendation-system.git
!pip install ultralytics albumentations

# Train YOLOv8-cls
!yolo task=classify mode=train \
  model=yolov8n-cls.pt \
  data=/content/drive/MyDrive/fitness-system/data/cv/data.yaml \
  epochs=50 imgsz=640 \
  project=/content/drive/MyDrive/fitness-system/models/runs

# Train YOLOv8-pose
!yolo task=pose mode=train \
  model=yolov8n-pose.pt \
  data=/content/drive/MyDrive/fitness-system/data/cv/data.yaml \
  epochs=50 imgsz=640 \
  project=/content/drive/MyDrive/fitness-system/models/runs
```
