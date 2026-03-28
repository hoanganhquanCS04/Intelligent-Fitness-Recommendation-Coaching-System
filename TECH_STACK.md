# TECH STACK — Intelligent Fitness Recommendation & Coaching System
## Công nghệ chi tiết từng module · Lý do chọn · Requirements

> **Môi trường training:** Google Colab A100  
> **Storage:** Google Drive  
> **Runtime:** FastAPI + React + PostgreSQL + ChromaDB

---

## Tổng quan Tech Stack

```
┌────────────────────────────────────────────────────────────────────────┐
│                         FRONTEND LAYER                                 │
│              React (Vite) · Tailwind CSS · Recharts · Axios            │
├────────────────────────────────────────────────────────────────────────┤
│                          API LAYER                                     │
│          FastAPI · Uvicorn · Pydantic · JWT (python-jose)              │
├───────────────┬────────────────┬───────────────────────────────────────┤
│  CV MODULE    │  RECSYS MODULE │  RAG / NLP MODULE                     │
│  YOLOv8       │  Scikit-learn  │  LangChain · ChromaDB · BGE-M3        │
│  Ultralytics  │  Pandas/NumPy  │  Gemini API / GPT-4o-mini             │
│  OpenCV       │  JSON plan     │  PyMuPDF · BeautifulSoup              │
├───────────────┴────────────────┴───────────────────────────────────────┤
│                         DATA LAYER                                     │
│     PostgreSQL · ChromaDB (HNSW) · Google Drive · Roboflow             │
├────────────────────────────────────────────────────────────────────────┤
│                       INFRA / DEVOPS                                   │
│              Docker Compose · Google Colab A100 · Git                  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Module 1 — CV Data Pipeline (Member 1)

### Công nghệ

| Công nghệ | Version | Mục đích | Lý do chọn |
|---|---|---|---|
| **yt-dlp** | latest | Crawl video YouTube theo keyword bài tập | Ổn định hơn youtube-dl, hỗ trợ nhiều format |
| **OpenCV** | 4.9+ | Extract frame từ video @10fps | Industry standard cho video processing |
| **Albumentations** | 1.4+ | Augmentation: flip, rotate, brightness | Nhanh hơn torchvision transforms, pipeline sạch |
| **Roboflow** | web/API | Annotation 17 keypoints + class label, export YOLO pose | UI annotation tốt nhất cho pose dataset, export chuẩn YOLO |
| **CVAT** | web | Alternative annotation tool | Open-source, self-host được |
| **Google Drive** | — | Lưu toàn bộ raw data, processed data, model weights | Dễ share trong team, mount được từ Colab |
| **Google Colab A100** | — | Preprocessing, resize, augmentation | Free GPU mạnh, tích hợp Drive |

### Quy trình

```python
# 1. Crawl YouTube
# yt-dlp --extract-audio no -f mp4 "squat tutorial" -o "%(title)s.%(ext)s"

# 2. Extract frames với OpenCV
import cv2

def extract_frames(video_path, output_dir, fps=10):
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(video_fps / fps)
    frame_count = 0
    saved = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            frame = cv2.resize(frame, (640, 640))
            cv2.imwrite(f"{output_dir}/frame_{saved:05d}.jpg", frame)
            saved += 1
        frame_count += 1
    cap.release()

# 3. Augmentation
import albumentations as A

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=15, p=0.3),
])
```

---

## Module 2 — CV Model Training & Inference (Member 2)

### Công nghệ

| Công nghệ | Version | Mục đích | Lý do chọn |
|---|---|---|---|
| **Ultralytics YOLOv8** | 8.x | Fine-tune classification + pose estimation | State-of-the-art, API đơn giản, xuất `.pt` dễ deploy |
| **YOLOv8n-cls** | — | Exercise classification (squat/deadlift/bench) | Nhẹ, phù hợp inference realtime |
| **YOLOv8n-pose** | — | 17 keypoints COCO → map MPII 15pts | Chính xác, confidence per keypoint |
| **Google Colab A100** | — | Training toàn bộ | Đủ VRAM để train YOLOv8 với batch lớn |
| **OpenCV** | 4.9+ | Keypoint visualization, skeleton overlay | |
| **NumPy** | 1.24+ | Angle computation | |
| **Matplotlib** | 3.8+ | Plot training curves, confusion matrix | |

### Training trên Colab A100

```python
# Cài đặt
!pip install ultralytics

# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

DRIVE_BASE = '/content/drive/MyDrive/fitness-system'

# Train Classification
!yolo task=classify mode=train \
    model=yolov8n-cls.pt \
    data={DRIVE_BASE}/data/cv/data.yaml \
    epochs=50 \
    imgsz=640 \
    batch=32 \
    patience=10 \
    project={DRIVE_BASE}/models/runs \
    name=yolov8_cls_v1

# Train Pose Estimation
!yolo task=pose mode=train \
    model=yolov8n-pose.pt \
    data={DRIVE_BASE}/data/cv/data.yaml \
    epochs=50 \
    imgsz=640 \
    batch=16 \
    project={DRIVE_BASE}/models/runs \
    name=yolov8_pose_v1
```

### Inference Module

```python
# src/cv/inference.py
from ultralytics import YOLO
import os

DRIVE_CLS_PATH  = os.getenv("YOLO_CLS_PATH",  "models/yolov8_cls/best.pt")
DRIVE_POSE_PATH = os.getenv("YOLO_POSE_PATH", "models/yolov8_pose/best.pt")
CONFIDENCE_THRESHOLD = 0.7

cls_model  = YOLO(DRIVE_CLS_PATH)
pose_model = YOLO(DRIVE_POSE_PATH)

def classify_exercise(image_path: str) -> dict:
    """Stage 1 — Classify squat / deadlift / bench_press."""
    result = cls_model(image_path)[0]
    conf   = float(result.probs.top1conf)
    label  = result.names[result.probs.top1]
    if conf < CONFIDENCE_THRESHOLD:
        return {"status": "rejected", "confidence": conf}
    return {"status": "ok", "label": label, "confidence": conf}

def extract_pose(image_path: str) -> dict:
    """Stage 2 — Extract 17 keypoints (x, y, conf)."""
    result    = pose_model(image_path)[0]
    keypoints = result.keypoints.xyn.tolist()   # normalized (x, y)
    confs     = result.keypoints.conf.tolist()
    return {"keypoints": keypoints, "confs": confs}
```

### Heuristic Angle Engine

```python
# src/cv/heuristics.py
import numpy as np

COCO_KEYPOINTS = {
    "left_hip": 11, "left_knee": 13, "left_ankle": 15,
    "right_hip": 12, "right_knee": 14, "right_ankle": 16,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_hip": 11, "right_hip": 12,
}

THRESHOLDS = {
    "squat_knee_angle_min":  70,   # độ — knee quá thẳng
    "squat_knee_angle_max":  160,  # độ — knee quá gập
    "spine_lean_max":        45,   # độ so với vertical
    "elbow_flare_max":       75,   # độ — bench press
}

def compute_angle(p1, p2, p3) -> float:
    """Tính góc tại p2 giữa vector p2→p1 và p2→p3."""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))

def check_squat_form(keypoints: list) -> list[dict]:
    """Trả về danh sách lỗi form cho bài squat."""
    errors = []
    knee_angle = compute_angle(
        keypoints[COCO_KEYPOINTS["left_hip"]],
        keypoints[COCO_KEYPOINTS["left_knee"]],
        keypoints[COCO_KEYPOINTS["left_ankle"]],
    )
    if knee_angle > THRESHOLDS["squat_knee_angle_max"]:
        errors.append({"joint": "knee", "issue": "Chưa xuống đủ sâu", "angle": knee_angle})
    # ... thêm các rule khác
    return errors
```

---

## Module 3 — Recommendation System (Member 3)

### Công nghệ

| Công nghệ | Version | Mục đích | Lý do chọn |
|---|---|---|---|
| **Python** | 3.10+ | Core logic | |
| **Pandas** | 2.x | Feature engineering từ lịch sử tập | |
| **NumPy** | 1.24+ | Vector operations, cosine similarity | |
| **Scikit-learn** | 1.4+ | TF-IDF, cosine similarity, SVD | Standard ML toolkit |
| **Pydantic** | 2.x | Schema validation (user profile, workout plan) | Tích hợp sẵn FastAPI |

### Phase 1 — Rule-based (Baseline)

```python
# src/recsys/rule_recommender.py
from schemas import UserProfile, ExerciseCatalog

def rule_recommend(user: UserProfile, catalog: ExerciseCatalog) -> list[str]:
    """Filter bài tập theo goal + level + equipment."""
    filtered = [
        ex for ex in catalog.exercises
        if ex.difficulty == user.level
        and ex.goal in user.goals
        and all(eq in user.available_equipment for eq in ex.required_equipment)
    ]
    return [ex.id for ex in filtered[:10]]
```

### Phase 2 — Content-based

```python
# src/recsys/content_based.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def build_content_matrix(exercises_df: pd.DataFrame):
    """TF-IDF trên exercise features."""
    feature_text = (
        exercises_df["muscle_group"] + " " +
        exercises_df["difficulty"]   + " " +
        exercises_df["equipment"]    + " " +
        exercises_df["goal"]
    )
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(feature_text)
    return matrix, vectorizer

def content_recommend(user_history: list[str], exercises_df: pd.DataFrame, top_k=10):
    matrix, _ = build_content_matrix(exercises_df)
    history_idx = exercises_df[exercises_df["id"].isin(user_history)].index
    user_vec = matrix[history_idx].mean(axis=0)
    scores = cosine_similarity(user_vec, matrix).flatten()
    top_idx = scores.argsort()[-top_k:][::-1]
    return exercises_df.iloc[top_idx]["id"].tolist()
```

### Workout Plan JSON Schema

```json
{
  "user_id": "usr_123",
  "generated_at": "2025-01-01T00:00:00Z",
  "goal": "fat_loss",
  "week": 1,
  "days": [
    {
      "day": 1,
      "focus": "Lower body",
      "exercises": [
        {
          "id": "squat_001",
          "name": "Barbell Squat",
          "sets": 4,
          "reps": "8-10",
          "rest_seconds": 90,
          "notes": "Giữ lưng thẳng, xuống song song sàn"
        }
      ]
    }
  ],
  "progressive_overload": {
    "week_2_volume_increase_pct": 5
  }
}
```

---

## Module 4 — RAG / NLP System (Member 4)

### Công nghệ

| Công nghệ | Version | Mục đích | Lý do chọn |
|---|---|---|---|
| **LangChain** | 0.2+ | Orchestration RAG pipeline, chain management | Standard framework cho RAG, dễ swap LLM |
| **ChromaDB** | 0.5+ | Vector store với HNSW index | Nhẹ, chạy local, không cần managed service |
| **BGE-M3** | — | Multilingual embedding (tiếng Việt + Anh) | Hỗ trợ 100+ ngôn ngữ, MTEB top-tier |
| **PyMuPDF (fitz)** | 1.24+ | Parse PDF sách → plain text | Nhanh, giữ layout tốt hơn pdfplumber |
| **BeautifulSoup4** | 4.12+ | Crawl và parse blog HTML | Standard web scraping |
| **Gemini API** | 1.5 Flash/Pro | LLM backbone | Chi phí thấp, context window lớn |
| **GPT-4o-mini** | — | Fallback LLM | OpenAI alternative |
| **PubMed API (Entrez)** | — | Crawl abstract + conclusion từ NCBI | Free API, XML format chuẩn |
| **Requests** | 2.31+ | HTTP calls cho PubMed API, blog crawl | |

### Knowledge Base Sources (Tier 1–3)

```
Tier 1 — Sách khoa học (nguồn gold standard):
├── Muscle & Strength Pyramid — Eric Helms (training + nutrition)
├── Sci. Principles of Hypertrophy — Dr. Mike Israetel (RP Strength)
├── Starting Strength — Mark Rippetoe (barbell basics)
├── Back Mechanic — Stuart McGill (spine biomechanics)
└── Trail Guide to the Body (anatomy & muscle groups)

Tier 2 — Research papers:
├── PubMed / NCBI  — Hypertrophy, MPS, injury studies
├── Google Scholar — Abstract + conclusion scraping
├── Semantic Scholar — Open access, API available
├── JSCR / NSCA    — Strength & conditioning journal
└── Sports Medicine — Rehabilitation & performance

Tier 3 — Supplementary:
├── RP Strength blog   — Mike Israetel, evidence-based
├── Stronger By Science — Greg Nuckols, research reviews
├── ExRx.net           — Exercise database
├── ACSM guidelines    — Official exercise prescription
└── Nội dung tiếng Việt — Blog gym.vn, Wheystore...
```

### RAG Pipeline

```python
# src/rag/chain.py
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings  import HuggingFaceEmbeddings
from langchain_google_genai          import ChatGoogleGenerativeAI
from langchain.chains                import RetrievalQA
from langchain.prompts               import PromptTemplate

EMBED_MODEL = "BAAI/bge-m3"
CHROMA_PATH = "data/chroma_db"

embeddings   = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vectorstore  = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
retriever    = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""Bạn là chuyên gia fitness dựa trên bằng chứng khoa học.
Chỉ trả lời dựa trên ngữ cảnh được cung cấp. Nếu không có thông tin, hãy nói rõ.

Ngữ cảnh: {context}
Câu hỏi: {question}

Trả lời bằng ngôn ngữ của câu hỏi (tiếng Việt hoặc tiếng Anh):"""
)

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT},
)

def ask(question: str) -> dict:
    result = rag_chain({"query": question})
    return {
        "answer":   result["result"],
        "sources":  [doc.metadata for doc in result["source_documents"]],
    }
```

### Ingestion Pipeline

```python
# src/rag/embedder.py
import fitz                          # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings  import HuggingFaceEmbeddings

splitter = RecursiveCharacterTextSplitter(
    chunk_size=750,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " "],
)

def ingest_pdf(pdf_path: str, source_name: str, vectorstore: Chroma):
    doc   = fitz.open(pdf_path)
    text  = "\n".join(page.get_text() for page in doc)
    chunks = splitter.create_documents(
        [text],
        metadatas=[{"source": source_name, "tier": 1}]
    )
    vectorstore.add_documents(chunks)
    print(f"✅ {source_name}: {len(chunks)} chunks indexed")
```

---

## Module 5 — Backend API (Member 5)

### Công nghệ

| Công nghệ | Version | Mục đích | Lý do chọn |
|---|---|---|---|
| **FastAPI** | 0.111+ | REST API framework | Async, tự sinh OpenAPI docs, tích hợp Pydantic |
| **Uvicorn** | 0.30+ | ASGI server | Hiệu năng cao, production-ready |
| **Pydantic v2** | 2.x | Request/response schema validation | Native FastAPI |
| **SQLAlchemy** | 2.x | ORM cho PostgreSQL | Async support, type-safe |
| **PostgreSQL** | 15+ | Relational DB — users, sessions, form logs | ACID, tốt cho user data có cấu trúc |
| **python-jose** | 3.x | JWT authentication | |
| **python-multipart** | — | Upload file (ảnh/video) | |
| **Docker Compose** | 3.9+ | Orchestration local dev | |

### API Endpoints

```
POST /upload          — Upload ảnh/video, trả về form feedback
                        Input:  multipart/form-data (file)
                        Output: {exercise, confidence, errors[], keypoints, skeleton_url}

POST /predict         — Predict từ base64 image
                        Input:  {image_base64, user_id}
                        Output: {label, confidence, errors[], keypoints}

POST /recommend       — Gợi ý workout plan
                        Input:  {user_id, goal, level, equipment[], available_days}
                        Output: {plan_id, days[{day, focus, exercises[]}], progressive_overload}

POST /ask             — RAG chatbot
                        Input:  {question, user_id, session_id?}
                        Output: {answer, sources[], confidence}

GET  /user/profile    — Lấy thông tin user
PUT  /user/profile    — Cập nhật user profile
GET  /user/sessions   — Lịch sử tập luyện
```

### FastAPI App Structure

```python
# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import pose, recommend, qa, user

app = FastAPI(
    title="Fitness AI API",
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pose.router,      prefix="/predict",   tags=["CV"])
app.include_router(recommend.router, prefix="/recommend", tags=["RecSys"])
app.include_router(qa.router,        prefix="/ask",       tags=["RAG"])
app.include_router(user.router,      prefix="/user",      tags=["User"])
```

### PostgreSQL Schema

```sql
-- db/schema.sql

CREATE TABLE users (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email       VARCHAR(255) UNIQUE NOT NULL,
    name        VARCHAR(100),
    goal        VARCHAR(50),    -- fat_loss | muscle_gain | strength | endurance
    level       VARCHAR(20),    -- beginner | intermediate | advanced
    weight_kg   FLOAT,
    height_cm   FLOAT,
    age         INT,
    created_at  TIMESTAMP DEFAULT NOW()
);

CREATE TABLE sessions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID REFERENCES users(id),
    exercise_label  VARCHAR(50),
    confidence      FLOAT,
    duration_sec    INT,
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE form_logs (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  UUID REFERENCES sessions(id),
    joint       VARCHAR(50),
    issue       TEXT,
    angle       FLOAT,
    severity    VARCHAR(20),  -- warning | error
    created_at  TIMESTAMP DEFAULT NOW()
);

CREATE TABLE workout_plans (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID REFERENCES users(id),
    plan_json   JSONB NOT NULL,
    week_number INT DEFAULT 1,
    created_at  TIMESTAMP DEFAULT NOW()
);
```

### Docker Compose

```yaml
# docker-compose.yml
version: "3.9"
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@db:5432/fitness
      - YOLO_CLS_PATH=${YOLO_CLS_PATH}
      - YOLO_POSE_PATH=${YOLO_POSE_PATH}
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    volumes:
      - ./models:/app/models        # weights local
      - ./data/chroma_db:/app/data/chroma_db
    depends_on:
      - db

  db:
    image: postgres:15
    environment:
      POSTGRES_DB:       fitness
      POSTGRES_USER:     postgres
      POSTGRES_PASSWORD: postgres
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./db/schema.sql:/docker-entrypoint-initdb.d/schema.sql

volumes:
  pgdata:
```

---

## Module 5 — Frontend React (Member 5)

### Công nghệ

| Công nghệ | Version | Mục đích | Lý do chọn |
|---|---|---|---|
| **React** | 18+ | UI framework | |
| **Vite** | 5+ | Build tool | Nhanh hơn CRA đáng kể |
| **Tailwind CSS** | 3.x | Styling | Utility-first, không cần viết CSS |
| **Axios** | 1.6+ | HTTP client gọi FastAPI | Interceptors, cleaner than fetch |
| **Recharts** | 2.x | Biểu đồ progress tracking | React-native charts, đơn giản |
| **React Router v6** | 6.x | Client-side routing | |
| **Canvas API** | native | Vẽ skeleton overlay lên ảnh | Không cần thư viện ngoài |

### Component Structure

```
frontend/src/
├── pages/
│   ├── FormCheck.jsx       # Upload + skeleton overlay + error list
│   ├── Recommendation.jsx  # User profile form + 7-day plan display
│   ├── Chatbot.jsx         # Chat interface + citations panel
│   └── Progress.jsx        # Session history + form error trend (Recharts)
├── components/
│   ├── SkeletonOverlay.jsx # Canvas vẽ keypoints + bones
│   ├── WorkoutCard.jsx     # Hiển thị 1 exercise trong plan
│   ├── ChatMessage.jsx     # Tin nhắn + source citations
│   └── ErrorBadge.jsx      # Form error chip (severity color)
├── services/
│   └── api.js              # Axios instance + all API calls
└── hooks/
    ├── useUpload.js        # Upload logic + loading state
    └── useChat.js          # Chat history management
```

---

## Requirements

### `requirements.txt`

```txt
# API & Web
fastapi==0.111.0
uvicorn[standard]==0.30.0
python-multipart==0.0.9
python-jose[cryptography]==3.3.0
pydantic==2.7.0

# Database
sqlalchemy==2.0.30
asyncpg==0.29.0
psycopg2-binary==2.9.9

# Computer Vision
ultralytics==8.2.0
opencv-python-headless==4.9.0.80
numpy==1.26.4

# RAG / NLP
langchain==0.2.0
langchain-community==0.2.0
langchain-google-genai==1.0.5
chromadb==0.5.0
sentence-transformers==3.0.0    # BGE-M3
pymupdf==1.24.0                 # PyMuPDF / fitz
beautifulsoup4==4.12.3
requests==2.31.0
biopython==1.83                 # PubMed Entrez API

# Recommendation
scikit-learn==1.4.2
pandas==2.2.2

# Utilities
python-dotenv==1.0.1
tqdm==4.66.4
```

### `frontend/package.json` (dependencies)

```json
{
  "dependencies": {
    "react": "^18.3.0",
    "react-dom": "^18.3.0",
    "react-router-dom": "^6.23.0",
    "axios": "^1.7.0",
    "recharts": "^2.12.0",
    "tailwindcss": "^3.4.0"
  },
  "devDependencies": {
    "vite": "^5.2.0",
    "@vitejs/plugin-react": "^4.3.0"
  }
}
```

---

## Environment Variables

```bash
# .env.example

# Google Drive (đường dẫn tuyệt đối sau khi mount Colab hoặc sync local)
YOLO_CLS_PATH=models/yolov8_cls/best.pt
YOLO_POSE_PATH=models/yolov8_pose/best.pt

# LLM API Keys (chọn 1 trong 2)
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here   # optional fallback

# Database
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/fitness

# ChromaDB
CHROMA_DB_PATH=data/chroma_db

# Security
SECRET_KEY=your_jwt_secret_key_here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440
```

---

## Compatibility Matrix

| Component | Python | CUDA | OS |
|---|---|---|---|
| Ultralytics YOLOv8 | 3.8–3.11 | 11.8 / 12.1 | Linux / macOS / Windows |
| ChromaDB | 3.8+ | N/A | All |
| FastAPI | 3.8+ | N/A | All |
| BGE-M3 (sentence-transformers) | 3.8+ | Optional | All |
| Google Colab A100 | 3.10 (default) | 12.x | Linux |

> **Lưu ý:** Training bắt buộc dùng Colab A100. Inference local không cần GPU (CPU đủ cho demo).
