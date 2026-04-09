# TEAM ASSIGNMENTS — Intelligent Fitness Recommendation & Coaching System
## Phân công công việc chi tiết theo Phase & Thành viên

> **Môi trường:** Dữ liệu lưu trên **Google Drive** · Training trên **Google Colab A100** · Annotation qua **Roboflow / CVAT**

---

## Tổng quan phân công

| | Member 1 | Member 2 | Member 3 | Member 4 | Member 5 |
|---|---|---|---|---|---|
| **Role** | CV Data Engineer | ML Engineer (CV) | ML Engineer (RecSys) | NLP / RAG Engineer | Full-Stack + Backend |
| **Layer** | Data collection, annotation & Drive pipeline | YOLO fine-tune, inference & heuristics | User profiling & Hybrid recommender | Knowledge base, RAG chain & chatbot | FastAPI + React + PostgreSQL |
| **Primary tools** | yt-dlp · Roboflow · OpenCV · Albumentations | YOLOv8-cls/pose · Colab A100 · Ultralytics | Rule-based → Content-based → Hybrid · JSON plan | PyMuPDF · BeautifulSoup · LangChain · ChromaDB | FastAPI · React · PostgreSQL · Docker |

---

## Chi tiết theo thành   

---

### 👤 Member 1 — CV Data Engineer (Dataset Preprocessing & Pipeline)

**Trách nhiệm chính:** Xử lý dataset hình ảnh tập gym từ CSV keypoints có sẵn, chuẩn hóa, tổ chức theo YOLO format và upload lên Google Drive. Toàn bộ nhóm phụ thuộc vào dataset của bạn để train model.

**🔴 UPDATE:** Dataset đã có CSV với keypoints sẵn → **KHÔNG CẦN ANNOTATION THỦ CÔNG**

#### Nhiệm vụ theo Phase

**Phase 1: ✅ Hoàn thành — Dataset đã tải từ Drive**

| Task | Status | Output |
|---|---|---|
| 1.1 | ✅ DONE | Squat dataset: 2,837 images (từ 59 videos) |
| 1.2 | ✅ DONE | CSV keypoints: `yolo_keypoints_cleaned_final.csv` (18 điểm x0,y0 → x17,y17) |
| 1.3 | ✅ DONE | Folder structure: `squat/video_id/*.jpg` |

**Phase 2: Preprocessing & Format Conversion (CURRENT PHASE)**

| Task | Mô tả | Output | Status |
|---|---|---|---|
| 2.1 | **Filter CSV** → chỉ giữ rows tương ứng với ảnh đã tải về | `squat_local_filtered.csv` | ✅ DONE |
| 2.2 | **Convert CSV → YOLO .txt** labels với normalized bbox + keypoints | `cv_dataset/all_labels/*.txt` | ✅ DONE |
| 2.3 | **Resize images** về 640×640 (or keep original nếu training tự resize) | `cv_dataset/images_resized/` | ⏳ TODO |
| 2.4 | **Split dataset** 70/15/15 (train/val/test) | `cv_dataset/train/`, `val/`, `test/` | ⏳ TODO |
| 2.5 | **Tạo data.yaml** config file cho YOLO | `cv_dataset/data.yaml` | ⏳ TODO |
| 2.6 | **Verify dataset** — check images + labels match, visualize samples | Quality report | ⏳ TODO |

**Phase 3: Augmentation & Mở rộng**

| Task | Mô tả | Output |
|---|---|---|
| 3.1 | Augmentation bằng **Albumentations**: flip, rotate, brightness | Colab notebook |
| 3.2 | Hỗ trợ M2 nếu accuracy chưa đạt: bổ sung data theo class yếu | Thêm samples vào Drive |
| 3.3 | Quality filter: loại ảnh có detection confidence < 0.7 | `preprocessing/quality_filter.py` |

**Phase 4: Báo cáo & Hoàn thiện**

| Task | Mô tả | Output |
|---|---|---|
| 4.1 | Thống kê dataset cuối (số mẫu mỗi class, phân bố) | `reports/dataset_stats.md` |
| 4.2 | Viết phần báo cáo: Data Collection & Annotation Methodology | Báo cáo cuối kỳ |

**Deliverables:**
- [ ] ≥ 3 loại bài tập (squat / deadlift / bench) với đủ data
- [ ] `data.yaml` chuẩn YOLO pose format
- [ ] Annotated dataset trên Roboflow, exported `.txt`
- [ ] Drive folder structure sạch đúng layout
- [ ] Train/val/test split 70/15/15

---

### 👤 Member 2 — ML Engineer (Computer Vision — YOLO Fine-tune & Inference)

**Trách nhiệm chính:** Fine-tune YOLOv8 classification và pose estimation trên Colab A100, viết inference module và heuristic angle engine cho form feedback.

#### Nhiệm vụ theo Phase

**Phase 1: EDA & Baseline Setup**

| Task | Mô tả | Output |
|---|---|---|
| 1.1 | Mount Google Drive trên Colab A100, load dataset từ M1 | Colab notebook |
| 1.2 | EDA: visualize keypoints, phân bố class, kiểm tra imbalance | `notebooks/eda_pose.ipynb` |
| 1.3 | Cài Ultralytics, thử train nhanh 5 epoch kiểm tra pipeline | `notebooks/yolo_sanity_check.ipynb` |

**Phase 2: Model Training trên Colab A100**

| Task | Mô tả | Output |
|---|---|---|
| 2.1 | Fine-tune **YOLOv8n-cls** (classification): label squat/deadlift/bench | `Drive:/models/yolov8_cls/best.pt` |
| 2.2 | Fine-tune **YOLOv8n-pose** (pose estimation): 17 keypoints COCO → map sang MPII 15pts | `Drive:/models/yolov8_pose/best.pt` |
| 2.3 | Monitor **mAP, val/AP loss** — log ra `Drive:/models/runs/` | Training logs |
| 2.4 | Thiết lập **confidence threshold > 0.7** cho rejection flow | Config trong `inference.py` |
| 2.5 | Benchmark latency: đảm bảo inference < 150ms/frame | `benchmarks/inference_speed.md` |
| 2.6 | Đánh giá Accuracy ≥ 85% — nếu chưa: yêu cầu thêm data từ M1 | `reports/pose_eval_phase2.md` |

**Phase 3: Inference Module & Heuristics**

| Task | Mô tả | Output |
|---|---|---|
| 3.1 | Viết `inference.py`: load weights từ Drive, chạy YOLO prediction | `src/cv/inference.py` |
| 3.2 | Viết **heuristic angle engine**: tính góc đầu gối/cột sống/cùi chỏ, so threshold | `src/cv/heuristics.py` |
| 3.3 | Triển khai **keypoint visualization**: vẽ overlay skeleton lên ảnh/frame | `src/cv/visualizer.py` |
| 3.4 | Viết **Form Error Report**: output JSON danh sách lỗi form | `src/cv/form_reporter.py` |
| 3.5 | Test end-to-end: ảnh/video → YOLO → heuristics → feedback JSON | Integration test |

**Phase 4: Đánh giá & Báo cáo**

| Task | Mô tả | Output |
|---|---|---|
| 4.1 | Benchmark cuối: Accuracy (cls), mAP (pose), Latency | `reports/pose_final_benchmark.md` |
| 4.2 | Viết phần báo cáo: YOLO Fine-tuning, Heuristics & Real-time System | Báo cáo cuối kỳ |

**Deliverables:**
- [ ] `yolov8_cls/best.pt` — Accuracy ≥ 85%
- [ ] `yolov8_pose/best.pt` — mAP documented
- [ ] `inference.py` + `heuristics.py` + `visualizer.py`
- [ ] Form error JSON schema
- [ ] Latency benchmark < 150ms

---

### 👤 Member 3 — ML Engineer (Recommendation System)

**Trách nhiệm chính:** Xây dựng hệ thống gợi ý bài tập cá nhân hóa theo thể trạng, mục tiêu, lịch sử tập luyện. Tiến hành theo 3 phase: Rule-based → Content-based → Hybrid.

#### Nhiệm vụ theo Phase

**Phase 1: User Profiling & Schema**

| Task | Mô tả | Output |
|---|---|---|
| 1.1 | Thiết kế **user schema**: goal, weight, height, level, age, equipment | `schemas/user_schema.json` |
| 1.2 | Thiết kế **exercise schema**: muscle_group, difficulty, equipment, duration | `schemas/exercise_schema.json` |
| 1.3 | Feature engineering từ lịch sử tập: completion_rate, avg_intensity, fatigue_score | `src/recsys/feature_engineering.py` |
| 1.4 | Tạo mock dataset người dùng (≥ 500 users) để test system | `data/mock_users.json` |

**Phase 2: Recommendation Models**

| Task | Mô tả | Output |
|---|---|---|
| 2.1 | **Rule-based recommender** (Phase 1 baseline): filter theo goal + level + equipment | `src/recsys/rule_recommender.py` |
| 2.2 | **Content-based filtering** (Phase 2): TF-IDF trên exercise features, cosine similarity | `src/recsys/content_based.py` |
| 2.3 | **Hybrid model** (Phase 3): kết hợp rule + content với trọng số | `src/recsys/hybrid_recommender.py` |
| 2.4 | Đánh giá: Precision@K, Recall@K | `notebooks/rec_evaluation.ipynb` |

**Phase 3: Workout Plan Generator**

| Task | Mô tả | Output |
|---|---|---|
| 3.1 | Xây dựng **7-day Workout Plan Generator** từ top-K recommendations | `src/recsys/plan_generator.py` |
| 3.2 | Triển khai **Progressive Overload Logic**: tăng dần intensity theo tuần | `src/recsys/progressive_overload.py` |
| 3.3 | Thiết kế **JSON schema chuẩn** cho workout plan (sync với M5) | `schemas/workout_plan_schema.json` |
| 3.4 | API endpoint `POST /recommend` trả về workout plan JSON | `app/routers/recommend.py` |

**Phase 4: Hoàn thiện**

| Task | Mô tả | Output |
|---|---|---|
| 4.1 | So sánh rule-based vs content-based vs hybrid | `reports/rec_comparison.md` |
| 4.2 | Viết phần báo cáo: Recommendation System Architecture & Evaluation | Báo cáo cuối kỳ |

**Deliverables:**
- [ ] Rule-based baseline recommender
- [ ] Content-based + Hybrid recommender
- [ ] **Workout Plan Generator** 7 ngày với progressive overload
- [ ] `workout_plan_schema.json` chuẩn hóa (sync M5)
- [ ] API `/recommend` hoạt động
- [ ] Evaluation report (Precision@K)

---

### 👤 Member 4 — NLP / RAG Engineer (Fitness QA & Knowledge Base)

**Trách nhiệm chính:** Xây dựng knowledge base fitness từ Tier 1–3 nguồn, embed vào ChromaDB, xây dựng LangChain RAG chain với Gemini/GPT-4o-mini. Hỗ trợ tiếng Việt.

#### Nhiệm vụ theo Phase

**Phase 1: Knowledge Base Construction**

| Task | Mô tả | Output |
|---|---|---|
| 1.1 | Thu thập **Tier 1 — Sách PDF**: Helms (Muscle & Strength Pyramid), McGill (Back Mechanic), Rippetoe (Starting Strength) | `Drive:/data/rag/raw/books/` |
| 1.2 | Crawl **Tier 2 — PubMed API**: abstract + conclusion XML (hypertrophy, MPS, injury studies) | `Drive:/data/rag/raw/pubmed/` |
| 1.3 | Crawl **Tier 3 — Blog & ACSM**: RP Strength, Stronger By Science, gym.vn, Wheystore bằng BeautifulSoup | `Drive:/data/rag/raw/blogs/` |
| 1.4 | Viết **Document Parser**: PyMuPDF cho PDF, BeautifulSoup cho HTML → plain `.txt` | `src/rag/document_loader.py` |
| 1.5 | Chunking: **500–1000 tokens, overlap 100** | `src/rag/chunker.py` |
| 1.6 | Embedding bằng **BGE-M3** (multilingual, hỗ trợ tiếng Việt) → lưu ChromaDB HNSW index | `src/rag/embedder.py` + `src/rag/vector_store.py` |

**Phase 2: RAG Chain**

| Task | Mô tả | Output |
|---|---|---|
| 2.1 | Xây dựng **LangChain RAG pipeline**: query → ChromaDB retrieve → augment → Gemini/GPT-4o-mini | `src/rag/chain.py` |
| 2.2 | Tích hợp **LLM connector**: Gemini API hoặc GPT-4o-mini via API key | `src/rag/llm_connector.py` |
| 2.3 | Test với 50 câu hỏi mẫu (tiếng Việt + tiếng Anh) | `tests/test_qa.py` |
| 2.4 | Đánh giá: Context Relevance, Answer Faithfulness | `reports/rag_eval_phase2.md` |

**Phase 3: Advanced Features**

| Task | Mô tả | Output |
|---|---|---|
| 3.1 | **Myth-busting logic**: fact-check bro-science queries | `src/rag/myth_buster.py` |
| 3.2 | **Safe fallback**: no-context guard khi không tìm được chunk liên quan | `src/rag/guardrail.py` |
| 3.3 | **Citations panel**: trả về source_docs kèm câu trả lời | Response schema update |
| 3.4 | API endpoint `POST /ask` → answer + citations JSON | `app/routers/qa.py` |

**Phase 4: Đánh giá & Báo cáo**

| Task | Mô tả | Output |
|---|---|---|
| 4.1 | Đánh giá cuối: ≥ 85% câu hỏi đúng ngữ cảnh | `reports/rag_final_eval.md` |
| 4.2 | Viết phần báo cáo: RAG Architecture, Knowledge Base & Evaluation | Báo cáo cuối kỳ |

**Deliverables:**
- [ ] KB từ ≥ 3 nguồn Tier 1 + Tier 2 + Tier 3
- [ ] ChromaDB HNSW index đầy đủ (BGE-M3 embed)
- [ ] LangChain RAG chain hoạt động (tiếng Việt + tiếng Anh)
- [ ] Myth-busting + Safe fallback
- [ ] API `/ask` — ≥ 85% đúng ngữ cảnh

---

### 👤 Member 5 — Full-Stack + Backend (FastAPI + React + PostgreSQL)

**Trách nhiệm chính:** Xây dựng toàn bộ backend API và frontend web app. Tích hợp tất cả module: CV (M2), RecSys (M3), RAG (M4). Là điểm kết nối cuối cùng của hệ thống.

**Stack:** FastAPI · React (Vite) · PostgreSQL · Docker Compose

#### Nhiệm vụ theo Phase

**Phase 1: Backend Skeleton + Database**

| Task | Mô tả | Output |
|---|---|---|
| 1.1 | Thiết kế **PostgreSQL schema**: `users`, `sessions`, `workout_plans`, `form_logs` | `db/schema.sql` |
| 1.2 | Khởi tạo **FastAPI project** với cấu trúc chuẩn | `app/` |
| 1.3 | Viết API skeleton: `/upload`, `/predict`, `/recommend`, `/ask`, `/user/profile` | `app/routers/` |
| 1.4 | Cài CORS, JWT authentication, middleware logging | `app/middleware/` |
| 1.5 | **Docker Compose**: FastAPI + PostgreSQL + ChromaDB | `docker-compose.yml` |
| 1.6 | Prototype **React frontend**: layout cơ bản 4 trang | `frontend/src/pages/` |

**Phase 2: Frontend & DB Integration**

| Task | Mô tả | Output |
|---|---|---|
| 2.1 | Trang **Upload & Form Check**: upload ảnh/video, preview, hiển thị skeleton overlay + error report | `pages/FormCheck.jsx` |
| 2.2 | Trang **Recommendation**: form user profile → hiển thị workout plan 7 ngày | `pages/Recommendation.jsx` |
| 2.3 | Trang **Fitness Chatbot**: chat interface + citations panel gọi `/ask` API | `pages/Chatbot.jsx` |
| 2.4 | Trang **Progress Tracking**: biểu đồ lịch sử tập (Recharts) | `pages/Progress.jsx` |
| 2.5 | Kết nối backend từ React (axios) | `frontend/src/services/api.js` |

**Phase 3: Full Integration**

| Task | Mô tả | Output |
|---|---|---|
| 3.1 | Tích hợp YOLO model M2 vào `/predict` endpoint (load weights từ Drive) | `app/routers/pose.py` |
| 3.2 | Tích hợp RecSys M3 vào `/recommend` endpoint | `app/routers/recommend.py` |
| 3.3 | Tích hợp RAG chain M4 vào `/ask` endpoint | `app/routers/qa.py` |
| 3.4 | Lưu session log, form errors vào PostgreSQL | `app/services/session_service.py` |
| 3.5 | End-to-end test: upload → form check → plan → chatbot | Integration tests |

**Phase 4: Hoàn thiện & Demo**

| Task | Mô tả | Output |
|---|---|---|
| 4.1 | Deploy app (Docker local cho demo) | Deployment guide |
| 4.2 | Performance test: API response time, concurrent users | `benchmarks/api_perf.md` |
| 4.3 | Demo script: video walkthrough các tính năng chính | Demo video |
| 4.4 | Viết phần báo cáo: System Architecture, API Design, Deployment | Báo cáo cuối kỳ |

**Deliverables:**
- [ ] **FastAPI backend** với 5 endpoints chính
- [ ] **PostgreSQL schema** đầy đủ
- [ ] **React web app** 4 trang
- [ ] Tích hợp đầy đủ 3 module AI (CV + RecSys + RAG)
- [ ] Docker Compose cho development
- [ ] Demo hoàn chỉnh

---

## Timeline tổng thể (4 Phases)

```
Phase │ M1 (CV Data)           │ M2 (YOLO/CV)            │ M3 (RecSys)             │ M4 (RAG/NLP)            │ M5 (Full-stack)
──────┼─────────────────────── ┼─────────────────────────┼─────────────────────────┼─────────────────────────┼──────────────────────────
      │ MPII + Leeds download  │ Mount Drive trên Colab  │ User schema design      │ Collect PDF + crawl blog│ PostgreSQL schema
      │ YouTube crawl (yt-dlp) │ EDA keypoints           │ Feature engineering     │ PyMuPDF document parse  │ FastAPI skeleton
  1   │ Tự chụp gym            │ YOLO sanity check       │ Mock dataset 500 users  │ Chunking 500-1000 tokens│ Docker Compose
      │ Resize 640×640         │ Baseline config         │ Rule-based rec (v1)     │ BGE-M3 embed → Chroma   │ React layout 4 trang
──────┼────────────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┼──────────────────────────
      │ Roboflow annotation    │ YOLOv8-cls fine-tune    │ Content-based filtering │ LangChain RAG pipeline  │ React pages (4)
      │ Export YOLO pose .txt  │ YOLOv8-pose fine-tune   │ Hybrid recommender      │ LLM integration         │ API integration
  2   │ data.yaml + splits     │ Accuracy ≥ 85% target   │ Evaluation Precision@K  │ 50-question test        │ DB connection
      │ Albumentations augment │ Latency benchmark       │ workout_plan_schema.json│ RAGAS evaluation        │ Auth + middleware
──────┼────────────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┼──────────────────────────
      │ Quality filter pass    │ inference.py            │ Workout plan gen 7 ngày │ Myth-busting            │ Tích hợp CV M2
      │ Bổ sung data nếu cần  │ heuristics.py (angles)  │ Progressive overload    │ Safe fallback guardrail │ Tích hợp RecSys M3
  3   │ Dataset final stats    │ visualizer.py skeleton  │ Fatigue score           │ Citations panel         │ Tích hợp RAG M4
      │                        │ Form error JSON         │ /recommend API          │ /ask API                │ Session logging
──────┼────────────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┼──────────────────────────
      │ dataset_stats.md       │ Final benchmark report  │ rec_comparison.md       │ rag_final_eval.md       │ Deploy + demo video
  4   │ Write report section   │ Write report section    │ Write report section    │ Write report section    │ API perf benchmark
      │ Review + submit        │ Review + submit         │ Review + submit         │ Review + submit         │ Write report
```

---

## Dependencies giữa các Member

### Sơ đồ tổng quan

```
┌─────────────────────────────────────────────────────────────────────┐
│   M1 (CV Data Engineer) — CRITICAL PATH                             │
│   Tất cả CV model đều phụ thuộc dataset từ M1                       │
│   Phải xong data.yaml + annotated dataset trước khi Phase 2 bắt đầu │
└──────┬──────────────────────────────────────────────────────────────┘
       │  pose_sequences · data.yaml · exercise_schema.json
       ▼
┌──────────────────────────────────────────────────────────────────┐
│   M2 (ML/CV)         M3 (RecSys)          M4 (RAG/NLP)          │
│   YOLOv8 fine-tune   Hybrid Recommender   LangChain + Chroma     │
│   inference.py       plan_generator.py    chain.py               │
└──────┬───────────────────┬──────────────────────┬────────────────┘
       │ best.pt           │ workout_plan JSON     │ RAG pipeline
       │ form_error JSON   │ /recommend API        │ /ask API
       └───────────────────┴──────────────────────►│
                                                   │
                                        ┌──────────▼──────────────┐
                                        │   M5 (Full-Stack)       │
                                        │   FastAPI + React       │
                                        │   PostgreSQL            │
                                        └─────────────────────────┘
```

---

### Ma trận phụ thuộc chi tiết

> ✅ = phụ thuộc trực tiếp, cần chờ | ⚡ = soft dependency, có workaround | — = không phụ thuộc

| Người nhận ↓ / Người cung cấp → | **M1** | **M2** | **M3** | **M4** | **M5** |
|---|---|---|---|---|---|
| **M1** | — | — | — | — | — |
| **M2** | ✅ Cần `data.yaml` + annotated dataset để bắt đầu train | — | — | — | ⚡ DB lưu session (dùng JSON local tạm) |
| **M3** | ✅ Cần `exercise_schema.json` chuẩn hóa | — | — | — | ⚡ DB user profile (dùng mock JSON tạm) |
| **M4** | — | — | ⚡ Danh mục bài tập để giới hạn KB scope | — | — |
| **M5** | — | ✅ Cần `inference.py` + `best.pt` để gọi `/predict` | ✅ Cần `hybrid_recommender` + `workout_plan_schema.json` | ✅ Cần `chain.py` + ChromaDB path để gọi `/ask` | — |

---

### Bảng Hard Blockers — Không làm được nếu chưa có

> **Hard blocker** = bắt buộc phải chờ, không có cách workaround.

| Người bị block | Chờ ai | Chờ gì cụ thể | Deadline cần có | Nếu trễ thì làm gì |
|---|---|---|---|---|
| M2 | **M1** | `data.yaml` + annotated dataset YOLO pose `.txt` | Cuối Phase 1 | M2 dùng dataset công khai (COCO Pose) tạm thời |
| M3 | **M1** | `exercise_schema.json` chuẩn hóa class | Cuối Phase 1 | M3 tự thiết kế schema draft, sync M1 sau |
| M5 | **M2** | `inference.py` + weights để tích hợp `/predict` | Đầu Phase 3 | M5 build UI với mock pose response trước |
| M5 | **M3** | `hybrid_recommender` trả về workout plan JSON | Đầu Phase 3 | M5 dùng hardcoded workout plan tạm |
| M5 | **M4** | `chain.py` + ChromaDB path để tích hợp `/ask` | Giữa Phase 2 | M5 mockup chatbot với dummy response |

---

### Bảng Soft Dependencies — Nên có nhưng không bắt buộc dừng

| Ai | Cần từ ai | Mô tả | Workaround |
|---|---|---|---|
| M3 | M2 | Biết loại bài tập YOLO nhận diện được để align schema | M3 dùng danh sách cố định từ `exercise_schema.json` |
| M4 | M3 | Biết danh mục bài tập để giới hạn scope KB | M4 cover toàn bộ fitness KB |
| M5 | M3 | Biết format JSON workout plan để render UI | M5 thiết kế UI theo `workout_plan_schema.json` đã thống nhất |
| M2 | M5 | PostgreSQL để lưu session log + form errors | M2 lưu local JSON tạm trong quá trình dev |

---

### Giao thức Handoff (Bàn giao giữa các thành viên)

Mỗi khi hoàn thành phần mình để người khác dùng, **phải thực hiện đủ 3 bước**:

```
Bước 1: Tạo PR vào branch develop, tag người nhận vào để review
Bước 2: Ghi vào file HANDOFF_LOG.md:
         - Tên artifact (file, API endpoint, schema)
         - Đường dẫn chính xác (Google Drive path hoặc local path)
         - Format input/output (JSON với fields gì)
         - Ví dụ request/response cụ thể
Bước 3: Ping người nhận trên nhóm chat, confirm họ đã test được
```

**Các mốc handoff cụ thể:**

| Phase | Từ | Đến | Artifact | Format / Path |
|---|---|---|---|---|
| Cuối Phase 1 | M1 | M2 | Annotated dataset YOLO pose | `Drive:/data/cv/annotated/` + `data.yaml` |
| Cuối Phase 1 | M1 | M3, M5 | Exercise metadata schema | `schemas/exercise_schema.json` |
| Cuối Phase 2 | M2 | M5 | YOLO weights + inference wrapper | `Drive:/models/yolov8_cls/best.pt` + `src/cv/inference.py` |
| Cuối Phase 2 | M3 | M5 | Workout plan schema + recommender | `schemas/workout_plan_schema.json` + `src/recsys/hybrid_recommender.py` |
| Giữa Phase 2 | M4 | M5 | RAG chain + ChromaDB | `src/rag/chain.py` + ChromaDB path + API spec |
| Đầu Phase 3 | M5 | M2, M3, M4 | PostgreSQL schema live | DB connection string + table docs |

---

## Google Drive Folder Structure (toàn team dùng chung)

```
MyDrive/
└── fitness-system/
    ├── data/
    │   ├── cv/
    │   │   ├── raw/
    │   │   │   ├── mpii/            # M1 — MPII dataset .mat
    │   │   │   ├── youtube/         # M1 — frames từ yt-dlp
    │   │   │   ├── leeds/           # M1 — Leeds Sport Pose
    │   │   │   └── self_captured/   # M1 — tự chụp gym
    │   │   ├── processed/           # M1 — sau resize 640×640
    │   │   ├── annotated/           # M1 — YOLO pose .txt (Roboflow export)
    │   │   ├── splits/              # M1 — train/ val/ test/
    │   │   └── data.yaml            # M1 → M2
    │   └── rag/
    │       ├── raw/
    │       │   ├── books/           # M4 — PDF sách
    │       │   ├── pubmed/          # M4 — PubMed XML
    │       │   └── blogs/           # M4 — Blog crawl
    │       └── processed/           # M4 — plain .txt sau parsing
    ├── models/
    │   ├── yolov8_cls/
    │   │   └── best.pt              # M2 → M5
    │   ├── yolov8_pose/
    │   │   └── best.pt              # M2 → M5
    │   └── runs/                    # M2 — training logs
    └── notebooks/
        ├── cv_preprocess.ipynb      # M1 — Colab A100
        ├── yolo_cls_training.ipynb  # M2 — Colab A100
        ├── yolo_pose_training.ipynb # M2 — Colab A100
        └── rag_ingest.ipynb         # M4 — Colab
```

---

## Shared Responsibilities (Tất cả thành viên)

| Trách nhiệm chung | Mô tả |
|---|---|
| **Code review** | Mỗi PR cần ít nhất 1 người review trước khi merge vào `develop` |
| **Unit tests** | Mỗi module phải có file `tests/test_*.py` cơ bản |
| **Git workflow** | Làm việc trên `feature/<tên>` branch, không push thẳng vào `main` |
| **Documentation** | Mỗi function phải có docstring, mỗi module có README ngắn |
| **Drive hygiene** | Không xóa file người khác, đặt tên file rõ ràng theo convention |
| **Colab notebooks** | Commit `.ipynb` vào Git sau mỗi training run quan trọng |
| **Meeting hàng tuần** | Báo cáo tiến độ, blockers, điều chỉnh kế hoạch |
| **Báo cáo cuối kỳ** | Mỗi người viết section của mình (xem Phase 4 tasks) |

---

## Cấu trúc thư mục dự án (Target — Local Repo)

```
fitness-recommendation-system/
├── src/
│   ├── cv/
│   │   ├── inference.py          # M2 — YOLO inference, load weights từ Drive
│   │   ├── heuristics.py         # M2 — angle computation
│   │   ├── visualizer.py         # M2 — keypoint overlay
│   │   └── form_reporter.py      # M2 — form error JSON
│   ├── recsys/
│   │   ├── feature_engineering.py# M3
│   │   ├── rule_recommender.py   # M3
│   │   ├── content_based.py      # M3
│   │   ├── hybrid_recommender.py # M3
│   │   ├── plan_generator.py     # M3
│   │   └── progressive_overload.py# M3
│   └── rag/
│       ├── document_loader.py    # M4 — PyMuPDF + BeautifulSoup
│       ├── chunker.py            # M4 — 500-1000 tokens, overlap 100
│       ├── embedder.py           # M4 — BGE-M3
│       ├── vector_store.py       # M4 — ChromaDB HNSW
│       ├── chain.py              # M4 — LangChain RAG chain
│       ├── llm_connector.py      # M4 — Gemini / GPT-4o-mini
│       ├── myth_buster.py        # M4
│       └── guardrail.py          # M4
├── app/
│   ├── main.py                   # M5
│   ├── routers/
│   │   ├── pose.py               # M5 — gọi M2 inference
│   │   ├── recommend.py          # M5 — gọi M3 recommender
│   │   ├── qa.py                 # M5 — gọi M4 RAG chain
│   │   └── user.py               # M5 — user profile CRUD
│   ├── services/
│   │   └── session_service.py    # M5
│   └── middleware/               # M5 — CORS, JWT, logging
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── FormCheck.jsx     # M5 — upload + skeleton overlay
│   │   │   ├── Recommendation.jsx# M5 — workout plan 7 ngày
│   │   │   ├── Chatbot.jsx       # M5 — chat + citations
│   │   │   └── Progress.jsx      # M5 — progress tracking
│   │   └── services/
│   │       └── api.js            # M5
│   └── package.json
├── db/
│   └── schema.sql                # M5
├── schemas/
│   ├── exercise_schema.json      # M1 → M3
│   ├── user_schema.json          # M3
│   └── workout_plan_schema.json  # M3 → M5
├── notebooks/                    # Colab notebooks (committed sau training)
├── reports/                      # Mỗi member viết section
├── benchmarks/
├── tests/
│   └── test_qa.py                # M4
├── docker-compose.yml            # M5
├── requirements.txt
├── README.md
├── TECH_STACK.md
├── TEAM_ASSIGNMENTS.md           # file này
└── HANDOFF_LOG.md                # cập nhật sau mỗi handoff

```

---

## Checklist Phase Gate

### ✅ Phase 1 Done khi:
- [ ] Annotated CV dataset trên Drive, `data.yaml` sẵn sàng (M1)
- [ ] `exercise_schema.json` đã thống nhất giữa M1, M3, M5 (M1)
- [ ] Knowledge base embedded vào ChromaDB (M4)
- [ ] FastAPI skeleton 5 endpoints chạy được (M5)
- [ ] PostgreSQL schema khởi tạo xong (M5)
- [ ] YOLO sanity check 5 epoch không lỗi (M2)

### ✅ Phase 2 Done khi:
- [ ] YOLOv8-cls + YOLOv8-pose fine-tune xong, Accuracy ≥ 85% (M2)
- [ ] Hybrid recommender trả về workout plan đúng schema (M3)
- [ ] RAG pipeline trả lời đúng câu hỏi cơ bản tiếng Việt + tiếng Anh (M4)
- [ ] React frontend 4 trang kết nối được API (M5)

### ✅ Phase 3 Done khi:
- [ ] End-to-end: Upload ảnh → YOLO detect → Form feedback → Recommend → Chatbot (tất cả)
- [ ] `heuristics.py` tính góc đúng ≥ 3 rule (M2)
- [ ] Workout plan 7 ngày với progressive overload (M3)
- [ ] Guardrail + Myth-busting hoạt động (M4)
- [ ] Full integration test pass (M5)

### ✅ Phase 4 Done khi:
- [ ] Demo hoàn chỉnh với dữ liệu thực
- [ ] Báo cáo cuối kỳ: mỗi người đã viết section của mình
- [ ] Benchmark latency < 150ms documented
- [ ] Tất cả deliverables đã tick xong
