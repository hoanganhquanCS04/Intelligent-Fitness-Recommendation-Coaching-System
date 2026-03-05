# TEAM ASSIGNMENTS — Intelligent Fitness Recommendation & Coaching System
## Phân công công việc chi tiết theo Phase & Thành viên

---

## Tổng quan phân công

| | Member 1 | Member 2 | Member 3 | Member 4 | Member 5 |
|---|---|---|---|---|---|
| **Role** | Data Engineer | ML Engineer (CV) | ML Engineer (RecSys) | NLP Engineer | Full-Stack + Backend |
| **Layer** | Data & Video Pipeline | Pose Recognition | Recommendation Engine | QA + Knowledge Base | API + App + DB |
| **Primary** | Video crawl + Frame extraction + Pose sequences | MediaPipe + MLP + LSTM/GRU | User Profiling + Hybrid Recommender | LangChain + Chroma + RAG | FastAPI + React + PostgreSQL |

---

## Chi tiết theo thành viên

---

### 👤 Member 1 — Data Engineer (Video & Dataset Pipeline)

**Trách nhiệm chính:** Xây dựng pipeline dữ liệu video và metadata cho bài tập. Toàn bộ nhóm phụ thuộc vào dataset của bạn để train và test model.

#### Nhiệm vụ theo Phase

**Phase 1: Dataset Collection**

| Task | Mô tả | Output |
|---|---|---|
| 1.1 | Thu thập video workout từ YouTube API (squat, push-up, plank, lunge...) | `data/raw/videos/` |
| 1.2 | Crawl metadata bài tập (muscle group, level, equipment) | `data/metadata.json` |
| 1.3 | Chuẩn hóa nhãn động tác (squat, push-up, deadlift...) | `schemas/exercise_schema.json` |
| 1.4 | Extract frame từ video bằng OpenCV (fps=10) | `data/frames/` |
| 1.5 | Tạo train/val/test split (70/15/15) | `data/splits/` |

**Phase 2: Preprocessing Pipeline**

| Task | Mô tả | Output |
|---|---|---|
| 2.1 | Tích hợp MediaPipe Pose để extract 33 keypoints từ frame | `preprocessing/pose_extractor.py` |
| 2.2 | Chuẩn hóa keypoints: normalize tọa độ theo bounding box | `preprocessing/normalize_pose.py` |
| 2.3 | Tạo sliding window sequences (window=30 frames, stride=10) | `preprocessing/sequence_builder.py` |
| 2.4 | Lưu pose sequences dạng NumPy array | `data/pose_sequences/` |
| 2.5 | Kiểm tra chất lượng: loại bỏ frame detection confidence < 0.7 | `preprocessing/quality_filter.py` |

**Phase 3: Hỗ trợ & Mở rộng**

| Task | Mô tả | Output |
|---|---|---|
| 3.1 | Data augmentation: flip, scale, time warp cho pose sequences | `preprocessing/augment.py` |
| 3.2 | Hỗ trợ M2 fine-tune dataset nếu accuracy chưa đạt | Thêm data theo class yếu |
| 3.3 | Xây dựng user feedback dataset (correct/incorrect form samples) | `data/form_labels/` |

**Phase 4: Báo cáo & Hoàn thiện**

| Task | Mô tả | Output |
|---|---|---|
| 4.1 | Thống kê dataset cuối (số mẫu mỗi class, phân bố) | `reports/dataset_stats.md` |
| 4.2 | Viết phần báo cáo: Data Collection & Preprocessing Methodology | Báo cáo cuối kỳ |

**Deliverables:**
- [ ] Ít nhất **10 loại bài tập** với đủ video mẫu
- [ ] **≥ 50K pose sequences** NumPy (33 keypoints × 30 frames)
- [ ] `pose_extractor.py` tích hợp MediaPipe
- [ ] Dataset split chuẩn (train/val/test)
- [ ] `exercise_schema.json` chuẩn hóa nhãn

---

### 👤 Member 2 — ML Engineer (Computer Vision – Pose Recognition)

**Trách nhiệm chính:** Xây dựng và tối ưu mô hình nhận diện & phân loại động tác tập luyện. Phát hiện lỗi form và đếm repetition trong video real-time.

#### Nhiệm vụ theo Phase

**Phase 1: Baseline Pose Classification**

| Task | Mô tả | Output |
|---|---|---|
| 1.1 | Load pose sequences từ M1, EDA (visualize keypoints) | `notebooks/eda_pose.ipynb` |
| 1.2 | Train **MLP baseline** classifier (input: 33×3 vector) | `models/mlp_pose.pth` |
| 1.3 | Đánh giá Accuracy / F1 / Confusion Matrix | `reports/pose_eval_phase1.md` |
| 1.4 | Phân tích lỗi: class nào bị nhầm nhiều nhất? | `notebooks/error_analysis.ipynb` |

**Phase 2: Sequence Modeling**

| Task | Mô tả | Output |
|---|---|---|
| 2.1 | Xây dựng **LSTM** classifier cho chuỗi pose (input: 30×99) | `models/lstm_pose.pth` |
| 2.2 | Thử nghiệm **GRU** và **CNN-1D** so sánh với LSTM | `notebooks/model_comparison.ipynb` |
| 2.3 | Hyperparameter tuning: hidden_dim, n_layers, dropout | `notebooks/hyperparam_tuning.ipynb` |
| 2.4 | Đánh giá Accuracy ≥ 85% — nếu chưa đạt: yêu cầu thêm data từ M1 | `reports/pose_eval_phase2.md` |
| 2.5 | Benchmark latency: đảm bảo inference < 150ms/frame | `benchmarks/inference_speed.md` |

**Phase 3: Real-time Integration**

| Task | Mô tả | Output |
|---|---|---|
| 3.1 | Viết API endpoint `POST /predict_pose` nhận video stream | `api/pose_router.py` |
| 3.2 | Triển khai **Repetition Counter** (peak detection trên keypoint velocity) | `models/rep_counter.py` |
| 3.3 | Triển khai **Form Checker** (rule-based: góc khớp, độ lệch trục) | `models/form_checker.py` |
| 3.4 | Test end-to-end: camera → MediaPipe → model → feedback UI | Integration test |

**Phase 4: Đánh giá & Báo cáo**

| Task | Mô tả | Output |
|---|---|---|
| 4.1 | Benchmark cuối: Accuracy, Latency, Rep Counter accuracy | `reports/pose_final_benchmark.md` |
| 4.2 | Viết phần báo cáo: Pose Recognition Model & Real-time System | Báo cáo cuối kỳ |

**Deliverables:**
- [ ] MLP baseline model (Accuracy documented)
- [ ] **LSTM/GRU model** — Accuracy ≥ 85%
- [ ] **Rep counter** hoạt động đúng ≥ 90% trường hợp
- [ ] **Form checker** với ≥ 3 rule phát hiện sai form
- [ ] API `/predict_pose` latency < 150ms

---

### 👤 Member 3 — ML Engineer (Recommendation System)

**Trách nhiệm chính:** Xây dựng hệ thống gợi ý bài tập cá nhân hóa theo thể trạng, mục tiêu, và lịch sử tập luyện của người dùng.

#### Nhiệm vụ theo Phase

**Phase 1: User Profiling**

| Task | Mô tả | Output |
|---|---|---|
| 1.1 | Thiết kế **user schema**: goal, weight, height, level, age, equipment | `schemas/user_schema.json` |
| 1.2 | Thiết kế **exercise schema**: muscle_group, difficulty, equipment, duration | `schemas/exercise_schema.json` |
| 1.3 | Feature engineering từ lịch sử tập: completion_rate, avg_intensity, fatigue_score | `preprocessing/feature_engineering.py` |
| 1.4 | Tạo mock dataset người dùng (≥ 500 users) để test system | `data/mock_users.json` |

**Phase 2: Recommendation Models**

| Task | Mô tả | Output |
|---|---|---|
| 2.1 | **Rule-based recommender** (baseline): filter theo goal + level + equipment | `models/rule_recommender.py` |
| 2.2 | **Content-based filtering**: TF-IDF trên exercise features, cosine similarity | `models/content_based.py` |
| 2.3 | **Collaborative filtering**: Matrix Factorization (SVD hoặc ALS) | `models/collaborative.py` |
| 2.4 | **Hybrid model**: kết hợp content + collaborative với trọng số | `models/hybrid_recommender.py` |
| 2.5 | Đánh giá: Precision@K, Recall@K, NDCG | `notebooks/rec_evaluation.ipynb` |

**Phase 3: Personalized Training Plan**

| Task | Mô tả | Output |
|---|---|---|
| 3.1 | Xây dựng **Workout Plan Generator** 7 ngày từ top-K recommendations | `models/plan_generator.py` |
| 3.2 | Triển khai **Progressive Overload Logic**: tăng dần intensity theo tuần | `models/progressive_overload.py` |
| 3.3 | Triển khai **Fatigue Score**: giảm khuyến nghị nếu user tập liên tục | `models/fatigue_score.py` |
| 3.4 | API endpoint `POST /recommend` trả về workout plan JSON | `api/recommend_router.py` |

**Phase 4: Hoàn thiện**

| Task | Mô tả | Output |
|---|---|---|
| 4.1 | Đánh giá cuối: so sánh rule-based vs content-based vs hybrid | `reports/rec_comparison.md` |
| 4.2 | Viết phần báo cáo: Recommendation System Architecture & Evaluation | Báo cáo cuối kỳ |

**Deliverables:**
- [ ] Rule-based baseline recommender
- [ ] Content-based + Collaborative Filtering models
- [ ] **Hybrid recommender** (chính)
- [ ] **Workout Plan Generator** 7 ngày
- [ ] API `/recommend` hoạt động
- [ ] Evaluation report (Precision@K, NDCG)

---

### 👤 Member 4 — NLP Engineer (Fitness QA – RAG System)

**Trách nhiệm chính:** Xây dựng hệ thống hỏi đáp fitness sử dụng Retrieval-Augmented Generation (RAG). Dùng LangChain + Chroma.

#### Nhiệm vụ theo Phase

**Phase 1: Knowledge Base Construction**

| Task | Mô tả | Output |
|---|---|---|
| 1.1 | Thu thập tài liệu fitness: PDF sách, blog uy tín (ACE, NSCA...) | `data/knowledge_base/raw/` |
| 1.2 | Viết **Document Loader**: xử lý PDF, HTML, plain text | `rag/document_loader.py` |
| 1.3 | Chunking: chia văn bản thành chunks (chunk_size=512, overlap=50) | `rag/chunker.py` |
| 1.4 | Embedding chunks bằng `sentence-transformers` (multilingual) | `rag/embedder.py` |
| 1.5 | Lưu vector embeddings vào **Chroma** vector store | `rag/vector_store.py` |

**Phase 2: Retrieval-Augmented Generation**

| Task | Mô tả | Output |
|---|---|---|
| 2.1 | Xây dựng **RAG pipeline** cơ bản: query → retrieve → augment → generate | `rag/pipeline.py` |
| 2.2 | Chọn và tích hợp LLM (GPT-4o-mini hoặc Gemini via API) | `rag/llm_connector.py` |
| 2.3 | Test với 50 câu hỏi mẫu: "Cách giảm mỡ bụng?", "Protein cần nạp bao nhiêu?" | `tests/test_qa.py` |
| 2.4 | Đánh giá: **Context Relevance**, **Answer Faithfulness** (RAGAS framework) | `reports/rag_eval_phase2.md` |

**Phase 3: Advanced QA**

| Task | Mô tả | Output |
|---|---|---|
| 3.1 | **Multi-query retrieval**: sinh nhiều biến thể câu hỏi, merge kết quả | `rag/multi_query.py` |
| 3.2 | **Guardrail**: từ chối trả lời câu hỏi nguy hiểm / ngoài phạm vi fitness | `rag/guardrail.py` |
| 3.3 | **Context Memory**: nhớ lịch sử hội thoại (ConversationBufferMemory) | `rag/memory.py` |
| 3.4 | API endpoint `POST /ask` nhận question → trả về answer + source_docs | `api/qa_router.py` |

**Phase 4: Đánh giá & Báo cáo**

| Task | Mô tả | Output |
|---|---|---|
| 4.1 | Đánh giá cuối: ≥ 85% câu hỏi đúng ngữ cảnh, đo Faithfulness & Relevance | `reports/rag_final_eval.md` |
| 4.2 | Viết phần báo cáo: RAG Architecture, Knowledge Base & Evaluation | Báo cáo cuối kỳ |

**Deliverables:**
- [ ] Knowledge base với ≥ 50 tài liệu fitness
- [ ] Chroma vector store đã embed đầy đủ
- [ ] **RAG pipeline** LangChain hoạt động
- [ ] Multi-query retrieval + Guardrail
- [ ] Context memory cho hội thoại nhiều lượt
- [ ] API `/ask` — ≥ 85% câu trả lời đúng ngữ cảnh

---

### 👤 Member 5 — Full-Stack + Backend

**Trách nhiệm chính:** Xây dựng toàn bộ backend API và frontend web app. Tích hợp tất cả các module: Pose Recognition, Recommendation, RAG QA.

**Stack:** FastAPI (Backend) · React (Frontend) · PostgreSQL (Database)

#### Nhiệm vụ theo Phase

**Phase 1: Backend API Skeleton + Database**

| Task | Mô tả | Output |
|---|---|---|
| 1.1 | Thiết kế **PostgreSQL schema**: users, exercises, sessions, workout_plans | `db/schema.sql` |
| 1.2 | Khởi tạo **FastAPI project** với cấu trúc thư mục chuẩn | `app/` |
| 1.3 | Viết API skeleton: `/predict_pose`, `/recommend`, `/ask`, `/user/profile` | `app/routers/` |
| 1.4 | Cài CORS, authentication (JWT), middleware logging | `app/middleware/` |
| 1.5 | Docker Compose: FastAPI + PostgreSQL + Chroma | `docker-compose.yml` |
| 1.6 | Prototype **React frontend**: layout cơ bản 4 trang | `frontend/src/pages/` |

**Phase 2: Frontend & DB Integration**

| Task | Mô tả | Output |
|---|---|---|
| 2.1 | Trang **Live Workout**: webcam component + hiển thị skeleton overlay | `frontend/src/pages/LiveWorkout.jsx` |
| 2.2 | Trang **Recommendation**: form user profile → hiển thị workout plan | `frontend/src/pages/Recommendation.jsx` |
| 2.3 | Trang **Progress Tracking**: biểu đồ lịch sử tập (Recharts) | `frontend/src/pages/Progress.jsx` |
| 2.4 | Trang **Fitness Chatbot**: chat interface gọi `/ask` API | `frontend/src/pages/Chatbot.jsx` |
| 2.5 | Kết nối backend API từ React (axios / fetch) | `frontend/src/services/api.js` |

**Phase 3: Full Integration**

| Task | Mô tả | Output |
|---|---|---|
| 3.1 | Tích hợp Pose model M2 vào `/predict_pose` endpoint | `app/routers/pose.py` |
| 3.2 | Tích hợp Recommendation engine M3 vào `/recommend` endpoint | `app/routers/recommend.py` |
| 3.3 | Tích hợp RAG system M4 vào `/ask` endpoint | `app/routers/qa.py` |
| 3.4 | Lưu lịch sử tập vào PostgreSQL (session logs, rep counts) | `app/services/session_service.py` |
| 3.5 | End-to-end test: user flow từ đăng nhập → tập → nhận gợi ý → hỏi chatbot | Integration tests |

**Phase 4: Hoàn thiện & Demo**

| Task | Mô tả | Output |
|---|---|---|
| 4.1 | Deploy app lên server (hoặc Docker local cho demo) | Deployment guide |
| 4.2 | Performance test: API response time, concurrent users | `benchmarks/api_perf.md` |
| 4.3 | Demo script: video walkthrough các tính năng chính | Demo video |
| 4.4 | Viết phần báo cáo: System Architecture, API Design, Deployment | Báo cáo cuối kỳ |

**Deliverables:**
- [ ] **FastAPI backend** với 4 endpoints chính
- [ ] **PostgreSQL schema** đầy đủ
- [ ] **React web app** 4 trang
- [ ] Tích hợp đầy đủ 3 module AI
- [ ] Docker Compose cho development
- [ ] Demo hoàn chỉnh với dữ liệu thực

---

## Timeline tổng thể (4 Phases)

```
Phase │ M1 (Data)           │ M2 (CV/Pose)        │ M3 (RecSys)         │ M4 (NLP/RAG)        │ M5 (Full-stack)
─────┼─────────────────────┼─────────────────────┼─────────────────────┼─────────────────────┼────────────────────
     │ YouTube API crawl   │ EDA pose data       │ User schema design  │ Collect KB docs     │ DB schema design
     │ Metadata collection │ MLP baseline train  │ Feature engineering │ Document loading    │ FastAPI skeleton
  1  │ Frame extraction    │ Baseline evaluation │ Mock dataset        │ Chunking + embed    │ Docker Compose
     │ Pose extraction     │ Error analysis      │ Rule-based rec      │ Chroma vector store │ React layout
─────┼─────────────────────┼─────────────────────┼─────────────────────┼─────────────────────┼────────────────────
     │ Sequence builder    │ LSTM training       │ Content-based       │ RAG pipeline basic  │ React pages (4)
     │ Quality filter      │ GRU comparison      │ Collaborative (MF)  │ LLM integration     │ API integration
  2  │ Augmentation        │ Latency benchmark   │ Hybrid model        │ 50-question test    │ DB connection
     │ ≥ 50K sequences     │ Accuracy ≥ 85%      │ Evaluation metrics  │ RAGAS evaluation    │ Auth + middleware
─────┼─────────────────────┼─────────────────────┼─────────────────────┼─────────────────────┼────────────────────
     │ Form label dataset  │ Rep counter         │ Workout plan gen.   │ Multi-query         │ Pose integration
     │ Help M2 if needed   │ Form checker        │ Progressive overload│ Guardrail           │ Rec integration
  3  │ Final dataset stats │ Real-time API       │ Fatigue score       │ Context memory      │ RAG integration
     │                     │ End-to-end test     │ /recommend API      │ /ask API            │ Session logging
─────┼─────────────────────┼─────────────────────┼─────────────────────┼─────────────────────┼────────────────────
     │ Dataset report      │ Final benchmark     │ Final comparison    │ Final evaluation    │ Deploy + demo
  4  │ Write report        │ Write report        │ Write report        │ Write report        │ Perf test
     │ Review + submit     │ Review + submit     │ Review + submit     │ Review + submit     │ Write report
```

---

## Dependencies giữa các Member

### Sơ đồ tổng quan

```
              ┌──────────────────────────────────────────────┐
              │   M1 (Data Engineer) — CRITICAL PATH         │
              │   Tất cả ML models đều cần dataset từ M1     │
              │   Phải xong ≥ 50K sequences trước Phase 2    │
              └──────┬───────────┬──────────────┬────────────┘
                     │           │              │
        ┌────────────▼──┐   ┌────▼──────┐  ┌────▼───────────┐
        │  M2 (CV/Pose) │   │ M3 (Rec)  │  │ M4 (RAG/NLP)  │
        └───────┬───────┘   └─────┬─────┘  └────────┬───────┘
                │                 │                  │
        pose_model            rec_model           rag_system
        rep_counter          workout_plan         /ask API
                │                 │                  │
                └────────────────►│◄─────────────────┘
                                  │
                         ┌────────▼─────────┐
                         │   M5 (Full-stack) │
                         │   FastAPI + React │
                         │   PostgreSQL      │
                         └──────────────────┘
```

---

### Bảng Hard Blockers — Không làm được nếu chưa có

> **Hard blocker** = bắt buộc phải chờ, không có cách workaround.

| Người bị block | Chờ ai | Chờ gì cụ thể | Deadline cần có | Nếu trễ thì làm gì |
|---|---|---|---|---|
| M2 | **M1** | ≥ 10K pose sequences để bắt đầu training | Cuối Phase 1 | M2 dùng dataset công khai (NTU RGB+D, UCF101) tạm |
| M3 | **M1** | Exercise schema chuẩn hóa + metadata | Cuối Phase 1 | M3 tự thiết kế schema, sync với M1 sau |
| M4 | **—** | M4 không phụ thuộc ML pipeline, chỉ cần LLM API key | — | — |
| M5 | **M2** | `/predict_pose` model hoạt động để tích hợp | Đầu Phase 3 | M5 build UI với mock pose response trước |
| M5 | **M3** | `/recommend` model trả về workout plan | Đầu Phase 3 | M5 dùng hardcoded workout plan tạm |
| M5 | **M4** | `/ask` RAG pipeline hoạt động | Giữa Phase 2 | M5 mockup chatbot interface với dummy response |
| M2 | **M5** | Database để lưu session + form feedback | Phase 2 | M2 lưu local JSON tạm |

---

### Bảng Soft Dependencies — Nên có nhưng không bắt buộc dừng

| Ai | Cần từ ai | Mô tả | Workaround |
|---|---|---|---|
| M3 | M2 | Biết loại bài tập nào model nhận diện được | M3 dùng danh sách bài tập từ schema |
| M4 | M3 | Biết danh mục bài tập để giới hạn KB scope | M4 cover toàn bộ fitness KB |
| M5 | M3 | Biết format output của workout plan để render UI | M5 thiết kế UI theo JSON schema thống nhất |

---

### Giao thức Handoff (Bàn giao giữa các thành viên)

Mỗi khi một thành viên hoàn thành phần mình để người khác dùng, **phải thực hiện đủ 3 bước**:

```
Bước 1: Tạo PR vào branch develop, tag người nhận vào để review
Bước 2: Ghi vào file HANDOFF_LOG.md:
         - Tên artifact (file model, API endpoint, schema)
         - Đường dẫn chính xác (local path / model registry)
         - Format input/output (ví dụ: JSON với fields gì)
         - Ví dụ request/response
Bước 3: Ping người nhận trên nhóm chat, confirm họ đã test được
```

**Các mốc handoff cụ thể:**

| Phase | Từ | Đến | Artifact | Format |
|---|---|---|---|---|
| Cuối Phase 1 | M1 | M2, M3 | ≥ 50K pose sequences | `data/pose_sequences/*.npy`, shape: (N, 30, 99) |
| Cuối Phase 1 | M1 | M3, M5 | Exercise metadata | `data/metadata.json` |
| Cuối Phase 2 | M2 | M5 | Pose model checkpoint | `models/lstm_pose.pth` + inference wrapper |
| Cuối Phase 2 | M3 | M5 | Recommendation model | `models/hybrid_recommender.pkl` + API spec |
| Giữa Phase 2 | M4 | M5 | RAG pipeline | `rag/pipeline.py` + Chroma DB path + API spec |
| Đầu Phase 3 | M5 | M2, M3, M4 | PostgreSQL schema live | DB connection string + table docs |

---

## Shared Responsibilities (Tất cả thành viên)

| Trách nhiệm chung | Mô tả |
|---|---|
| **Code review** | Mỗi PR cần ít nhất 1 người review trước khi merge vào `develop` |
| **Unit tests** | Mỗi module phải có file `tests/test_*.py` cơ bản |
| **Git workflow** | Làm việc trên `feature/<tên>` branch, không push thẳng vào `main` |
| **Documentation** | Mỗi function phải có docstring, mỗi module có README ngắn |
| **Meeting hàng tuần** | Báo cáo tiến độ, blockers, điều chỉnh kế hoạch |
| **Báo cáo cuối kỳ** | Mỗi người viết section của mình (xem Phase 4 tasks) |

---

## Cấu trúc thư mục dự án (Target)

```
fitness-recommendation-system/
├── data/
│   ├── raw/
│   │   └── videos/
│   ├── frames/
│   ├── pose_sequences/
│   ├── splits/
│   ├── metadata.json
│   ├── mock_users.json
│   ├── form_labels/
│   └── knowledge_base/
│       └── raw/
├── preprocessing/
│   ├── pose_extractor.py
│   ├── normalize_pose.py
│   ├── sequence_builder.py
│   ├── quality_filter.py
│   └── augment.py
├── models/
│   ├── mlp_pose.pth
│   ├── lstm_pose.pth
│   ├── rep_counter.py
│   ├── form_checker.py
│   ├── rule_recommender.py
│   ├── content_based.py
│   ├── collaborative.py
│   ├── hybrid_recommender.py
│   ├── plan_generator.py
│   ├── progressive_overload.py
│   └── fatigue_score.py
├── rag/
│   ├── document_loader.py
│   ├── chunker.py
│   ├── embedder.py
│   ├── vector_store.py
│   ├── pipeline.py
│   ├── llm_connector.py
│   ├── multi_query.py
│   ├── guardrail.py
│   └── memory.py
├── app/
│   ├── main.py
│   ├── routers/
│   │   ├── pose.py
│   │   ├── recommend.py
│   │   ├── qa.py
│   │   └── user.py
│   ├── services/
│   │   └── session_service.py
│   └── middleware/
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── LiveWorkout.jsx
│   │   │   ├── Recommendation.jsx
│   │   │   ├── Progress.jsx
│   │   │   └── Chatbot.jsx
│   │   └── services/
│   │       └── api.js
│   └── package.json
├── db/
│   └── schema.sql
├── schemas/
│   ├── exercise_schema.json
│   └── user_schema.json
├── notebooks/
│   ├── eda_pose.ipynb
│   ├── model_comparison.ipynb
│   └── rec_evaluation.ipynb
├── reports/
├── benchmarks/
├── tests/
│   └── test_qa.py
├── docker-compose.yml
├── requirements.txt
├── README.md
├── TECH_STACK.md
└── TEAM_GUIDE.md
```

---

## Checklist Phase Gate

### ✅ Phase 1 Done khi:
- [ ] ≥ 50K pose sequences NumPy ready (M1)
- [ ] Knowledge base embedded vào Chroma (M4)
- [ ] FastAPI skeleton 4 endpoints chạy được (M5)
- [ ] PostgreSQL schema khởi tạo xong (M5)
- [ ] MLP baseline có kết quả đầu tiên (M2)

### ✅ Phase 2 Done khi:
- [ ] Pose model Accuracy ≥ 85% (M2)
- [ ] Hybrid recommender trả về workout plan (M3)
- [ ] RAG pipeline trả lời đúng câu hỏi cơ bản (M4)
- [ ] React frontend 4 trang kết nối được API (M5)

### ✅ Phase 3 Done khi:
- [ ] End-to-end: User tập → Pose detected → Session logged → Recommendation → Chatbot (tất cả)
- [ ] Rep counter và form checker hoạt động live (M2)
- [ ] Workout plan 7 ngày với progressive overload (M3)
- [ ] Guardrail + Context memory hoạt động (M4)
- [ ] Full integration test pass (M5)

### ✅ Phase 4 Done khi:
- [ ] Demo hoàn chỉnh với dữ liệu thực
- [ ] Báo cáo cuối kỳ: mỗi người đã viết section của mình
- [ ] Benchmark latency + accuracy documented
- [ ] Tất cả deliverables đã tick xong
