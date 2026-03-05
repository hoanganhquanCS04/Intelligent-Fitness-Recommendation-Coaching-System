**Dự án: Intelligent Fitness Recommendation & Coaching System**  
Hệ thống gồm 3 trụ cột chính:

1. Nhận diện động tác tập (Pose Recognition)

2. Gợi ý bài tập theo nhu cầu người dùng (Personalized Recommendation)

3. Hỏi đáp dựa trên cơ sở tri thức (RAG-based Fitness QA)

Nhóm: 5 thành viên — 4 Phases

---

**Tổng quan phân công**

| Member | Role | Layer | Core Responsibility |
| :---: | :---: | :---: | :---: |
| M1 | Data Engineer | Data & Video Pipeline | Thu thập \+ xử lý video \+ dataset |
| M2 | ML Engineer (CV) | Pose Recognition | Nhận diện & phân loại động tác |
| M3 | ML Engineer (RecSys) | Recommendation Engine | Cá nhân hóa bài tập |
| M4 | NLP Engineer | QA \+ Knowledge Base | Hỏi đáp fitness |
| M5 | Full-Stack \+ Backend | API \+ App \+ DB | Triển khai hệ thống |

---

**👤 Member 1 — Data Engineer (Video & Dataset Pipeline)**

**Trách nhiệm chính:**

Xây dựng pipeline dữ liệu video và metadata cho bài tập.

---

**Phase 1: Dataset Collection**

| Task | Mô tả | Output |
| :---: | :---: | :---: |
| 1.1 | Thu thập video workout từ YouTube API | data/raw/videos/ |
| 1.2 | Crawl metadata bài tập (muscle group, level) | data/metadata.json |
| 1.3 | Chuẩn hóa nhãn động tác (squat, push-up...) | schemas/exercise\_schema.json |
| 1.4 | Extract frame từ video (OpenCV) | data/frames/ |
| 1.5 | Tạo dataset train/test | data/splits/ |

---

**Phase 2: Preprocessing Pipeline**

| Task | Mô tả | Output |
| :---: | :---: | :---: |
| 2.1 | Tích hợp pose extractor (MediaPipe) | preprocessing/pose\_extractor.py |
| 2.2 | Chuẩn hóa keypoints (33 landmarks → vector) | preprocessing/normalize\_pose.py |
| 2.3 | Lưu pose sequences dạng NumPy | data/pose\_sequences/ |
| 2.4 | Tạo dataset ≥ 50K pose samples | Pose dataset ready |

---

**Deliverables**

* ≥ 50K pose sequences

* Metadata chuẩn hóa

---

**👤 Member 2 — ML Engineer (Computer Vision – Pose Recognition)**

Sử dụng MediaPipe hoặc OpenPose.

---

**Phase 1: Baseline Pose Classification**

| Task | Mô tả | Output |
| :---: | :---: | :---: |
| 1.1 | Trích xuất keypoints từ frame | pose vectors |
| 1.2 | Train MLP baseline classifier | models/mlp\_pose.pth |
| 1.3 | Đánh giá Accuracy / F1 | reports/pose\_eval.md |

---

**Phase 2: Sequence Modeling**

| Task | Mô tả | Output |
| :---: | :---: | :---: |
| 2.1 | Xây dựng LSTM / GRU cho chuỗi động tác | models/lstm\_pose.pth |
| 2.2 | So sánh CNN-1D vs LSTM | comparison report |
| 2.3 | Tối ưu latency \< 150ms/inference | benchmark |

---

**Phase 3: Real-time Integration**

| Task | Mô tả |
| :---- | :---- |
| 3.1 | Viết API predict\_pose(video\_stream) |
| 3.2 | Tính repetition counter |
| 3.3 | Phát hiện sai form (rule-based threshold) |

---

**Deliverables**

* Model Accuracy ≥ 85%

* Real-time inference

* Rep counter hoạt động

---

**👤 Member 3 — ML Engineer (Recommendation System)**

---

**Phase 1: User Profiling**

| Task | Mô tả |
| :---: | :---: |
| 1.1 | Thiết kế user schema (goal, weight, level) |
| 1.2 | Feature engineering từ lịch sử tập |

---

**Phase 2: Recommendation Models**

| Task | Mô tả |
| :---- | :---- |
| 2.1 | Rule-based recommender (baseline) |
| 2.2 | Content-based filtering |
| 2.3 | Collaborative filtering (Matrix Factorization) |
| 2.4 | Hybrid model |

---

**Phase 3: Personalized Training Plan**

| Task | Mô tả |
| :---- | :---- |
| 3.1 | Sinh workout plan 7 ngày |
| 3.2 | Progressive overload logic |
| 3.3 | Tối ưu theo fatigue score |

---

**Deliverables**

* Top-K recommendation accuracy

* Workout plan generator

* Hybrid recommender

---

**👤 Member 4 — NLP Engineer (Fitness QA – RAG System)**

Sử dụng LangChain \+ Chroma.

---

**Phase 1: Knowledge Base**

| Task | Mô tả |
| :---- | :---- |
| 1.1 | Thu thập tài liệu fitness (PDF, blog) |
| 1.2 | Chunking \+ embedding |
| 1.3 | Lưu vector vào Chroma |

---

**Phase 2: Retrieval-Augmented Generation**

| Task | Mô tả |
| :---- | :---- |
| 2.1 | Xây dựng pipeline RAG |
| 2.2 | Test câu hỏi: “Cách giảm mỡ bụng?” |
| 2.3 | Đánh giá relevance |

---

**Phase 3: Advanced QA**

| Task | Mô tả |
| :---- | :---- |
| 3.1 | Multi-query retrieval |
| 3.2 | Guardrail tránh advice nguy hiểm |
| 3.3 | Context memory cho hội thoại |

---

**Deliverables**

* RAG hoạt động

* ≥ 85% câu trả lời đúng ngữ cảnh

* API ask(question)

---

**👤 Member 5 — Full-Stack \+ Backend**

Backend: FastAPI  
Frontend: React  
Database: PostgreSQL

---

**Phase 1: Backend API**

* /predict\_pose

* /recommend

* /ask

* /user/profile

---

**Phase 2: Frontend**

Trang:

1. Live Workout

2. Recommendation

3. Progress Tracking

4. Fitness Chatbot

---

**Phase 3: Integration**

* Connect pose → DB

* Connect recommendation engine

* Connect RAG system

---

**Deliverables**

* Full web app

* REST API hoàn chỉnh

* Database schema

---

**Timeline 4 Phases**

| Phase | CV | RecSys | NLP | Backend |
| :---- | :---- | :---- | :---- | :---- |
| 1 | Dataset | User schema | KB build | API skeleton |
| 2 | Train model | Hybrid Rec | RAG | Connect DB |
| 3 | Real-time | Workout plan | Guardrail | Integration |
| 4 | Benchmark | Evaluation | QA test | Demo |

---

**Phase Gate**

 **Phase 1 Done khi:**

* ≥ 50K pose samples

* KB embedding xong

* API skeleton chạy

 **Phase 2 Done khi:**

* Pose accuracy ≥ 85%

* Recommendation hoạt động

* RAG trả lời đúng

 **Phase 3 Done khi:**

* End-to-end pipeline hoạt động

* User có thể tập → nhận feedback → được gợi ý → hỏi đáp

 **Phase 4 Done khi:**

* Demo hoàn chỉnh

* Báo cáo đầy đủ

* Benchmark latency \+ accuracy

