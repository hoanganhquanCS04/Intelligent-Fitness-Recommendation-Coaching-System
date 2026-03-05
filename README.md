# Intelligent Fitness Recommendation & Coaching System
### Hệ thống gợi ý bài tập cá nhân hóa & huấn luyện thông minh — Deep Learning + RAG + Computer Vision

> **Trạng thái:** 🟢 Phase 1 — Thiết kế & Thu thập dữ liệu

---

## Chúng ta đang xây dựng gì?

> **Một câu:** Hệ thống AI tự động nhận diện động tác tập luyện qua camera, gợi ý bài tập cá nhân hóa theo mục tiêu người dùng, và trả lời câu hỏi fitness trong thời gian thực dựa trên knowledge base chuyên sâu.

**Sản phẩm cuối cùng trông như thế nào:**
- Một **web app** có thể mở camera lên và hệ thống tự động đếm repetition, phát hiện lỗi form khi tập
- Giao diện **Recommendation** gợi ý kế hoạch tập 7 ngày phù hợp với thể trạng & mục tiêu từng người
- **Fitness Chatbot** trả lời câu hỏi như "Cách giảm mỡ bụng?" dựa trên cơ sở tri thức fitness chuyên sâu
- **Progress Tracking** theo dõi tiến trình tập luyện theo thời gian

---

## Định nghĩa "Thành công" (Definition of Done)

Dự án coi là **hoàn thành** khi đáp ứng đủ các tiêu chí sau:

| # | Tiêu chí |
|---|---|
| 1 | Pipeline pose recognition chạy **real-time** với latency < 150ms |
| 2 | Pose classifier đạt **Accuracy ≥ 85%** trên tập test |
| 3 | Recommendation engine sinh được **workout plan 7 ngày** cá nhân hóa |
| 4 | RAG system trả lời đúng **≥ 85%** câu hỏi fitness theo ngữ cảnh |
| 5 | **Rep counter** và **form checker** hoạt động trong luồng video live |
| 6 | Web app end-to-end **chạy được demo** với người dùng thực tế |

---

## Mỗi người đóng góp gì vào sản phẩm?

```
Member 1 (Data Engineer)     →  Thu thập video workout, chuẩn hóa dataset pose
                                Xây dựng pipeline xử lý frame + pose sequences

Member 2 (ML Engineer – CV)  →  Nhận diện & phân loại động tác thời gian thực
                                MLP baseline → LSTM/GRU sequence model → Real-time API

Member 3 (ML Engineer – Rec) →  Gợi ý bài tập cá nhân hóa theo mục tiêu người dùng
                                Rule-based → Content-based → Collaborative Filtering → Hybrid

Member 4 (NLP Engineer)      →  Hệ thống hỏi đáp fitness dựa trên RAG
                                LangChain + Chroma + LLM → Guardrail → Context Memory

Member 5 (Full-Stack)        →  Biến tất cả thành sản phẩm người dùng có thể dùng
                                FastAPI + React + PostgreSQL → Web App hoàn chỉnh
```

---

## Kiến trúc hệ thống

```
[Camera / Video Input]
        │  stream
        ▼
  MediaPipe / OpenPose
  (Pose Extraction — 33 keypoints)
        │  keypoint vectors
        ▼
  Pose Classifier (MLP → LSTM/GRU)
  ├── Exercise Recognition   → "squat", "push-up"...
  ├── Rep Counter            → đếm lần lặp
  └── Form Checker           → phát hiện sai form
        │
        ▼
  User Profile + History (PostgreSQL)
        │
  ┌─────────────────────────┐
  │  Recommendation Engine  │
  │  Content-based Filtering│
  │  Collaborative Filtering│
  │  → Hybrid Model         │
  └──────────┬──────────────┘
             │  workout plan
             ▼
  ┌─────────────────────────┐
  │  RAG QA System          │
  │  LangChain + Chroma     │
  │  Knowledge Base Fitness │
  └──────────┬──────────────┘
             │
             ▼
     FastAPI Backend
             │
             ▼
     React Frontend  ← user tương tác ở đây
```

> **Backend:** FastAPI  
> **Frontend:** React  
> **Database:** PostgreSQL  
> **Vector Store:** Chroma (RAG)

---

## Tài liệu chi tiết

| Tài liệu | Nội dung |
|---|---|
| [`TECH_STACK.md`](TECH_STACK.md) | Tech stack, kiến trúc từng layer, code mẫu, requirements |
| [`TEAM_ASSIGNMENTS.md`](TEAM_ASSIGNMENTS.md) | Phân công tasks chi tiết theo Phase cho 5 thành viên |
| [`TEAM_GUIDE.md`](TEAM_GUIDE.md) | Git workflow, coding conventions, PR process |

---

## Lộ trình 4 Phases

| Phase | Mục tiêu cần đạt được |
|---|---|
| **Phase 1** | Dataset ≥ 50K pose samples, KB embedding xong, API skeleton chạy được |
| **Phase 2** | Pose Accuracy ≥ 85%, Recommendation hoạt động, RAG trả lời đúng |
| **Phase 3** | End-to-end pipeline, User tập → nhận feedback → được gợi ý → hỏi đáp |
| **Phase 4** | Demo hoàn chỉnh, báo cáo đầy đủ, benchmark latency + accuracy |

---

## Quick Start

```bash
# 1. Clone repo
git clone https://github.com/<org>/fitness-recommendation-system.git
cd fitness-recommendation-system

# 2. Đọc tài liệu theo thứ tự này:
#    README.md (file này)  →  TEAM_ASSIGNMENTS.md  →  TECH_STACK.md

# 3. Setup môi trường local
docker-compose up -d          # Khởi động PostgreSQL local

conda create -n fitness-ai python=3.10
conda activate fitness-ai
pip install -r requirements.txt

# 4. Chạy API server
uvicorn app.main:app --reload

# 5. Chạy frontend
cd frontend
npm install && npm start
```
