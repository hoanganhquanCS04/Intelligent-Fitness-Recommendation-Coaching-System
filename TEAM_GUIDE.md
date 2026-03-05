# TEAM GUIDE — Git Workflow & Contribution Guide
## Intelligent Fitness Recommendation & Coaching System

> **Hướng dẫn làm việc nhóm**  
> **Cập nhật:** Tháng 3/2026

---

## 1. Git Workflow

### Branch Structure

```
main                              → Production-ready code
  │
  └── develop                     → Integration branch (merge vào đây)
        │
        ├── feature/<tên-feature> → Tính năng mới
        ├── fix/<tên-bug>         → Sửa bug
        └── hotfix/<tên-issue>    → Sửa khẩn cấp
```

### Quy trình cơ bản

```bash
# 1. Tạo branch mới từ develop
git checkout develop
git pull origin develop
git checkout -b feature/ten-feature

# 2. Code và commit
git add .
git commit -m "feat(scope): mô tả ngắn gọn"

# 3. Push lên remote
git push origin feature/ten-feature

# 4. Tạo Pull Request trên GitHub → develop
# 5. Chờ review & merge
# 6. Sync lại develop
git checkout develop
git pull origin develop
```

---

## 2. Quy tắc đặt tên Branch

### Format: `<type>/<mô-tả-ngắn>`

| Type | Dùng khi | Ví dụ |
|------|----------|-------|
| `feature/` | Tính năng mới | `feature/pose-lstm-model` |
| `fix/` | Sửa bug | `fix/rag-retrieval-empty` |
| `hotfix/` | Sửa khẩn cấp | `hotfix/api-crash-prod` |

**Lưu ý:**
- Dùng chữ thường, ngăn cách bằng `-`
- Ngắn gọn, tối đa 50 ký tự
- KHÔNG dùng tên cá nhân hoặc số issue

---

## 3. Commit Convention

### Format: `<type>(<scope>): <mô tả>`

```bash
feat(pose): add LSTM sequence classifier
fix(rag): handle empty retrieval result
docs(readme): update setup instructions
```

### Các type commit:

| Type | Khi nào dùng | Ví dụ |
|------|--------------|-------|
| `feat` | Tính năng mới | `feat(rec): add collaborative filtering` |
| `fix` | Sửa bug | `fix(pose): handle missing keypoint` |
| `docs` | Cập nhật docs | `docs(api): add endpoint documentation` |
| `refactor` | Refactor code | `refactor(rag): optimize chunk retrieval` |
| `test` | Thêm tests | `test(rec): add unit tests for hybrid model` |
| `chore` | Maintenance | `chore(deps): update transformers version` |
| `data` | Cập nhật dataset/model | `data(pose): add 5K new squat sequences` |

### Scope (phạm vi) theo module:
- `pose`, `rec`, `rag`, `api`, `frontend`, `db`, `data`, `infra`

### Quy tắc:
- ✅ Chữ thường, không dấu chấm cuối
- ✅ Dùng động từ: "add", "fix", "update" (không dùng "added", "fixed")
- ✅ Ngắn gọn, tối đa 72 ký tự
- ❌ KHÔNG commit code không chạy được
- ❌ KHÔNG commit file `.env`, model weights lớn, hoặc data thô

---

## 4. Pull Request (PR) Process

### Khi tạo PR:

1. **Điền thông tin đầy đủ:**
   ```markdown
   ## Mô tả
   - Thêm LSTM classifier cho pose sequences
   - Input: (batch, 30, 99) — 30 frames, 33 keypoints × 3 (x,y,z)
   
   ## Thay đổi
   - `models/lstm_pose.py`: LSTM model architecture
   - `notebooks/model_comparison.ipynb`: So sánh LSTM vs GRU vs CNN-1D
   - `tests/test_lstm.py`: Unit tests
   
   ## Test
   - [x] Unit tests pass
   - [x] Accuracy ≥ 85% trên val set
   - [x] Inference < 150ms trên CPU
   
   ## Liên quan
   Closes #12
   ```

2. **Assign reviewer:** Ít nhất 1 người liên quan
3. **Self-review:** Kiểm tra lại code trước khi assign
4. **Đảm bảo:**
   - ✅ Code chạy được
   - ✅ Không có file thừa (`.pyc`, `.env`, data files lớn)
   - ✅ Tests pass
   - ✅ Docstring đầy đủ

### Khi review PR của người khác:

1. **Review trong 24 giờ** (12h cho urgent)
2. **Comment constructive:**
   - ✅ "Có thể thêm try/except ở dòng 34 không? Nếu MediaPipe không detect được pose thì sẽ lỗi"
   - ✅ "Suggest dùng `torch.no_grad()` trong inference để tiết kiệm RAM"
   - ❌ "Code này tệ"
   - ❌ "Tại sao lại làm thế?"
3. **Approve hoặc Request Changes**
4. **Không approve nếu:**
   - Code không chạy
   - Thiếu tests cho logic quan trọng
   - Logic sai về model/data pipeline

### Sau khi merge:

```bash
# Sync lại develop
git checkout develop
git pull origin develop

# Xóa branch cũ
git branch -d feature/old-branch
```

---

## 5. Các lệnh Git thường dùng

### Branch operations
```bash
# Xem danh sách branch
git branch -a

# Chuyển branch
git checkout develop

# Tạo branch mới
git checkout -b feature/new-feature

# Xóa branch local
git branch -d old-branch

# Xóa branch remote
git push origin --delete old-branch
```

### Sync & Update
```bash
# Pull updates từ develop
git checkout develop
git pull origin develop

# Merge develop vào branch hiện tại
git merge develop

# Sync branch với remote
git fetch origin
```

### Commit operations
```bash
# Stage files
git add .                    # Tất cả files
git add file1.py file2.py    # Specific files

# Commit
git commit -m "feat(scope): message"

# Sửa commit cuối
git commit --amend

# Undo commit (giữ changes)
git reset --soft HEAD~1
```

### Stash (lưu tạm changes)
```bash
# Lưu tạm changes
git stash

# Lấy lại changes
git stash pop

# Xem danh sách stash
git stash list
```

### Xử lý conflicts
```bash
# Khi có conflict sau git merge
# 1. Mở file có conflict
# 2. Tìm và sửa phần:
#    <<<<<<< HEAD
#    (your code)
#    =======
#    (their code)
#    >>>>>>> branch-name
# 3. Giữ code đúng, xóa markers <<< === >>>
# 4. Add và commit
git add <resolved-file>
git commit -m "merge: resolve conflicts"
```

### Kiểm tra status
```bash
# Xem status
git status

# Xem lịch sử commits
git log --oneline --graph

# Xem diff
git diff                      # Chưa stage
git diff --staged            # Đã stage
```

---

## 6. Coding Standards

### Python (Backend & ML)

```python
# ✅ Tốt: Docstring đầy đủ, type hints rõ ràng
def predict_pose(keypoints: np.ndarray, threshold: float = 0.7) -> dict:
    """
    Phân loại động tác từ chuỗi keypoints.
    
    Args:
        keypoints: numpy array shape (30, 99) — 30 frames, 33 joints × 3
        threshold: ngưỡng confidence tối thiểu
    
    Returns:
        dict với keys: exercise_name, confidence, is_correct_form
    """
    ...

# ❌ Không tốt: Thiếu docstring, tên biến không rõ
def pred(kp, t=0.7):
    ...
```

### API Response Format (FastAPI)

Tất cả endpoints trả về JSON theo format chuẩn:

```python
# ✅ Format chuẩn
{
    "success": true,
    "data": { ... },
    "error": null
}

# Khi lỗi:
{
    "success": false,
    "data": null,
    "error": "Mô tả lỗi rõ ràng"
}
```

### React (Frontend)

```jsx
// ✅ Component có PropTypes và comment rõ ràng
const WorkoutCard = ({ exercise, onSelect }) => {
  // exercise: { name, muscle_group, difficulty, duration }
  return (
    <div className="workout-card" onClick={() => onSelect(exercise)}>
      <h3>{exercise.name}</h3>
      <p>{exercise.muscle_group} | {exercise.difficulty}</p>
    </div>
  );
};
```

---

## 7. Environment Setup

### Python Environment

```bash
# Tạo môi trường conda
conda create -n fitness-ai python=3.10
conda activate fitness-ai

# Cài dependencies
pip install -r requirements.txt

# Cài pre-commit hooks (kiểm tra code trước khi commit)
pip install pre-commit
pre-commit install
```

### Frontend Setup

```bash
cd frontend
npm install
npm start           # Dev server: http://localhost:3000
```

### Backend Setup

```bash
# Chạy FastAPI server
uvicorn app.main:app --reload --port 8000
# API docs: http://localhost:8000/docs

# Chạy với Docker
docker-compose up -d
```

### Environment Variables (.env)

```bash
# .env — KHÔNG commit file này lên git!
DATABASE_URL=postgresql://user:password@localhost/fitness_db
OPENAI_API_KEY=sk-...
CHROMA_PERSIST_DIR=./chroma_db
SECRET_KEY=your-secret-key-here
```

> **Lưu ý:** File `.env` đã có trong `.gitignore`. Tạo file `.env.example` với giá trị placeholder để người khác biết cần set những biến gì.

---

## 8. Testing Guidelines

### Chạy tests

```bash
# Chạy tất cả tests
pytest tests/ -v

# Chạy tests theo module
pytest tests/test_pose_model.py -v
pytest tests/test_rag.py -v

# Coverage report
pytest tests/ --cov=app --cov-report=html
```

### Cấu trúc tests

```python
# tests/test_pose_model.py
import pytest
import numpy as np
from models.lstm_pose import LSTMPoseClassifier

def test_pose_classifier_output_shape():
    """Model phải trả về logits đúng shape."""
    model = LSTMPoseClassifier(num_classes=10)
    dummy_input = np.random.randn(1, 30, 99).astype(np.float32)
    output = model.predict(dummy_input)
    assert output.shape == (1, 10)

def test_pose_classifier_confidence():
    """Confidence score phải trong khoảng [0, 1]."""
    ...
```

---

## 9. File & Data Management

### Files KHÔNG được commit lên Git:
- `.env` (credentials)
- `*.pth`, `*.pkl`, `*.h5` — model weights lớn (dùng Git LFS hoặc cloud storage)
- `data/raw/videos/` — video files
- `data/pose_sequences/` — numpy arrays lớn
- `chroma_db/` — vector store

### Files chia sẻ qua:
- **Model weights:** Google Drive / HuggingFace Hub → ghi link vào `HANDOFF_LOG.md`
- **Datasets:** Google Drive chia sẻ → ghi path vào README của module
- **Chroma DB:** Shared folder hoặc regenerate từ scripts

### `.gitignore` quan trọng:
```gitignore
# Environment
.env
*.env

# Python
__pycache__/
*.pyc
.venv/

# Data & Models
data/raw/
data/pose_sequences/
*.pth
*.pkl
chroma_db/

# Frontend
node_modules/
frontend/build/
```

---

## 10. Meeting & Communication

### Weekly Meeting Checklist

Mỗi tuần, mỗi người báo cáo:
1. **✅ Đã làm xong:** (so với kế hoạch Phase hiện tại)
2. **🔄 Đang làm:** (task đang thực hiện)
3. **❌ Bị block bởi:** (dependency chưa có)
4. **📅 Kế hoạch tuần tới:** (tasks sẽ làm)

### Quy trình xử lý khi bị block

```
1. Thông báo ngay trên nhóm chat: "M2 đang chờ M1 cung cấp pose sequences"
2. Trong khi chờ: chuyển sang task khác hoặc dùng mock/dummy data
3. Cập nhật HANDOFF_LOG.md khi blocker được giải quyết
4. Ping người liên quan để confirm
```

---

**Good luck! 💪 Hãy build một sản phẩm AI fitness thật cool!**
