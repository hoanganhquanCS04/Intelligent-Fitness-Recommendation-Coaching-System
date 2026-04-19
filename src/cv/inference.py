import torch
from ultralytics import YOLO
from dataclasses import dataclass, asdict
from collections import deque
import numpy as np

# Cắt Lắp từ phân xưởng Module bên cạnh
from .heuristics import extract_critical_angles
from .form_reporter import generate_form_report

@dataclass
class InferenceDecision:
    status: str
    reason: str
    confidence: float

def apply_confidence_gate(pose_result, threshold=0.75):
    """
    Chức năng: Lọc nhiễu Confidence Frame
    """
    if not pose_result or len(pose_result.keypoints) == 0:
        return InferenceDecision("rejected", "no_person_detected", 0.0)

    kpts_conf = pose_result.keypoints.conf[0]
    avg_conf = float(torch.mean(kpts_conf)) if kpts_conf is not None else 0.0

    if avg_conf < threshold:
        return InferenceDecision("rejected", "confidence_below_threshold", avg_conf)
    return InferenceDecision("accepted", "pass", avg_conf)

class ExerciseStateTracker:
    """
    Cục Pin Lắp Nhớ (State Tracking Machine).
    Lưu giữ các góc đỉnh, đáy, và hướng di chuyển của cơ qua từng Frame.
    """
    def __init__(self):
        self.history = deque(maxlen=5) # Nhớ 5 Frame gần nhất để tính Xu hướng (Trend)
        self.phase = "standing" # Phase mặt định lúc mới quét
        self.min_knee_angle = 180.0
        self.min_elbow_angle = 180.0
        
        # Deadlift setup
        self.prev_hip_y = 0.0
        self.prev_shoulder_y = 0.0
        
        # Persistence Lỗi hiển thị
        self.active_errors = []
        self.error_hold_frames = 0
        
    def update(self, exercise_type, angles):
        ex = exercise_type.lower()
        knee = angles.get("knee_angle", 180)
        elbow = angles.get("elbow_angle", 180)
        hip_y = angles.get("hip_y", 0)
        sho_y = angles.get("shoulder_y", 0)
        
        # Ghi Lịch Sử Cuốn Chiếu
        self.history.append({"knee": knee, "elbow": elbow, "hip_y": hip_y, "sho_y": sho_y})
        
        # Rò rỉ Lỗi dần cho rỗng sau N Frame để màn hình k bị dồn ứ cục Lỗi cũ
        if self.error_hold_frames > 0:
            self.error_hold_frames -= 1
        else:
            self.active_errors = []

        # ============ LOGIC SQUAT ============
        if ex == "squat":
            if self.phase == "standing" and knee < 155:
                self.phase = "descending"
                self.min_knee_angle = knee
                
            elif self.phase == "descending":
                self.min_knee_angle = min(self.min_knee_angle, knee)
                # Đi tìm Đáy: Nếu góc gối bắt đầu Nẩy lớn lên 10 độ so với Đáy -> Đã qua Đáy
                if knee > self.min_knee_angle + 10:
                    self.phase = "ascending"
                    
            elif self.phase == "ascending":
                if knee > 160: # Trở về đứng thẳng
                    self.phase = "standing"
                    self.min_knee_angle = 180.0 # Clear ván mới
                    
        # ============ LOGIC DEADLIFT ============
        elif ex == "deadlift":
            if self.phase == "standing": self.phase = "setup"
            
            if self.phase == "setup" and (self.prev_hip_y - hip_y) > 15: # Hông bắt đầu nhổm lên cao (Ngược Y trục)
                self.phase = "pulling"
                
            elif self.phase == "pulling":
                if knee > 165: # Đứng thẳng Lock tạ trên Đỉnh cao
                    self.phase = "lockout"
                    
            elif self.phase == "lockout":
                if knee < 150: # Bỏ rơi tạ sập chùng gối đập đất
                    self.phase = "setup"

        # ============ LOGIC BENCH PRESS ============
        elif ex == "bench_pressing":
            if self.phase == "standing": self.phase = "lockout" # Cầm tạ nhấc rời kệ
            
            if self.phase == "lockout" and elbow < 150:
                self.phase = "lowering"
                self.min_elbow_angle = elbow
                
            elif self.phase == "lowering":
                self.min_elbow_angle = min(self.min_elbow_angle, elbow)
                if elbow > self.min_elbow_angle + 10: # Nảy Cùi chỏ đẩy vút tạ nhô lên không trung
                    self.phase = "pressing"
                    
            elif self.phase == "pressing":
                if elbow > 160:
                    self.phase = "lockout"

        # Lưu vết Y Cổ định để tính đạo hàm vận tốc cho Frame kế
        self.prev_hip_y = hip_y
        self.prev_shoulder_y = sho_y

        # Gói Hộp Trạng thái (Trọng lượng thông tin Phân tuyến qua Reporter)
        return {
            "phase": self.phase,
            "min_knee_angle": self.min_knee_angle,
            "min_elbow_angle": self.min_elbow_angle,
            "prev_hip_y": self.history[0]["hip_y"] if len(self.history)>0 else hip_y,
            "prev_shoulder_y": self.history[0]["sho_y"] if len(self.history)>0 else sho_y
        }

class FitnessInferenceSystem:
    """
    Khu Tổ Hợp Trung Tâm Đa Lõi (Core API Wrapper).
    """
    def __init__(self, pose_model_path, cls_model_path=None, exercise_type="squat"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model = YOLO(pose_model_path).to(self.device)
        self.cls_model = YOLO(cls_model_path).to(self.device) if cls_model_path else None
            
        self.exercise_type = exercise_type
        self.tracker = ExerciseStateTracker() # Lắp Cục Pin theo dõi vòng lặp cá nhân hóa

    def predict_frame(self, image_np):
        current_exercise = self.exercise_type
        if self.cls_model:
            cls_out = self.cls_model(image_np, verbose=False)
            if cls_out and len(cls_out[0]) > 0 and cls_out[0].probs is not None:
                top1_id = cls_out[0].probs.top1
                current_exercise = cls_out[0].names[top1_id]

        results = self.model(image_np, verbose=False)
        if not results or len(results[0]) == 0:
            return {"status": "rejected", "reason": "Không bắt được Skeleton người."}
            
        pose_out = results[0]
        decision = apply_confidence_gate(pose_out)
        if decision.status == "rejected":
            return asdict(decision)
            
        kpts_array = pose_out.keypoints.data[0].cpu().numpy()
        
        # 1. Đo Đoán Góc Môn Tập Tinh Gọn Bằng Mắt Thần New!
        angles = extract_critical_angles(kpts_array)
        
        # 2. Tuôn Góc vào Cỗ Máy Lọc Nhịp (Phase Tracker New!)
        current_state = self.tracker.update(current_exercise, angles)
        
        # 3. Y Bác Sĩ Đọc Lỗi dựa theo Chỉ định Nhịp Nhồi & Môn Gì
        fresh_errors = generate_form_report(current_exercise, angles, current_state)
        
        # 4. Trì trệ Cảnh báo (Giữ mực trên màn hình)
        if fresh_errors:
            self.tracker.active_errors = fresh_errors
            self.tracker.error_hold_frames = 45 # Đun nóng ghim chặt chữ này suốt 45 Frames ~ 1.5 giây
            
        return {
            "status": "accepted",
            "exercise_type": current_exercise, 
            "phase": current_state["phase"],      # Ném Trạng Thái Giai đoạn ra mảng JSON
            "confidence": round(decision.confidence, 4),
            "angles": {k: round(v, 2) for k, v in angles.items()},
            "form_report": self.tracker.active_errors,
            "raw_kpts": kpts_array 
        }
