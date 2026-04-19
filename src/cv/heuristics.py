import numpy as np
import cv2

def calculate_angle(p1, p2, p3):
    """
    Chức năng:
        Đo một góc lượng giác học (Angle) được tạo bởi 3 điểm bất kỳ trên màn hình 2D.
        Đây là trái tim toán học của toàn bộ cỗ máy AI để xác định bộ xương đang gập hay duỗi.
        
    Args:
        p1 (list/tuple): Tọa độ điểm xương Gốc (VD: Hông) dạng [x, y, confidence].
        p2 (list/tuple): Tọa độ điểm xương Đỉnh Góc xoay (VD: Đầu Gối).
        p3 (list/tuple): Tọa độ điểm xương Kéo dãn (VD: Cổ chân).
        
    Returns:
        float: Mức độ góc được mở (Từ 0.0 -> 180.0 độ). 
               Ví dụ: Đứng thẳng là 180 độ. Ngồi xổm bẹp là tầm 60-70 độ.
    """
    a = np.array(p1[:2]) # Giữ lại x,y bỏ confidence
    b = np.array(p2[:2])
    c = np.array(p3[:2])
    
    # Tính chênh lệch trục Y và X để ép ra radian
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360.0 - angle
        
    return float(angle)

def extract_critical_angles(kpts):
    """
    Chức năng:
        Camera AI sẽ tuôn ra 17 điểm rác toạ độ. Hàm này làm phễu lọc chặn lại những điểm quan trọng nhất 
        của 3 Bài Tập Lớn (Big 3: Squat, Deadlift, Bench Press) và chuyển hóa chúng thành Góc Độ.
        Thuật toán lấy "Trạm Đứng Bên Nào Rõ Nhất" (Average Confidence) để chấm điểm chính xác ngay cả khi bị che một nửa người.
        
    Args:
        kpts (numpy.ndarray): Chuỗi Matrix 17x3 toạ độ x,y,conf xuất ra nguyên bản từ YOLO Pose.
        
    Returns:
        dict: Bộ Tự Điển chứa tất cả các góc Gập (như độ cong lưng, gập chỏ, gập gối). 
              Đây là nguyên liệu để Truyền vào Máy Trạng Thái (State Machine - Phân luồng môn tập).
    """
    angles_dict = {
        "knee_angle": 0.0,
        "hip_y": 0.0,
        "shoulder_y": 0.0,
        "torso_angle": 0.0,
        "elbow_angle": 0.0,
        "shoulder_flare_angle": 0.0
    }
    
    try:
        # Bóc Tách Tách Điểm (Mô hình COCO Pose)
        l_sh, r_sh = kpts[5], kpts[6]    # Vai rái, Vai Phải
        l_el, r_el = kpts[7], kpts[8]    # Cùi chỏ
        l_wr, r_wr = kpts[9], kpts[10]   # Cổ tay
        l_hp, r_hp = kpts[11], kpts[12]  # Hông 
        l_kn, r_kn = kpts[13], kpts[14]  # Gối
        l_an, r_an = kpts[15], kpts[16]  # Cổ chân
        
        # Chiến Thuật Chọn Góc Bóng Đổ: Lấy góc nghiêng nào sáng rõ nhất theo chỉ số Confidence
        conf_left = (l_sh[2] + l_hp[2] + l_kn[2]) / 3
        conf_right = (r_sh[2] + r_hp[2] + r_kn[2]) / 3
        
        # Chỉ định trỏ bên sẽ Quét Angles
        if conf_left >= conf_right:
            sh, el, wr, hp, kn, an = l_sh, l_el, l_wr, l_hp, l_kn, l_an
        else:
            sh, el, wr, hp, kn, an = r_sh, r_el, r_wr, r_hp, r_kn, r_an

        # 1. Thuật Toán Bài Gánh Đùi SQUAT
        # Góc Gối (Hip - Knee - Ankle)
        angles_dict["knee_angle"] = calculate_angle(hp, kn, an)
        
        # Góc Dốc Bề Mặt Lưng Mông (So với mặt đất vuông góc) - Chống bài Lỗi Sập Lưng Buttwink
        hip_vertical = [hp[0], hp[1] - 50, 1.0] # Mọc 1 điểm Thẳng đứng giả định chiếu từ Mông lên Trời
        # Đo độ chéo từ Điểm giả định đó xoay tới Cái Vai, Tâm gốc là Cái Mông
        angles_dict["torso_angle"] = calculate_angle(hip_vertical, hp, sh)

        # 2. Thuật Toán Bài Cúp Lưng DEADLIFT
        # Tọa độ trực quan trục Y
        angles_dict["hip_y"] = float(hp[1])
        angles_dict["shoulder_y"] = float(sh[1])
        
        # 3. Thuật Toán Đẩy Ngực BENCH PRESS
        # Góc co cơ cùi chỏ Pít-tông
        angles_dict["elbow_angle"] = calculate_angle(sh, el, wr)
        # Góc Bành Nách (Đo coi nách bị banh 90 độ hay khép 45 độ sát sườn chữ A)
        angles_dict["shoulder_flare_angle"] = calculate_angle(hp, sh, el)

    except Exception as e:
        # Giữ bộ máy an toàn không bị sập (Crash) nếu thiếu khung xương
        pass
        
    return angles_dict
