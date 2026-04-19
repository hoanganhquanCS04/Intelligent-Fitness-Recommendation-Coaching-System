def generate_squat_report(angles, state):
    """
    Ban Cố Vấn Y Khoa Bài Tập: SQUAT
    Chỉ bung thông báo Nông/Sâu ngay tại Giai đoạn Điểm Đáy Đảo Chiều (Ascending).
    
    Args:
        angles (dict): Kho tàng Tọa độ góc lấy từ heuristics.
        state (dict): Nắp đậy Hộp đen Trạng thái Memory của hệ thống (chứa phase, min_knee_angle).
        
    Returns:
        list: Chuỗi các bản án (Tên Lỗi & Đoạn văn khuyên nhủ).
    """
    errors = []
    current_phase = state.get("phase", "standing")
    min_knee = state.get("min_knee_angle", 180.0)
    torso = angles.get("torso_angle", 90.0)
    
    # 1. Bệnh Lý Đứng Khóa Sụn Thẳng Tắp
    if current_phase == "standing" and angles.get("knee_angle", 180.0) > 175:
        errors.append({
            "error_code": "knee_locked_danger",
            "suggestion": "Bạn đang Khóa Đầu Gối đứng thẳng băng. Hãy chùng 2 đầu gối xuống 1 tẹo để tạ lọt cơ đùi, kẻo gãy sụn chêm!"
        })

    # 2. Bệnh Chấn Thương Nông Đáy - Chỉ nổ khi bạn Rời Đáy!
    if current_phase == "ascending": # Chuyển pha đẩy lên
        if min_knee > 105:
             errors.append({
                "error_code": "squat_insufficient_depth",
                "suggestion": f"REP VỪA RỒI THẤT BẠI: Bạn ép khối Tâm xuống quá nông (Góc thấp nhất {int(min_knee)} độ). Yêu cầu Hông lún song song sàn nhà."
            })
            
    # 3. Lỗi Sập Lưng Chó (Buttwink / Good Morning) ở Tận cùng Đáy Cực
    if current_phase in ["bottom", "descending"] and torso > 60: # Torso angle so với đỉnh chóp đầu, > 60 là cúi lố
        errors.append({
            "error_code": "good_morning_squat_collapse",
            "suggestion": "CẢNH BÁO MẤT CORE LƯNG DƯỚI: Bạn đang cúi rập đầu xuống đất chứ không gập gối. Rất dễ Thoát vị đĩa đệm! Hãy hất ưỡn Ngực Phổi lên!"
        })
        
    return errors

def generate_deadlift_report(angles, state):
    """
    Ban Cố Vấn Y Khoa Bài Tập: DEADLIFT
    Canh me độ cứng cốt lõi lúc cất tạ.
    """
    errors = []
    current_phase = state.get("phase", "setup")
    
    # Ở giai đoạn Setup đang định kéo vút phi tiêu
    if current_phase == "pulling":
        # Check Nhổm Đít Sớm (Hông Bắn Lên mà Vai đứng yên)
        if state.get("prev_hip_y", 0) > 0:
            hip_move_up = state["prev_hip_y"] - angles.get("hip_y", 0) # Tọa độ máy tính Y hướng xuống đất
            shoulder_move_up = state["prev_shoulder_y"] - angles.get("shoulder_y", 0)
            
            if hip_move_up > 25 and shoulder_move_up < 5:
                errors.append({
                    "error_code": "hips_shoot_up_early",
                    "suggestion": "BẠN TRƯỢT TẠ LỖI NHIỆP: Mông bị đội hất lên cao nhưng Vai Lưng lại chưa đẩy tạ đi. Dồn hạ trọng tâm sâu xuống, kéo tạ cọ sát ống chân!"
                })

    # Trầm trọng Lưng Tôm - Lỗi kinh tởm nhất giới đồ tạ
    torso_tilt = angles.get("torso_angle", 0)
    if current_phase == "pulling" and torso_tilt > 75:  # Cong gập lưng quá sâu so với cột sống dọc
        errors.append({
            "error_code": "cat_back_lumbar_loss",
            "suggestion": "TỬ THẬN: Cột sống của bạn cong thành Lưng Tôm chữ C gập cúp rất thô! Rút tạ ngay đi, gập siết lõi nín chặt bụng để khóa Phẳng chiếc Lưng!"
        })
        
    return errors

def generate_benchpress_report(angles, state):
    """
    Ban Cố Vấn Y Khoa Bài Tập: ĐẨY NGỰC BENCH PRESS
    Đo góc dập thanh tạ ngang xuống mặt và góc nách dồn chóp xoay Vai.
    """
    errors = []
    current_phase = state.get("phase", "lockout")
    min_elbow = state.get("min_elbow_angle", 180.0)
    flare = angles.get("shoulder_flare_angle", 0.0)
    
    if current_phase in ["lowering", "bottom"] and flare > 80:
        errors.append({
            "error_code": "flared_elbows_rotator_damage",
            "suggestion": f"BẠN BANH 2 BỜ NÁCH VỀ VUÔNG GÓC RỒI ({int(flare)} độ). Lực ngã xuống sẽ ép Lủng màn Chóp Xoay Vai! Hãy Xoắn ép 2 Cùi chỏ Vô Sườn Cắt tạo góc chữ V khoảng 45 độ!"
        })
        
    if current_phase == "pressing": 
        if min_elbow > 100:
             errors.append({
                "error_code": "bench_half_rep_shallow",
                "suggestion": "Chưa hạ đủ sâu ăn cơ ngực. Bạn phải để thanh đòn mút dây thép chạm nhẹ vào lồng ngực rồi mới nén đẩy lên!"
            })
            
    if current_phase == "lockout" and angles.get("elbow_angle", 180) > 175:
        errors.append({
            "error_code": "elbow_violent_lockout",
            "suggestion": "Giữ góc khuỷu mập mờ chùng một xíu trên ngực, đừng đẩy lẹ Khóa cạch cùi Chỏ thẳng tắp. Đứt Đai tay đấy."
        })

    return errors

def generate_form_report(exercise_type, angles, state):
    """
    Cổng Router Thích Ứng (Adaptive Evaluator) Router Trung Sở.
    Phân làn xem đang được Mô Hình AI giao môn nào thì cấp Kỷ Luật Môn Đó lấy từ kho y tế.
    
    Args:
        exercise_type (str): Chuỗi báo hiệu Tên Ván Tập (Từ YOLO Cls model phán).
        angles (dict): Máy Đo Góc.
        state (dict): Phao nhắm Điểm (Vùng Đáy - Vùng Nhô).
        
    Returns:
        list: Phân giải toàn bộ Lỗi Văn Bản Json.
    """
    rule_machine = exercise_type.lower()
    
    if rule_machine == "squat":
        return generate_squat_report(angles, state)
    elif rule_machine == "deadlift":
        return generate_deadlift_report(angles, state)
    elif rule_machine == "bench_pressing":
        return generate_benchpress_report(angles, state)
        
    # Môn Isolation Không Lập Kỷ Luật (Tạm thờn cho an tâm qua ải)
    return []
