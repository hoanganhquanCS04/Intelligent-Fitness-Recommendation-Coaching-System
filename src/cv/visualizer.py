import cv2

# Từ điển sơ đồ xương ghép nối (COCO Pose Format) - format: (điểm_bắt_đầu, điểm_kết_thúc)
SKELETON_CONNECTIONS = [
    (15, 13), (13, 11), (16, 14), (14, 12),  # Two legs
    (11, 12), (5, 11), (6, 12), (5, 6),      # Torso (Thân hình trụ)
    (5, 7), (7, 9), (6, 8), (8, 10),         # Arms (2 cánh tay)
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),  # Head (Đầu/Mặt)
    (3, 5), (4, 6)                           # Head to shoulders (Cổ xương đòn)
]

def draw_skeleton_overlay(image_np, kpts, conf_threshold=0.5):
    """
    Vẽ họa đè khung xương khớp (skeleton) lên ảnh chụp/video nguyên bản.
    
    Chức năng: Đưa ra trải nghiệm hình ảnh mượt mà cho User nhìn thấy máy đang theo dõi 
    khớp chuyển động ra sao. Vẽ các khúc xương màu xanh lá chói và điểm chốt màu đỏ ruby.
    
    Args:
        image_np (numpy.ndarray): Ma trận điểm ảnh hệ RGB/BGR (Ảnh lấy từ WebRTC máy người dùng).
        kpts (numpy.ndarray): Mảng numpy chứa các tọa độ điểm neo. Cấu trúc array: Tối thiểu Nx3 [X, Y, Độ_Tự_Tin].
        conf_threshold (float, optional): Cắt cúp độ mờ tịt. Nếu camera AI nhìn thấy tay/chân tỷ lệ vỡ nét (tự tin dưới 50%) 
                                          thì sẽ KHÔNG VẼ đoạn xương đó để tránh loằng ngoằng rác sinh ra màn hình. Mặc định 0.5.
        
    Returns:
        numpy.ndarray: Ma trận ảnh đầu ra y hệt matrix size RGB cũ nhưng đã được cọ vẽ lên các dải phân bổ xương, 
                       có thể trả trực tiếp cho Backend Render thành Base64 đưa lên React JS.
    """
    # Trích tạo bản sao (Clone) ảnh gốc để bảo vệ không khí bộ nhớ máy
    drawn_img = image_np.copy()
    
    # 1. Vẽ các đường nối trục hình học làm Khung xương cứng (Edges)
    for p1, p2 in SKELETON_CONNECTIONS:
        try:
            x1, y1, c1 = kpts[p1]
            x2, y2, c2 = kpts[p2]
            
            # Line filter: Cả 2 điểm neo sinh ra cành cây nối phải tự tin hơn conf_threshold
            if c1 >= conf_threshold and c2 >= conf_threshold:
                pt1 = (int(x1), int(y1))
                pt2 = (int(x2), int(y2))
                # Màu vệt: BGR (0, 255, 0) tức Xanh lá sáng, Độ dày nét kéo = 2px
                cv2.line(drawn_img, pt1, pt2, (0, 255, 0), 2)
        except IndexError:
            pass
            
    # 2. Rải đinh tán lên Khớp nối (Joints point mode đè chồng lên đường vạch thẳng)
    for kpt in kpts:
        try:
            x, y, conf = kpt
            if conf >= conf_threshold:
                # Chấm đỏ, bán kính chấm = 4px, Đổ đặc vòng bằng -1 pixel thickness
                cv2.circle(drawn_img, (int(x), int(y)), 4, (0, 0, 255), -1)
        except Exception:
            pass
            
    return drawn_img
