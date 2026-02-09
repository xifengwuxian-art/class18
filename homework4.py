import cv2, mediapipe as mp, time, numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

filters = [None, 'GRAYSCALE', 'SEPIA', 'NEGATIVE', 'BLUR']
current_filter = 0

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam not accessible.")
    exit()

last_action_time = 0
debounce_time = 1
pinch_in_progress = False
capture_request = False

def apply_filter(frame, f_type):
    if f_type == 'GRAYSCALE':
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif f_type == 'SEPIA':
        sepia_filter = np.array([[0.272, 0.534, 0.131], 
                                 [0.349, 0.686, 0.168], 
                                 [0.393, 0.769, 0.189]])
        return np.clip(cv2.transform(frame, sepia_filter), 0, 255).astype(np.uint8)
    elif f_type == 'NEGATIVE':
        return cv2.bitwise_not(frame)
    elif f_type == 'BLUR':
        return cv2.GaussianBlur(frame, (15, 15), 0)
    return frame

while True:
    ret, img = cap.read()
    if not ret:
        break
    
    img = cv2.flip(img, 1)  # Mirror effect
    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    capture_request = False
    
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)
            lm = hand.landmark
            # Build a dictionary of finger tip pixel coordinates
            tips = {name: (int(lm[idx].x * w), int(lm[idx].y * h))
                    for name, idx in {
                        'thumb': mp_hands.HandLandmark.THUMB_TIP,
                        'index': mp_hands.HandLandmark.INDEX_FINGER_TIP,
                        'middle': mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                        'ring': mp_hands.HandLandmark.RING_FINGER_TIP,
                        'pinky': mp_hands.HandLandmark.PINKY_TIP
                    }.items()}
            # Draw each finger tip
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
            for i, (name, (x, y)) in enumerate(tips.items()):
                cv2.circle(img, (x, y), 10, colors[i], cv2.FILLED)

            thumb_x, thumb_y = tips['thumb']
            index_x, index_y = tips['index']
            current_time = time.time()
            # Detect thumb-index pinch (for photo capture)
            pinch = abs(thumb_x - index_x) < 30 and abs(thumb_y - index_y) < 30
            if abs(thumb_x - index_x) < 30 and abs(thumb_y - index_y) < 30:
                action = "capture"
            elif any(abs(thumb_x - tips[finger][0]) < 30 and abs(thumb_y - tips[finger][1]) < 30
                for finger in ['middle', 'ring', 'pinky']):
                    action = "switch_filter"

            # Detect filter change gesture: thumb near any of middle, ring, or pinky
            elif any(abs(thumb_x - tips[finger][0]) < 30 and abs(thumb_y - tips[finger][1]) < 30
                    for finger in ['middle', 'ring', 'pinky']):
                    current_time = time.time()
                    if current_time - last_action_time < debounce_time:
                        continue  # 跳过本轮循环，不触发动作
    filter_img = apply_filter(img, filters[current_filter])
    
    display_img = cv2.cvtColor(filter_img, cv2.COLOR_GRAY2BGR) if filters[current_filter]=='GRAYSCALE' else filter_img
    if capture_request:
        cv2.putText(display_img, f"Filter: {filters[current_filter]}", 
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(display_img, "Picture Captured!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        ts = int(time.time())
        cv2.imwrite(f"picture_{ts}.jpg", display_img)
        print(f"Saved: picture_{ts}.jpg")
    cv2.imshow("Gesture-Controlled Photo App", display_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release(); 
cv2.destroyAllWindows()