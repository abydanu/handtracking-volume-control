import cv2
import mediapipe as mp
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
import numpy as np

# Inisialisasi MediaPipe Hands dan OpenCV
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Inisialisasi pycaw untuk mengontrol volume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

# Dapatkan nilai volume range
vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Konversi warna ke RGB untuk MediaPipe
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Gambar landmarks dengan warna hijau modern
            mp_draw.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(22, 121, 54), thickness=2, circle_radius=4),  # Hijau gelap
                mp_draw.DrawingSpec(color=(44, 250, 113), thickness=2, circle_radius=2)  # Hijau terang
            )
            
            landmarks = hand_landmarks.landmark
            
            thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            thumb_tip_coords = (int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0]))
            index_tip_coords = (int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0]))
            
            # Gambar garis antara thumb dan index dengan warna hijau
            cv2.line(frame, thumb_tip_coords, index_tip_coords, (144, 238, 144), 3)  # Light green
            
            # Hitung jarak dan sesuaikan range
            distance = np.sqrt((thumb_tip_coords[0] - index_tip_coords[0])**2 +
                             (thumb_tip_coords[1] - index_tip_coords[1])**2)
            
            # Sesuaikan range jarak untuk volume yang lebih natural
            normalized_distance = np.interp(distance, [20, 150], [min_vol, max_vol])
            volume_percentage = np.interp(distance, [20, 150], [0, 100])
            
            # Set volume
            volume.SetMasterVolumeLevel(normalized_distance, None)
            
            # Modern volume bar dengan gradien effect
            bar_width = 40
            bar_x = 50
            
            # Background bar (semi-transparent)
            cv2.rectangle(frame, (bar_x, 150), (bar_x + bar_width, 400), (40, 40, 40), cv2.FILLED)
            cv2.rectangle(frame, (bar_x, 150), (bar_x + bar_width, 400), (100, 100, 100), 2)
            
            # Volume level bar dengan gradien hijau
            bar_height = int(np.interp(volume_percentage, [0, 100], [400, 150]))
            cv2.rectangle(frame, (bar_x, bar_height), (bar_x + bar_width, 400), 
                         (144, 238, 144), cv2.FILLED)  # Light green
            
            # Tambah efek highlight
            cv2.line(frame, (bar_x + 2, bar_height), (bar_x + 2, 400), 
                    (255, 255, 255), 2)
            
            # Volume percentage dengan style modern
            percentage_text = f'{int(volume_percentage)}%'
            text_size = cv2.getTextSize(percentage_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = bar_x + (bar_width - text_size[0]) // 2
            
            # Text shadow effect
            cv2.putText(frame, percentage_text, 
                       (text_x + 2, 450 + 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (40, 40, 40), 2)
            cv2.putText(frame, percentage_text, 
                       (text_x, 450), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (144, 238, 144), 2)  # Light green
            
            # Modern finger indicators
            # Outer glow effect
            cv2.circle(frame, thumb_tip_coords, 12, (40, 40, 40), cv2.FILLED)
            cv2.circle(frame, index_tip_coords, 12, (40, 40, 40), cv2.FILLED)
            # Inner circle
            cv2.circle(frame, thumb_tip_coords, 8, (144, 238, 144), cv2.FILLED)  # Light green
            cv2.circle(frame, index_tip_coords, 8, (144, 238, 144), cv2.FILLED)  # Light green

    # Modern header dengan gradient background
    header_height = 90
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], header_height), (40, 40, 40), cv2.FILLED)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    title_text = 'Hand Volume Control'
    cv2.putText(frame, title_text, (12, 32), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (40, 40, 40), 2)
    cv2.putText(frame, title_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (144, 238, 144), 2)  # Light green
    
    subtitle_text = 'Press Q to quit'
    cv2.putText(frame, subtitle_text, (12, 72), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)
    cv2.putText(frame, subtitle_text, (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (152, 251, 152), 2)  # Pale green

    cv2.imshow('Hand Volume Control', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
