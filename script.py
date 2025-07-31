import cv2
import mediapipe as mp
import numpy as np
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from monitorcontrol import get_monitors
import time
import sys
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QProgressBar, QFrame, QStackedLayout
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QColor, QPen, QConicalGradient
from PyQt5.QtCore import QTimer, Qt, QRectF
import random

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
#testing

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def interpolate(val, in_min, in_max, out_min, out_max):
    """Linear interpolation function"""
    return out_min + (out_max - out_min) * ((val - in_min) / (in_max - in_min))

def calculate_normalized_distance(lm1, lm2):
    return math.sqrt(
        (lm1.x - lm2.x) ** 2 +
        (lm1.y - lm2.y) ** 2 +
        (lm1.z - lm2.z) ** 2
    )

def draw_volume_bar(img, vol_percentage, vol_bar_x=50, vol_bar_y=150, vol_bar_width=50, vol_bar_height=200):
    """Draw volume bar on the image"""
    # Draw volume bar background
    cv2.rectangle(img, (vol_bar_x, vol_bar_y), (vol_bar_x + vol_bar_width, vol_bar_y + vol_bar_height), (0, 0, 0), 3)
    
    # Draw volume level
    vol_fill_height = int(vol_bar_height * vol_percentage / 100)
    cv2.rectangle(img, (vol_bar_x, vol_bar_y + vol_bar_height - vol_fill_height), 
                  (vol_bar_x + vol_bar_width, vol_bar_y + vol_bar_height), (0, 255, 0), -1)
    
    # Draw volume percentage text
    cv2.putText(img, f'{int(vol_percentage)}%', (vol_bar_x - 10, vol_bar_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add label
    cv2.putText(img, 'Volume', (vol_bar_x - 10, vol_bar_y + vol_bar_height + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

def draw_brightness_bar(img, brightness_percentage, brightness_bar_x=120, brightness_bar_y=150, brightness_bar_width=50, brightness_bar_height=200):
    """Draw brightness bar on the image"""
    # Draw brightness bar background
    cv2.rectangle(img, (brightness_bar_x, brightness_bar_y), (brightness_bar_x + brightness_bar_width, brightness_bar_y + brightness_bar_height), (0, 0, 0), 3)
    
    # Draw brightness level
    brightness_fill_height = int(brightness_bar_height * brightness_percentage / 100)
    cv2.rectangle(img, (brightness_bar_x, brightness_bar_y + brightness_bar_height - brightness_fill_height), 
                  (brightness_bar_x + brightness_bar_width, brightness_bar_y + brightness_bar_height), (255, 255, 0), -1)
    
    # Draw brightness percentage text
    cv2.putText(img, f'{int(brightness_percentage)}%', (brightness_bar_x - 10, brightness_bar_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add label
    cv2.putText(img, 'Brightness', (brightness_bar_x - 20, brightness_bar_y + brightness_bar_height + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

def draw_modern_hand_overlay(frame, landmarks, color, highlight_indices=None, pulse=False):
    """Draw minimal, techy hand overlay: thin green lines, small red dots, thin font numbers, and a line between thumb and index tip."""
    # Draw connections (very thin green lines)
    connections = mp_hands.HAND_CONNECTIONS
    for start, end in connections:
        pt1 = tuple(landmarks[start])
        pt2 = tuple(landmarks[end])
        cv2.line(frame, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)  # very thin green line
    # Draw thumb-index line (very thin, black)
    if len(landmarks) > 8:
        cv2.line(frame, tuple(landmarks[4]), tuple(landmarks[8]), (227, 154, 216), 1, cv2.LINE_AA)
    # Draw landmarks (small, thin red dots)
    for idx, (x, y) in enumerate(landmarks):
        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)  # small red dot
        # Landmark number (thin, small, techy font)
        cv2.putText(frame, str(idx), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

class HUDWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.volume = 50
        self.brightness = 50
        self.sci_fi_font = QFont('Orbitron', 18, QFont.Bold)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(30)
        self.angle_anim = 0
        self.angle_anim2 = 0
        self.hand_status = "No Hands Detected"
        self.left_hand = False
        self.right_hand = False
        self.status_glow = 0

    def set_values(self, volume, brightness, left_hand=False, right_hand=False):
        self.volume = volume
        self.brightness = brightness
        self.left_hand = left_hand
        self.right_hand = right_hand
        if left_hand and right_hand:
            self.hand_status = "BOTH HANDS ONLINE"
        elif left_hand:
            self.hand_status = "VOLUME CONTROL ONLINE"
        elif right_hand:
            self.hand_status = "BRIGHTNESS CONTROL ONLINE"
        else:
            self.hand_status = "SCANNING..."
        self.status_glow = 255 if (left_hand or right_hand) else 80
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        center_x = w - 200
        center_y = 200
        # Animated angles for rotating rings
        self.angle_anim = (self.angle_anim + 2) % 360
        self.angle_anim2 = (self.angle_anim2 + 1.2) % 360
        # Draw animated neon rings for volume and brightness
        self.draw_animated_rings(painter, center_x, center_y, 120, QColor(0,255,255,120), self.angle_anim)
        self.draw_animated_rings(painter, center_x, center_y, 90, QColor(0,255,255,60), -self.angle_anim2)
        self.draw_animated_rings(painter, center_x, center_y+220, 120, QColor(255,180,0,120), -self.angle_anim)
        self.draw_animated_rings(painter, center_x, center_y+220, 90, QColor(255,180,0,60), self.angle_anim2)
        # Draw glowing circular meters
        self.draw_circular_meter(painter, center_x, center_y, 120, self.volume, QColor(0, 255, 255, 220), 'VOLUME', QColor(0, 255, 255), self.angle_anim)
        self.draw_circular_meter(painter, center_x, center_y + 220, 120, self.brightness, QColor(255, 180, 0, 220), 'BRIGHTNESS', QColor(255, 180, 0), -self.angle_anim)
        # Draw pulsing dots
        self.draw_pulsing_dots(painter, center_x, center_y, 120, QColor(0,255,255), self.angle_anim)
        self.draw_pulsing_dots(painter, center_x, center_y+220, 120, QColor(255,180,0), -self.angle_anim)
        # Draw glassy panels for live data
        self.draw_glass_panel(painter, 40, 40, 340, 90)
        painter.setFont(QFont('Orbitron', 22, QFont.Bold))
        painter.setPen(QColor(0, 255, 255, 220))
        painter.drawText(60, 90, f'VOLUME: {int(self.volume)}%')
        painter.setPen(QColor(255, 180, 0, 220))
        painter.drawText(220, 90, f'BRIGHT: {int(self.brightness)}%')
        # Draw animated hand status panel
        self.draw_glass_panel(painter, 40, 150, 340, 70)
        painter.setFont(QFont('Orbitron', 18, QFont.Bold))
        glow = self.status_glow
        status_color = QColor(0,255,255,glow) if self.left_hand else QColor(255,180,0,glow) if self.right_hand else QColor(255,0,0,glow)
        pen = QPen(status_color)
        pen.setWidth(4)
        painter.setPen(pen)
        painter.drawText(60, 190, self.hand_status)
        # Draw animated JARVIS greeting
        painter.setFont(QFont('Orbitron', 30, QFont.Bold))
        painter.setPen(QColor(0,255,255,180))
        painter.drawText(60, 250, "Welcome, Sir. HUD Online.")
        # Draw animated grid lines for extra sci-fi effect
        self.draw_grid_lines(painter, w, h)
    def draw_grid_lines(self, painter, w, h):
        painter.save()
        pen = QPen(QColor(0,255,255,40))
        pen.setWidth(1)
        painter.setPen(pen)
        for x in range(0, w, 80):
            painter.drawLine(x, 0, x, h)
        for y in range(0, h, 80):
            painter.drawLine(0, y, w, y)
        painter.restore()

    def draw_circular_meter(self, painter, cx, cy, radius, value, glow_color, label, text_color, angle_anim):
        # Outer glow ring
        for i in range(10, 0, -2):
            pen = QPen(glow_color)
            pen.setWidth(i)
            pen.setColor(glow_color.lighter(120 + i*5))
            painter.setPen(pen)
            painter.drawEllipse(QRectF(cx - radius, cy - radius, 2*radius, 2*radius))
        # Animated rotating ring
        pen = QPen(glow_color)
        pen.setWidth(4)
        painter.setPen(pen)
        painter.drawArc(QRectF(cx - radius + 10, cy - radius + 10, 2*(radius-10), 2*(radius-10)), int(angle_anim*16), 90*16)
        # Meter arc
        pen = QPen(text_color)
        pen.setWidth(10)
        painter.setPen(pen)
        span_angle = int(360 * value / 100)
        painter.drawArc(QRectF(cx - radius + 20, cy - radius + 20, 2*(radius-20), 2*(radius-20)), 90*16, -span_angle*16)
        # Center label
        painter.setFont(QFont('Orbitron', 16, QFont.Bold))
        painter.setPen(text_color)
        painter.drawText(cx - 60, cy + 10, 120, 40, Qt.AlignCenter, label)
        # Value
        painter.setFont(QFont('Orbitron', 24, QFont.Bold))
        painter.drawText(cx - 60, cy - 40, 120, 40, Qt.AlignCenter, f'{int(value)}%')

    def draw_glass_panel(self, painter, x, y, w, h):
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(30, 60, 90, 180))
        painter.drawRoundedRect(QRectF(x, y, w, h), 20, 20)
        # Neon border
        pen = QPen(QColor(0, 255, 255, 180))
        pen.setWidth(3)
        painter.setPen(pen)
        painter.drawRoundedRect(QRectF(x, y, w, h), 20, 20)

    def draw_animated_rings(self, painter, cx, cy, radius, color, angle):
        painter.save()
        painter.translate(cx, cy)
        painter.rotate(angle)
        pen = QPen(color)
        pen.setWidth(3)
        painter.setPen(pen)
        for i in range(3):
            painter.drawEllipse(QRectF(-radius+i*7, -radius+i*7, 2*(radius-i*7), 2*(radius-i*7)))
        painter.restore()

    def draw_pulsing_dots(self, painter, cx, cy, radius, color, angle):
        painter.save()
        painter.translate(cx, cy)
        t = time.time()
        for i in range(6):
            a = angle + i*60
            rad = math.radians(a)
            x = math.cos(rad) * radius
            y = math.sin(rad) * radius
            pulse = 8 + 4 * (1 + math.sin(t*2 + i))
            painter.setBrush(color)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QRectF(x-pulse/2, y-pulse/2, pulse, pulse))
        painter.restore()

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('JARVIS Hand Gesture Control')
        self.setGeometry(100, 100, 1280, 720)
        self.setStyleSheet('background-color: #10131a;')

        # Main layout: video on left, HUD overlays on right
        main_layout = QHBoxLayout()

        # Video feed label (actual video will be shown here)
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(800, 600)
        self.video_label.setStyleSheet('background-color: rgba(20,30,40,180); border-radius: 20px; border: 2px solid #00ffff;')
        self.video_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.video_label)

        # HUD overlays on right
        hud_overlay = QWidget(self)
        hud_layout = QVBoxLayout()
        hud_layout.setContentsMargins(40, 40, 40, 40)
        hud_layout.setSpacing(40)

        # HUD image as semi-transparent background
        hud_img_label = QLabel(hud_overlay)
        hud_pixmap = QPixmap('Iron Man HUD Template/EditingCorp Iron man HUD.png')
        hud_pixmap = hud_pixmap.scaled(350, 350, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        hud_img_label.setPixmap(hud_pixmap)
        hud_img_label.setAlignment(Qt.AlignCenter)
        hud_img_label.setStyleSheet('background: transparent;')
        hud_layout.addWidget(hud_img_label)

        # Futuristic status panel
        self.status_label = QLabel(hud_overlay)
        self.status_label.setText('HUD ONLINE')
        self.status_label.setFont(QFont('Orbitron', 28, QFont.Bold))
        self.status_label.setStyleSheet('color: #00ffff; background: rgba(20,30,40,180); border-radius: 12px; padding: 12px; border: 2px solid #00ffff;')
        self.status_label.setAlignment(Qt.AlignCenter)
        hud_layout.addWidget(self.status_label)

        # Volume and brightness meters
        self.vol_label = QLabel(hud_overlay)
        self.vol_label.setText('VOLUME: 50%')
        self.vol_label.setFont(QFont('Orbitron', 22, QFont.Bold))
        self.vol_label.setStyleSheet('color: #00ffff; background: rgba(20,30,40,180); border-radius: 12px; padding: 8px; border: 2px solid #00ffff;')
        self.vol_label.setAlignment(Qt.AlignCenter)
        hud_layout.addWidget(self.vol_label)

        self.bright_label = QLabel(hud_overlay)
        self.bright_label.setText('BRIGHTNESS: 100%')
        self.bright_label.setFont(QFont('Orbitron', 22, QFont.Bold))
        self.bright_label.setStyleSheet('color: #ffaa00; background: rgba(20,30,40,180); border-radius: 12px; padding: 8px; border: 2px solid #ffaa00;')
        self.bright_label.setAlignment(Qt.AlignCenter)
        hud_layout.addWidget(self.bright_label)

        hud_overlay.setLayout(hud_layout)
        main_layout.addWidget(hud_overlay)
        self.setLayout(main_layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.last_vol_percentage = 50
        self.last_vol_distance = 0
        self.left_hand_detected = False
        self.last_brightness_percentage = 50
        self.last_brightness_distance = 0
        self.right_hand_detected = False
        self.last_sent_brightness = 50
        self.last_brightness_update = 0
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        vol_distance = self.last_vol_distance
        vol_percentage = self.last_vol_percentage
        brightness_distance = self.last_brightness_distance
        brightness_percentage = self.last_brightness_percentage
        left_hand_detected = False
        right_hand_detected = False
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Swap hand label due to horizontal flip
                raw_label = results.multi_handedness[hand_idx].classification[0].label
                hand_label = "Right" if raw_label == "Left" else "Left"
                landmarks = []
                for lm in hand_landmarks.landmark:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    landmarks.append([x, y])
                # Use normalized 3D distance for gesture logic
                norm_distance = calculate_normalized_distance(hand_landmarks.landmark[4], hand_landmarks.landmark[8])
                min_norm_distance = 0.03
                max_norm_distance = 0.18
                norm_distance_clamped = max(min_norm_distance, min(max_norm_distance, norm_distance))
                percentage = interpolate(norm_distance_clamped, min_norm_distance, max_norm_distance, 0, 100)
                thumb_tip = landmarks[4]
                index_tip = landmarks[8]
                # Ring finger tip (landmark 16)
                ring_mcp = landmarks[13]
                ring_pip = landmarks[14]
                # Use midpoint between MCP and PIP for ring detection (just above MCP)
                rx = int((ring_mcp[0] + ring_pip[0]) / 2)
                ry = int((ring_mcp[1] + ring_pip[1]) / 2)
                # Only detect left hand if black ring is present on ring finger
                left_hand_valid = False
                if hand_label == "Left":
                    # Crop region just above MCP (midpoint between MCP and PIP)
                    crop_size = 24  # smaller crop for precision
                    x1 = max(rx - crop_size, 0)
                    y1 = max(ry - crop_size, 0)
                    x2 = min(rx + crop_size, w)
                    y2 = min(ry + crop_size, h)
                    # Draw crop region for debug
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    ring_crop = frame[y1:y2, x1:x2]
                    if ring_crop.size > 0:
                        hsv = cv2.cvtColor(ring_crop, cv2.COLOR_BGR2HSV)
                        mask_black = cv2.inRange(hsv, (0, 0, 0), (180, 80, 50))
                        black_ratio = np.sum(mask_black > 0) / mask_black.size
                        avg_v = np.mean(hsv[:,:,2])
                        print(f"[DEBUG] black_ratio={black_ratio:.3f}, avg_v={avg_v:.1f}")
                        # Make detection stricter to reduce false positives
                        if black_ratio > 0.08 and avg_v < 120:
                            left_hand_valid = True
                    if left_hand_valid:
                        left_hand_detected = True
                        neon_green = (0, 255, 128)
                        draw_modern_hand_overlay(frame, landmarks, neon_green, highlight_indices=[4,8,13], pulse=True)
                        vol_distance = norm_distance
                        vol_percentage = percentage
                        vol_level = interpolate(vol_percentage, 0, 100, min_vol, max_vol)
                        volume.SetMasterVolumeLevel(vol_level, None)
                        self.last_vol_percentage = vol_percentage
                        self.last_vol_distance = vol_distance
                elif hand_label == "Right":
                    right_hand_detected = True
                    neon_blue = (0, 255, 255)
                    draw_modern_hand_overlay(frame, landmarks, neon_blue, highlight_indices=[4,8], pulse=True)
                    brightness_distance = norm_distance
                    brightness_percentage = percentage
                    try:
                        monitors = get_monitors()
                        current_time = time.time()
                        if monitors:
                            if abs(brightness_percentage - self.last_sent_brightness) >= 2 and (current_time - self.last_brightness_update > 0.2):
                                with monitors[0] as monitor:
                                    monitor.set_luminance(int(brightness_percentage))
                                self.last_sent_brightness = brightness_percentage
                                self.last_brightness_update = current_time
                        else:
                            print("No DDC/CI monitor found for brightness control.")
                    except Exception as e:
                        print(f"DDC/CI brightness control error: {e}")
                    self.last_brightness_percentage = brightness_percentage
                    self.last_brightness_distance = brightness_distance
        # Update HUD overlays
        self.status_label.setText(
            'BOTH HANDS ONLINE' if left_hand_detected and right_hand_detected else
            'VOLUME CONTROL ONLINE' if left_hand_detected else
            'BRIGHTNESS CONTROL ONLINE' if right_hand_detected else
            'SCANNING...')
        self.vol_label.setText(f'VOLUME: {int(vol_percentage)}%')
        self.bright_label.setText(f'BRIGHTNESS: {int(brightness_percentage)}%')
        # Show video
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())