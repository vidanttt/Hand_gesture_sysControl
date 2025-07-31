import sys
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt

class IronManHUD(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Iron Man HUD')
        self.setGeometry(100, 100, 1280, 720)
        self.setStyleSheet('background-color: #10131a;')

        # Main layout: video on left, HUD overlays on right
        main_layout = QHBoxLayout()

        # Video feed placeholder (replace with actual video widget in integration)
        video_label = QLabel(self)
        video_label.setFixedSize(800, 600)
        video_label.setStyleSheet('background-color: rgba(20,30,40,180); border-radius: 20px; border: 2px solid #00ffff;')
        video_label.setAlignment(Qt.AlignCenter)
        video_label.setText('VIDEO FEED')
        video_label.setFont(QFont('Orbitron', 24, QFont.Bold))
        video_label.setStyleSheet(video_label.styleSheet() + 'color: #00ffff;')

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
        status_label = QLabel(hud_overlay)
        status_label.setText('HUD ONLINE')
        status_label.setFont(QFont('Orbitron', 28, QFont.Bold))
        status_label.setStyleSheet('color: #00ffff; background: rgba(20,30,40,180); border-radius: 12px; padding: 12px; border: 2px solid #00ffff;')
        status_label.setAlignment(Qt.AlignCenter)
        hud_layout.addWidget(status_label)

        # Volume and brightness meters
        vol_label = QLabel(hud_overlay)
        vol_label.setText('VOLUME: 50%')
        vol_label.setFont(QFont('Orbitron', 22, QFont.Bold))
        vol_label.setStyleSheet('color: #00ffff; background: rgba(20,30,40,180); border-radius: 12px; padding: 8px; border: 2px solid #00ffff;')
        vol_label.setAlignment(Qt.AlignCenter)
        hud_layout.addWidget(vol_label)

        bright_label = QLabel(hud_overlay)
        bright_label.setText('BRIGHTNESS: 100%')
        bright_label.setFont(QFont('Orbitron', 22, QFont.Bold))
        bright_label.setStyleSheet('color: #ffaa00; background: rgba(20,30,40,180); border-radius: 12px; padding: 8px; border: 2px solid #ffaa00;')
        bright_label.setAlignment(Qt.AlignCenter)
        hud_layout.addWidget(bright_label)

        hud_overlay.setLayout(hud_layout)

        main_layout.addWidget(video_label)
        main_layout.addWidget(hud_overlay)
        self.setLayout(main_layout)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = IronManHUD()
    window.show()
    sys.exit(app.exec_())
