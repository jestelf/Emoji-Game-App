import sys
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QHBoxLayout, QVBoxLayout, QWidget
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap, QImage


# Инициализируем MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_pose_landmarks(image_bgr):
    """
    Возвращает список координат (x, y) для каждой ключевой точки скелета.
    Если скелет не найден, возвращает None.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if not results.pose_landmarks:
        return None
    landmarks = []
    for lm in results.pose_landmarks.landmark:
        landmarks.append((lm.x, lm.y))
    return landmarks

def compare_poses(landmarks1, landmarks2, threshold=0.15):
    """
    Сравнивает два набора координат (x, y) скелета.
    Возвращает True, если они достаточно близки друг к другу; иначе False.

    По умолчанию threshold=0.15 (15%), что более лояльно к неточному воспроизведению позы.
    """
    if landmarks1 is None or landmarks2 is None:
        return False
    if len(landmarks1) != len(landmarks2):
        return False

    total_dist = 0.0
    for (x1, y1), (x2, y2) in zip(landmarks1, landmarks2):
        dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        total_dist += dist
    
    avg_dist = total_dist / len(landmarks1)
    return (avg_dist < threshold)

def draw_pose_on_image(image_bgr, landmarks, color=(0, 0, 255)):
    """
    Рисует скелет поверх BGR-изображения.
    color - в формате (B, G, R).
    """
    if landmarks is None:
        return image_bgr

    pseudo_landmarks = landmark_pb2.NormalizedLandmarkList(
        landmark=[
            landmark_pb2.NormalizedLandmark(x=x, y=y)
            for (x, y) in landmarks
        ]
    )

    annotated_image = image_bgr.copy()
    mp_drawing.draw_landmarks(
        annotated_image,
        pseudo_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2)
    )
    return annotated_image


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Сравнение позы с фото — MediaPipe + PyQt5")
        self.resize(1280, 720)

        # ---------- Главный виджет и лайаут ----------
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # ---------- Левая часть: камера ----------
        self.camera_label = QLabel("Видео с камеры")
        self.camera_label.setAlignment(Qt.AlignCenter)

        # ---------- Правая часть: фото и сообщение ----------
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # 1) Лейбл для отображения эталонного фото
        self.reference_label = QLabel("Эталонная поза")
        self.reference_label.setAlignment(Qt.AlignCenter)

        # 2) Лейбл для сообщения "Молодцы!"
        self.message_label = QLabel("")
        self.message_label.setAlignment(Qt.AlignCenter)
        self.message_label.setStyleSheet("font-size: 24px; color: green;")

        right_layout.addWidget(self.reference_label)
        right_layout.addWidget(self.message_label)

        main_layout.addWidget(self.camera_label, stretch=3)
        main_layout.addWidget(right_widget, stretch=1)

        # ========== Шаг 1: Загрузка эталонного изображения ==========
        self.reference_image_bgr = cv2.imread("pose.jpg")  # <-- Замените на ваш путь
        if self.reference_image_bgr is None:
            raise FileNotFoundError("Невозможно загрузить pose.jpg")

        # Находим скелет эталонной позы
        self.reference_landmarks = extract_pose_landmarks(self.reference_image_bgr)

        # Рисуем его (пусть будет зеленым) на копии
        annotated_ref = draw_pose_on_image(self.reference_image_bgr, self.reference_landmarks, color=(0, 255, 0))

        # Отображаем эталонное изображение в self.reference_label
        ref_rgb = cv2.cvtColor(annotated_ref, cv2.COLOR_BGR2RGB)
        h, w, ch = ref_rgb.shape
        bytes_per_line = ch * w
        ref_qimage = QImage(ref_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.reference_label.setPixmap(QPixmap.fromImage(ref_qimage))

        # ========== Шаг 2: Инициализация камеры ==========
        self.cap = cv2.VideoCapture(0)

        # ========== Шаг 3: Таймер обновления ==========
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~30 кадров/с

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Зеркалим кадр для удобства пользователя
        frame = cv2.flip(frame, 1)

        # Извлекаем скелет пользователя
        user_landmarks = extract_pose_landmarks(frame)

        # Сравниваем позы (с увеличенным порогом)
        is_same = compare_poses(user_landmarks, self.reference_landmarks, threshold=0.15)

        # Если совпадают, цвет скелета = зелёный, иначе красный
        color = (0, 255, 0) if is_same else (0, 0, 255)

        # Рисуем скелет пользователя
        annotated_frame = draw_pose_on_image(frame, user_landmarks, color=color)

        # Пишем сообщение "Молодцы!", если поза совпала
        if is_same:
            self.message_label.setText("Молодцы!")
        else:
            self.message_label.setText("")

        # Показываем результат на камере
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(qt_image))

    def closeEvent(self, event):
        """При закрытии окна освобождаем камеру."""
        self.cap.release()
        event.accept()


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
