import os
import cv2
import random
import mediapipe as mp
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer, Qt, QRect
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QFont

# Инициализация Mediapipe для распознавания рук
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,          # Использование видеопотока, а не статичных изображений
    max_num_hands=2,                  # Максимальное количество рук для отслеживания
    min_detection_confidence=0.5,     # Минимальная уверенность для обнаружения руки
    min_tracking_confidence=0.5       # Минимальная уверенность для отслеживания руки
)
mp_drawing = mp.solutions.drawing_utils  # Утилиты для рисования распознанных ключевых точек

class Emoji:
    """
    Класс, представляющий эмодзи на экране.
    Управляет позицией, скоростью, вращением и состоянием эмодзи.
    """
    def __init__(self, x, y, emoji):
        self.x = x  # Координата X эмодзи
        self.y = y  # Координата Y эмодзи
        self.emoji = emoji  # Символ эмодзи
        self.vx = random.uniform(-2, 2)  # Случайная начальная скорость по оси X
        self.vy = 0  # Начальная скорость по оси Y
        self.radius = 30  # Радиус эмодзи для проверки коллизий
        self.angle = 0  # Текущий угол поворота эмодзи
        self.angular_velocity = random.uniform(-5, 5)  # Скорость вращения эмодзи
        self.burst = False  # Флаг состояния взрыва эмодзи
        self.burst_time = 0  # Время жизни состояния взрыва

    def update(self, gravity=0.5, width=800, height=600):
        """
        Обновляет состояние эмодзи: позицию, скорость и проверяет на столкновения с краями.
        
        :param gravity: Сила гравитации, влияющая на скорость по оси Y.
        :param width: Ширина окна приложения.
        :param height: Высота окна приложения.
        """
        if not self.burst:
            self.vy += gravity  # Применяем гравитацию к вертикальной скорости
            self.x += self.vx    # Обновляем позицию по оси X
            self.y += self.vy    # Обновляем позицию по оси Y
            self.angle += self.angular_velocity  # Обновляем угол поворота

            # Проверяем столкновение с краями окна
            if self.x - self.radius < 0 or self.x + self.radius > width or self.y + self.radius > height:
                self.burst = True       # Помечаем эмодзи для взрыва
                self.burst_time = 10    # Устанавливаем время жизни взрыва (количество обновлений)
        else:
            self.burst_time -= 1  # Уменьшаем время жизни взрыва

    def draw(self, painter):
        """
        Отрисовывает эмодзи на экране.
        
        :param painter: Объект QPainter для рисования.
        """
        if not self.burst:
            painter.save()  # Сохраняем текущее состояние painter
            painter.translate(int(self.x), int(self.y))  # Перемещаем систему координат к позиции эмодзи
            painter.rotate(self.angle)  # Вращаем систему координат на текущий угол
            rect = QRect(-self.radius, -self.radius, self.radius * 2, self.radius * 2)  # Создаём прямоугольник для эмодзи
            painter.drawText(rect, Qt.AlignCenter, self.emoji)  # Рисуем эмодзи в центре прямоугольника
            painter.restore()  # Восстанавливаем предыдущее состояние painter
        elif self.burst_time > 0:
            # Рисуем взрыв вместо эмодзи
            painter.setPen(QPen(Qt.yellow, 3, Qt.SolidLine))  # Устанавливаем жёлтый цвет и толщину пера
            painter.setFont(QFont("Arial", 24))  # Устанавливаем шрифт для взрыва
            rect = QRect(int(self.x - self.radius), int(self.y - self.radius), self.radius * 2, self.radius * 2)
            painter.drawText(rect, Qt.AlignCenter, "💥")  # Рисуем символ взрыва в центре прямоугольника

    def is_colliding_with_hand(self, landmarks, width, height):
        """
        Проверяет, столкнулся ли эмодзи с любой точкой руки.
        
        :param landmarks: Список ключевых точек (landmarks) руки.
        :param width: Ширина изображения.
        :param height: Высота изображения.
        :return: Координаты точки столкновения или None, если столкновения нет.
        """
        for lm in landmarks:
            hand_x = int(lm.x * width)  # Преобразуем нормализованные координаты X руки в пиксели
            hand_y = int(lm.y * height)  # Преобразуем нормализованные координаты Y руки в пиксели
            distance = ((self.x - hand_x) ** 2 + (self.y - hand_y) ** 2) ** 0.5  # Вычисляем расстояние до точки руки
            if distance < self.radius + 20:  # 20 пикселей — приблизительный радиус руки
                return (hand_x, hand_y)  # Возвращаем координаты точки столкновения
        return None  # Столкновения нет

class EmojiGameApp(QMainWindow):
    """
    Основной класс приложения, отвечающий за интерфейс, обработку видео и взаимодействие эмодзи с руками.
    """
    def __init__(self):
        super().__init__()

        # Настройка окна приложения
        self.setWindowTitle("Emoji Game App")  # Устанавливаем заголовок окна
        self.setGeometry(100, 100, 800, 600)  # Устанавливаем позицию и размер окна

        # Настройка интерфейса
        self.video_label = QLabel(self)  # Создаём QLabel для отображения видео
        self.video_label.setScaledContents(True)  # Разрешаем масштабирование содержимого QLabel
        layout = QVBoxLayout()  # Создаём вертикальный layout
        layout.addWidget(self.video_label)  # Добавляем QLabel в layout
        container = QWidget()  # Создаём контейнер для layout
        container.setLayout(layout)  # Устанавливаем layout для контейнера
        self.setCentralWidget(container)  # Устанавливаем контейнер как центральный виджет окна

        # Настройка захвата видео с камеры
        self.cap = cv2.VideoCapture(0)  # Открываем веб-камеру (устройство 0)
        self.timer = QTimer(self)  # Создаём таймер для обновления кадров
        self.timer.timeout.connect(self.update_frame)  # Подключаем метод обновления кадров к сигналу таймера
        self.timer.start(30)  # Запускаем таймер с интервалом 30 мс (примерно 33 кадра в секунду)

        # Настройка эмодзи
        self.emojis = ["😀", "✨", "🎉", "❤️", "🔥"]  # Список возможных эмодзи
        self.emoji_list = []  # Список текущих эмодзи на экране

    def update_frame(self):
        """
        Метод, вызываемый таймером для обновления кадра видео и состояния эмодзи.
        """
        ret, frame = self.cap.read()  # Захватываем кадр с камеры
        if not ret:
            return  # Если не удалось захватить кадр, выходим из метода

        # Убираем зеркальное отражение изображения
        frame = cv2.flip(frame, 1)

        # "Отдаление" камеры: добавляем границы для увеличения пространства
        scale = 1.5  # Коэффициент масштабирования
        h, w, _ = frame.shape  # Получаем высоту и ширину кадра
        new_h, new_w = int(h * scale), int(w * scale)  # Вычисляем новые размеры

        # Создаём пустое изображение большего размера с чёрными границами
        padded_frame = cv2.copyMakeBorder(
            frame, 
            top=(new_h - h) // 2, 
            bottom=(new_h - h) // 2, 
            left=(new_w - w) // 2, 
            right=(new_w - w) // 2, 
            borderType=cv2.BORDER_CONSTANT, 
            value=(0, 0, 0)  # Чёрный цвет для границ
        )

        # Обрезаем до оригинального размера для стабилизации изображения
        frame = padded_frame[
            (new_h - h) // 2 : (new_h + h) // 2, 
            (new_w - w) // 2 : (new_w + w) // 2
        ]

        # Обработка кадра с помощью Mediapipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Преобразуем цветовую схему из BGR в RGB
        results = hands.process(image)  # Обрабатываем изображение для обнаружения рук

        height, width, _ = image.shape  # Получаем размеры обработанного изображения

        # Случайное появление нового эмодзи
        if random.random() < 0.05:  # С вероятностью 5% на каждом кадре
            emoji_x = random.randint(30, width - 30)  # Случайная позиция X с отступом 30 пикселей от краёв
            emoji_y = 30  # Позиция Y устанавливается на 30 пикселей от верхнего края
            emoji = Emoji(emoji_x, emoji_y, random.choice(self.emojis))  # Создаём новый объект Emoji
            self.emoji_list.append(emoji)  # Добавляем эмодзи в список
            print(f"Создано эмодзи '{emoji.emoji}' на позиции ({emoji.x}, {emoji.y})")  # Отладочный вывод

        # Обновляем позиции всех эмодзи
        for emoji in self.emoji_list:
            emoji.update(gravity=0.5, width=width, height=height)  # Применяем гравитацию и обновляем позицию

        # Проверяем наличие обнаруженных рук
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Определяем, является ли рука левой или правой
                hand_label = handedness.classification[0].label  # Получаем метку 'Left' или 'Right'
                is_left = hand_label == 'Left'  # Флаг для левой руки

                # Выбираем цвет для рисования руки
                if is_left:
                    hand_color = (0, 255, 0)  # Зеленый цвет для левой руки
                else:
                    hand_color = (255, 0, 0)  # Синий цвет для правой руки

                # Рисуем ключевые точки и соединения руки на изображении
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=hand_color, thickness=2, circle_radius=2),  # Спецификации для ключевых точек
                    mp_drawing.DrawingSpec(color=hand_color, thickness=2)  # Спецификации для соединений
                )

                landmarks = hand_landmarks.landmark  # Получаем список ключевых точек руки

                # Проверяем каждое эмодзи на столкновение с текущей рукой
                for emoji in self.emoji_list:
                    if emoji.burst:
                        continue  # Пропускаем эмодзи, находящиеся в состоянии взрыва

                    collision_point = emoji.is_colliding_with_hand(landmarks, width, height)  # Проверяем столкновение
                    if collision_point:
                        ex, ey = emoji.x, emoji.y  # Текущая позиция эмодзи
                        hx, hy = collision_point  # Координаты точки столкновения
                        dx = ex - hx  # Разница по оси X
                        dy = ey - hy  # Разница по оси Y
                        distance = (dx ** 2 + dy ** 2) ** 0.5  # Расстояние до точки столкновения
                        if distance == 0:
                            distance = 1  # Избегаем деления на ноль
                        # Нормализованный вектор от точки столкновения к эмодзи
                        nx = dx / distance
                        ny = dy / distance

                        if is_left:
                            # Обработка взаимодействия с левой рукой (притягивание к указательному пальцу)
                            fingertip_x = int(landmarks[8].x * width)  # Координата X кончика указательного пальца
                            fingertip_y = int(landmarks[8].y * height)  # Координата Y кончика указательного пальца
                            
                            # Вычисляем разницу между позицией эмодзи и пальца
                            dx = emoji.x - fingertip_x
                            dy = emoji.y - fingertip_y
                            distance = (dx ** 2 + dy ** 2) ** 0.5  # Новое расстояние

                            if distance > 5:  # Если эмодзи не совсем на пальце, притягиваем его
                                force = 0.5  # Сила притяжения
                                emoji.vx -= (dx / distance) * force  # Изменяем скорость по оси X
                                emoji.vy -= (dy / distance) * force  # Изменяем скорость по оси Y
                            else:
                                # Удерживаем эмодзи на кончике пальца
                                emoji.vx = 0
                                emoji.vy = 0
                                emoji.x = fingertip_x
                                emoji.y = fingertip_y

                            # Реализация броска эмодзи, учитывая движение пальца
                            if hasattr(emoji, "prev_position"):
                                prev_fingertip_x, prev_fingertip_y = emoji.prev_position  # Предыдущая позиция пальца
                                delta_x = fingertip_x - prev_fingertip_x  # Изменение позиции по оси X
                                delta_y = fingertip_y - prev_fingertip_y  # Изменение позиции по оси Y
                                
                                # Если движение пальца резкое, отпускаем эмодзи
                                if abs(delta_x) > 10 or abs(delta_y) > 10:  # Порог для броска
                                    emoji.vx = delta_x * 0.3  # Преобразуем движение пальца в скорость эмодзи по оси X
                                    emoji.vy = delta_y * 0.3  # Преобразуем движение пальца в скорость эмодзи по оси Y
                                    print(f"Бросок эмодзи '{emoji.emoji}'!")  # Отладочный вывод

                            # Сохраняем текущую позицию пальца для использования в следующей итерации
                            emoji.prev_position = (fingertip_x, fingertip_y)

                        else:
                            # Обработка взаимодействия с правой рукой (отталкивание эмодзи)
                            force = 5.0  # Сила отталкивания
                            emoji.vx += nx * force  # Изменяем скорость по оси X
                            emoji.vy += ny * force  # Изменяем скорость по оси Y
                            print(f"Эмодзи '{emoji.emoji}' отталкивается от правой руки на позиции ({hx}, {hy})")  # Отладочный вывод

        # Удаление эмодзи, которые вышли за пределы экрана или имеют очень низкую скорость
        self.emoji_list = [
            emoji for emoji in self.emoji_list
            if not (emoji.burst and emoji.burst_time <= 0)  # Удаляем эмодзи после завершения взрыва
            and 0 <= emoji.y - emoji.radius <= height + 100  # Проверяем, находится ли эмодзи в допустимых пределах по Y
            and 0 <= emoji.x - emoji.radius <= width + 100    # Проверяем, находится ли эмодзи в допустимых пределах по X
            and not (abs(emoji.vx) < 0.1 and abs(emoji.vy) < 0.1 and emoji.y >= height - emoji.radius)  # Удаляем эмодзи с очень низкой скоростью, находящиеся у нижнего края
        ]

        # Преобразование изображения для отображения в PyQt
        q_image = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)  # Создаём QImage из массива данных
        pixmap = QPixmap.fromImage(q_image)  # Преобразуем QImage в QPixmap для отображения

        # Отрисовка эмодзи поверх видео
        painter = QPainter(pixmap)  # Создаём QPainter для рисования на QPixmap
        painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))  # Устанавливаем красный цвет пера и толщину линии
        painter.setFont(QFont("Arial", 32))  # Устанавливаем шрифт для эмодзи
        for emoji in self.emoji_list:
            emoji.draw(painter)  # Рисуем каждое эмодзи
        painter.end()  # Завершаем рисование

        self.video_label.setPixmap(pixmap)  # Отображаем итоговое изображение в QLabel

    def closeEvent(self, event):
        """
        Метод, вызываемый при закрытии окна.
        Освобождает ресурсы камеры.
        """
        self.cap.release()  # Освобождаем захват камеры
        super().closeEvent(event)  # Вызываем родительский метод

if __name__ == "__main__":
    app = QApplication([])  # Создаём приложение PyQt
    window = EmojiGameApp()  # Создаём главное окно приложения
    window.showFullScreen()  # Отображаем окно в полноэкранном режиме
    app.exec_()  # Запускаем цикл обработки событий приложения
