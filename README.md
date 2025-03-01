
# Emoji Game App

**Emoji Game App** - это интерактивное приложение, которое использует веб-камеру для обнаружения рук и эмоций пользователя, чтобы взаимодействовать с анимированными эмодзи.

## Описание

Приложение предоставляет следующий функционал:
1. Распознавание рук и их движений с помощью библиотеки Mediapipe.
2. Обнаружение и анализ эмоций лица с помощью DeepFace.
3. Взаимодействие эмодзи с руками:
   - Левая рука позволяет притягивать эмодзи к указательному пальцу.
   - Правая рука отталкивает эмодзи.
4. Игрушка для танцев, пока на один кадр и полным отслеживание скелета.
Эмодзи имеют гравитацию и могут "взрываться", если выходят за пределы экрана.

## Основные технологии

- Python
- PyQt5
- OpenCV
- Mediapipe
- DeepFace
- NumPy

## Установка

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/jestelf/Emoji-Game-App.git
   cd Emoji-Game-App
   ```

2. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

3. Запустите приложение:
   ```bash
   python emobot.py
   ```

## Версии

### Стандартный emobot.py
Включает в себя всё тоже самое что и Version with config, но из-за прямого включения в код, не глючит. (Рекомендую к использованию)

### Version with config
Версия приложения с использованием внешнего файла конфигурации (`config.json`) для настройки параметров, таких как гравитация эмодзи, вероятность их появления, параметры окна и захвата камеры.  
**Особенности:**
- Легко изменяемые параметры без изменения исходного кода.
- Подходит для тонкой настройки поведения приложения.

### DanceGame
Игрушка другого характера, с кадром для танцев.

**Известные проблемы:**
- Возможны сбои при распознавании эмоций из-за ошибок в работе библиотеки DeepFace.
- Явно глючит из-за подгрузки конфига.

### Version without DeepFace
Облегчённая версия приложения, в которой отсутствует распознавание эмоций. Подходит для использования в системах, где DeepFace не требуется или не может быть запущен.  
**Особенности:**
- Упрощённая логика без анализа эмоций.
- Повышенная производительность на слабых устройствах.

## Настройка через `config.json`

Пример содержимого файла `config.json`:
```json
{
    "static_image_mode": false,
    "max_num_hands": 2,
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
    "camera_index": 0,
    "video_frame_rate": 30,
    "window_title": "Emoji Game App",
    "window_geometry": [100, 100, 800, 600],
    "emojis": ["😀", "✨", "🎉", "❤️", "🔥"],
    "emoji_gravity": 0.5,
    "emoji_spawn_probability": 0.05
}
```

## Параметры `config.json`

- **`static_image_mode`**: Использовать ли статические изображения (`true`) или видеопоток (`false`).
- **`max_num_hands`**: Максимальное количество рук для отслеживания.
- **`min_detection_confidence`**: Минимальная уверенность для обнаружения руки (0-1).
- **`min_tracking_confidence`**: Минимальная уверенность для отслеживания руки (0-1).
- **`camera_index`**: Индекс камеры (например, 0 для первой камеры).
- **`video_frame_rate`**: Частота обновления кадров (FPS).
- **`window_title`**: Название окна приложения.
- **`window_geometry`**: Геометрия окна в формате `[x, y, width, height]`.
- **`emojis`**: Список эмодзи, используемых в приложении.
- **`emoji_gravity`**: Гравитация, влияющая на скорость падения эмодзи.
- **`emoji_spawn_probability`**: Вероятность появления нового эмодзи на каждом кадре (0-1).

## Лицензия

Этот проект распространяется под лицензией MIT. Подробнее см. файл LICENSE.
