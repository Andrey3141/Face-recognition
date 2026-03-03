# FACE RECOGNITION SERVER - Архитектура проекта
## Rasp v1.2.0 (2026)

Android-приложение для распознавания лиц в реальном времени с использованием MediaPipe и FaceNet.  
**Три режима работы:** распознавание, дообучение, HTTP-сервер.  
**Визуальная обратная связь:** цветные рамки вокруг лиц с именем и процентом уверенности.  
**Встроенный API:** возможность отправлять изображения по HTTP и получать результаты распознавания.

## КОРНЕВАЯ ДИРЕКТОРИЯ ПРОЕКТА
```
FaceRecognition/
├── app/
│   └── src/
│       └── main/
│           ├── AndroidManifest.xml        # КОНФИГУРАЦИЯ: Разрешения, activity, тема
│           ├── java/com/printer/rasp/
│           │   ├── MainActivity.kt        # ТОЧКА ВХОДА: Главная активность с камерой и UI
│           │   ├── CHANGELOG.md           # ИСТОРИЯ: История изменений проекта
│           │   ├── PROJECT_STRUCTURE.md   # ДОКУМЕНТАЦИЯ: Описание архитектуры
│           │   │
│           │   └── rasp/                  # ОСНОВНЫЕ КОМПОНЕНТЫ
│           │       ├── FaceRecognitionEngine.kt  # ЯДРО: MediaPipe + FaceNet + база
│           │       ├── FaceNetModel.kt           # МОДЕЛЬ: Обертка TensorFlow Lite
│           │       ├── FaceRecognitionServer.kt  # СЕРВЕР: Ktor HTTP API
│           │       └── FaceOverlayView.kt        # ВИД: Отрисовка рамок лиц
│           │
│           └── res/                        # РЕСУРСЫ
│               ├── layout/
│               │   └── activity_main.xml   # РАЗМЕТКА: Главный экран
│               ├── drawable/
│               │   ├── circle_button.xml    # Круглая кнопка меню
│               │   ├── spinner_background.xml # Фон выпадающего списка
│               │   ├── edittext_border.xml  # Рамка поля ввода
│               │   ├── ic_launcher_foreground.xml
│               │   └── ic_launcher_background.xml
│               ├── values/
│               │   ├── colors.xml           # Цветовая схема
│               │   ├── strings.xml          # Текстовые ресурсы
│               │   └── themes.xml           # Тема приложения
│               └── xml/
│                   ├── file_paths.xml       # Пути для file provider
│                   ├── backup_rules.xml     # Правила резервного копирования
│                   └── data_extraction_rules.xml
│
├── gradle/wrapper/
│   └── gradle-wrapper.properties
├── build.gradle.kts          # Проектный Gradle
├── app/build.gradle.kts      # Модульный Gradle
├── settings.gradle.kts       # Настройки Gradle
├── gradle.properties         # Свойства Gradle
└── local.properties          # Локальные пути SDK
```

## АКТИВЫ (app/src/main/assets/)
```
assets/
├── facenet.tflite                          # МОДЕЛЬ: FaceNet для извлечения признаков
└── face_landmarker_v2_with_blendshapes.task # МОДЕЛЬ: MediaPipe для детекции лиц
```

## ДАННЫЕ (Internal Storage)
```
/data/user/0/com.printer.rasp/
├── files/
│   └── mobile_model.json    # БАЗА: Пользователи и эмбеддинги
└── cache/                   # ВРЕМЕННЫЕ: Кэш изображений

```

### 🔢 FaceNetModel.kt - ОБЕРТКА TENSORFLOW LITE
**Инкапсулирует работу с TensorFlow Lite моделью FaceNet.**

**ОТВЕЧАЕТ ЗА:**
- Загрузку .tflite модели из assets
- Предобработку изображения (resize до 160x160)
- Нормализацию пикселей в [0, 1]
- Запуск модели и получение эмбеддинга
- Определение реального размера эмбеддинга

**КЛЮЧЕВЫЕ МЕТОДЫ:**
- `getEmbedding(bitmap)` - извлечение признаков
- `getEmbeddingSize()` - размер эмбеддинга
- `isReady()` - проверка готовности

**ПАРАМЕТРЫ:**
```kotlin
companion object {
    private const val INPUT_IMAGE_SIZE = 160
    // Размер эмбеддинга определяется автоматически из модели
}
```

### 🌐 FaceRecognitionServer.kt - HTTP СЕРВЕР
**Встроенный Ktor-сервер для удаленного доступа.**

**ОТВЕЧАЕТ ЗА:**
- Запуск Netty-сервера на порту 5000
- Настройку CORS для веб-клиентов
- Обработку HTTP запросов
- Распознавание изображений в формате base64

**ЭНДПОИНТЫ:**
```
GET  /         - Информация о сервере
GET  /ping     - Проверка доступности
POST /recognize - Распознавание лица
GET  /users    - Список пользователей
```

**ФОРМАТЫ ДАННЫХ:**
```kotlin
data class RecognizeRequest(
    val image: String  // base64 изображение
)

data class ApiResponse(
    val success: Boolean,
    val message: String?,
    val faces: List<FaceDetectionResult>?,
    val count: Int?,
    val users: List<String>?,
    val total_encodings: Int?,
    val error: String?
)
```

### 🎨 FaceOverlayView.kt - ОТРИСОВКА РАМОК
**Кастомная View для отображения рамок вокруг распознанных лиц.**

**ОТВЕЧАЕТ ЗА:**
- Масштабирование координат (640x480 → размер экрана)
- Отрисовку прямоугольников (зеленый/красный)
- Отображение имени и процента уверенности
- Фоновую подложку под текст

## АЛГОРИТМЫ

### 🔍 РАСПОЗНАВАНИЕ ЛИЦА
```
1. Получить кадр с камеры (640x480)
2. Детекция лица через MediaPipe
   ↓
3. Получить bounding box лица
4. Расширить область в 2.5 раза
5. Вырезать область лица
   ↓
6. Изменить размер до 160x160
7. Нормализовать пиксели в [0, 1]
8. Запустить FaceNet → эмбеддинг (128/512 чисел)
   ↓
9. Для каждого пользователя в базе:
   - Вычислить косинусное сходство
   - Сохранить лучшее и второе
   ↓
10. Если сходство > 0.65 И разница со вторым > 0.1:
    → Узнан (имя + %)
    Иначе:
    → Неизвестный
```

### 📸 ДООБУЧЕНИЕ
```
1. Ввести имя пользователя
2. Начать захват кадров (каждые 300 мс)
   ↓
3. Для каждого кадра:
   - Получить эмбеддинг
   - Добавить в trainingSamples
   - Обновить прогресс (X/12)
   ↓
4. Когда набрано 12 образцов:
   - Усреднить все эмбеддинги
   - L2-нормализовать
   - Добавить в базу
   - Сохранить в JSON
```