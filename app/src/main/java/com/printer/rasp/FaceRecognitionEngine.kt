package com.printer.rasp

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.CascadeClassifier
import java.io.File
import java.io.FileOutputStream
import java.util.*
import kotlin.collections.ArrayList
import kotlin.math.*

data class MobileModel(
    val users: List<String>,
    val embeddings: List<List<Double>>
)

data class FaceRect(
    val x: Int,
    val y: Int,
    val width: Int,
    val height: Int
)

data class FaceWithEncoding(
    val rect: FaceRect,
    val encoding: FloatArray,
    val name: String = "Неизвестный",
    val confidence: Float = 0f
)

class FaceRecognitionEngine(private val context: Context) {

    private val faceNetModel: FaceNetModel = FaceNetModel(context)
    private var faceDetector: CascadeClassifier? = null
    private var isOpenCVReady = false
    private var knownEmbeddings = mutableListOf<FloatArray>()
    private var knownNames = mutableListOf<String>()
    private val embeddingSize: Int = faceNetModel.getEmbeddingSize()

    // Для плавного отслеживания нескольких лиц
    private val faceTrackers = mutableMapOf<Int, FaceTracker>()
    private var nextFaceId = 0
    private val smoothFactor = 0.6f // ✅ Добавлено!

    companion object {
        private const val HAAR_CASCADE_FILE = "haarcascade_frontalface_default.xml"
        private const val DB_FILENAME = "mobile_model.json"
        private const val CONFIDENCE_THRESHOLD = 0.65f
        private const val MIN_FACE_SIZE = 100
        private const val FACE_EXPAND_FACTOR = 1.8f
        private const val MAX_FACES = 5
    }

    data class FaceTracker(
        val id: Int,
        var rect: FaceRect,
        var lastSeen: Long,
        var smoothRect: FaceRect
    )

    init {
        try {
            if (!OpenCVLoader.initDebug()) {
                println("❌ OpenCV не удалось загрузить через initDebug")
            } else {
                println("✅ OpenCV загружен успешно")
            }
        } catch (e: Exception) {
            println("❌ Ошибка инициализации OpenCV: ${e.message}")
        }

        println("📊 Размер эмбеддинга FaceNet: $embeddingSize")

        loadHaarCascade()
        loadModel()
    }

    private fun loadHaarCascade() {
        try {
            println("📁 Загрузка каскадов Хаара...")

            val cascadeFile = File(context.filesDir, HAAR_CASCADE_FILE)
            if (!cascadeFile.exists()) {
                context.assets.open(HAAR_CASCADE_FILE).use { input ->
                    FileOutputStream(cascadeFile).use { output ->
                        input.copyTo(output)
                    }
                }
                println("✅ Каскад Хаара скопирован, размер: ${cascadeFile.length()} байт")
            }

            faceDetector = CascadeClassifier(cascadeFile.absolutePath)

            if (faceDetector?.empty() == true) {
                faceDetector = CascadeClassifier()
                faceDetector?.load(cascadeFile.absolutePath)
            }

            isOpenCVReady = faceDetector?.empty() == false
            if (isOpenCVReady) {
                println("✅ Каскады Хаара загружены успешно")
            } else {
                println("❌ Ошибка загрузки каскадов")
            }

        } catch (e: Exception) {
            println("❌ Ошибка загрузки каскадов: ${e.message}")
            e.printStackTrace()
            isOpenCVReady = false
        }
    }

    fun loadModel(): Boolean {
        return try {
            val file = File(context.filesDir, DB_FILENAME)
            if (file.exists()) {
                val json = file.readText()
                val modelType = object : TypeToken<MobileModel>() {}.type
                val model = Gson().fromJson<MobileModel>(json, modelType)

                knownNames.clear()
                knownEmbeddings.clear()

                model.users.zip(model.embeddings).forEach { (user, emb) ->
                    val floatArray = emb.map { it.toFloat() }.toFloatArray()
                    if (floatArray.size != embeddingSize) {
                        val newArray = FloatArray(embeddingSize)
                        val copySize = min(floatArray.size, embeddingSize)
                        System.arraycopy(floatArray, 0, newArray, 0, copySize)
                        knownEmbeddings.add(newArray)
                    } else {
                        knownEmbeddings.add(floatArray)
                    }
                    knownNames.add(user)
                }
                println("✅ База загружена: ${knownNames.size} пользователей")
                true
            } else {
                println("📝 База данных не найдена")
                false
            }
        } catch (e: Exception) {
            e.printStackTrace()
            false
        }
    }

    suspend fun getFaceEncodings(bitmap: Bitmap): List<FaceWithEncoding> = withContext(Dispatchers.Default) {
        val results = mutableListOf<FaceWithEncoding>()

        try {
            if (!isOpenCVReady || faceDetector == null) {
                return@withContext results
            }

            val startTime = System.currentTimeMillis()

            val rgba = Mat()
            Utils.bitmapToMat(bitmap, rgba)

            val gray = Mat()
            Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY)
            Imgproc.equalizeHist(gray, gray)

            val faces = MatOfRect()
            faceDetector!!.detectMultiScale(
                gray,
                faces,
                1.1,
                3,
                0,
                Size(MIN_FACE_SIZE.toDouble(), MIN_FACE_SIZE.toDouble()),
                Size()
            )

            val facesArray = faces.toArray()

            if (facesArray.isNotEmpty()) {
                val sortedFaces = facesArray.sortedByDescending { it.width * it.height }
                val facesToProcess = sortedFaces.take(MAX_FACES)

                val currentTime = System.currentTimeMillis()
                val newTrackers = mutableMapOf<Int, FaceTracker>()

                for (face in facesToProcess) {
                    val centerX = face.x + face.width / 2
                    val centerY = face.y + face.height / 2
                    val size = max(face.width, face.height) * FACE_EXPAND_FACTOR

                    var x = (centerX - size / 2).toInt()
                    var y = (centerY - size / 2).toInt()
                    var w = size.toInt()
                    var h = size.toInt()

                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, bitmap.width - x)
                    h = min(h, bitmap.height - y)

                    val newRect = FaceRect(x, y, w, h)

                    val matchedTracker = findMatchingTracker(newRect)

                    val tracker = if (matchedTracker != null) {
                        matchedTracker.rect = newRect
                        matchedTracker.lastSeen = currentTime
                        matchedTracker.smoothRect = smoothFaceRect(newRect, matchedTracker.smoothRect)
                        matchedTracker
                    } else {
                        FaceTracker(
                            id = nextFaceId++,
                            rect = newRect,
                            lastSeen = currentTime,
                            smoothRect = newRect
                        )
                    }

                    newTrackers[tracker.id] = tracker

                    val faceBitmap = try {
                        Bitmap.createBitmap(bitmap, tracker.smoothRect.x, tracker.smoothRect.y,
                            tracker.smoothRect.width, tracker.smoothRect.height)
                    } catch (e: Exception) {
                        Bitmap.createBitmap(bitmap, newRect.x, newRect.y, newRect.width, newRect.height)
                    }

                    val encoding = faceNetModel.getEmbedding(faceBitmap)
                    faceBitmap.recycle()

                    if (encoding != null) {
                        val (name, confidence) = recognizeFace(encoding)
                        results.add(
                            FaceWithEncoding(
                                rect = tracker.smoothRect,
                                encoding = encoding,
                                name = name,
                                confidence = confidence
                            )
                        )
                    }
                }

                faceTrackers.clear()
                faceTrackers.putAll(newTrackers)
            }

            rgba.release()
            gray.release()

            val processingTime = System.currentTimeMillis() - startTime
            // println("⏱️ Время обработки: $processingTime ms")

        } catch (e: Exception) {
            println("❌ Ошибка в getFaceEncodings: ${e.message}")
            e.printStackTrace()
        }

        return@withContext results
    }

    private fun findMatchingTracker(newRect: FaceRect): FaceTracker? {
        val newCenterX = newRect.x + newRect.width / 2
        val newCenterY = newRect.y + newRect.height / 2

        return faceTrackers.values.minByOrNull { tracker ->
            val trackerCenterX = tracker.rect.x + tracker.rect.width / 2
            val trackerCenterY = tracker.rect.y + tracker.rect.height / 2
            val distance = sqrt(((newCenterX - trackerCenterX).toDouble().pow(2) +
                    (newCenterY - trackerCenterY).toDouble().pow(2)))
            distance
        }.takeIf { tracker ->
            if (tracker == null) return@takeIf false
            val trackerCenterX = tracker.rect.x + tracker.rect.width / 2
            val trackerCenterY = tracker.rect.y + tracker.rect.height / 2
            val distance = sqrt(((newCenterX - trackerCenterX).toDouble().pow(2) +
                    (newCenterY - trackerCenterY).toDouble().pow(2)))
            distance < max(tracker.rect.width, tracker.rect.height) * 0.5
        }
    }

    // ✅ ИСПРАВЛЕННЫЙ МЕТОД
    private fun smoothFaceRect(newRect: FaceRect, lastRect: FaceRect): FaceRect {
        val smoothX = (lastRect.x * smoothFactor + newRect.x * (1 - smoothFactor)).toInt()
        val smoothY = (lastRect.y * smoothFactor + newRect.y * (1 - smoothFactor)).toInt()
        val smoothW = (lastRect.width * smoothFactor + newRect.width * (1 - smoothFactor)).toInt()
        val smoothH = (lastRect.height * smoothFactor + newRect.height * (1 - smoothFactor)).toInt()

        return FaceRect(smoothX, smoothY, smoothW, smoothH)
    }

    suspend fun getFaceEncoding(bitmap: Bitmap): Pair<FloatArray?, FaceRect?> = withContext(Dispatchers.Default) {
        val faces = getFaceEncodings(bitmap)
        if (faces.isNotEmpty()) {
            val face = faces[0]
            face.encoding to face.rect
        } else {
            null to null
        }
    }

    fun recognizeFace(embedding: FloatArray): Pair<String, Float> {
        if (knownEmbeddings.isEmpty()) return "Неизвестный" to 0f

        var bestMatch = "Неизвестный"
        var bestSimilarity = 0f
        var secondBestSimilarity = 0f

        for (i in knownEmbeddings.indices) {
            val similarity = cosineSimilarity(embedding, knownEmbeddings[i])

            if (similarity > bestSimilarity) {
                secondBestSimilarity = bestSimilarity
                bestSimilarity = similarity
                bestMatch = knownNames[i]
            } else if (similarity > secondBestSimilarity) {
                secondBestSimilarity = similarity
            }
        }

        return if (bestSimilarity > CONFIDENCE_THRESHOLD &&
            (bestSimilarity - secondBestSimilarity) > 0.1f) {
            bestMatch to bestSimilarity
        } else {
            "Неизвестный" to bestSimilarity
        }
    }

    fun addFace(name: String, encoding: FloatArray): Boolean {
        if (encoding.size != embeddingSize) {
            if (encoding.isNotEmpty()) {
                val newEncoding = FloatArray(embeddingSize)
                val copySize = min(encoding.size, embeddingSize)
                System.arraycopy(encoding, 0, newEncoding, 0, copySize)
                knownEmbeddings.add(newEncoding)
                knownNames.add(name)
                return saveDatabase()
            }
            return false
        }

        knownEmbeddings.add(encoding)
        knownNames.add(name)
        return saveDatabase()
    }

    fun saveDatabase(): Boolean {
        return try {
            val embeddings = knownEmbeddings.map { it.map { f -> f.toDouble() } }
            val model = MobileModel(knownNames, embeddings)
            val json = Gson().toJson(model)
            File(context.filesDir, DB_FILENAME).writeText(json)
            println("💾 База сохранена: ${knownNames.size} пользователей")
            true
        } catch (e: Exception) {
            e.printStackTrace()
            false
        }
    }

    fun getUsers(): List<String> = knownNames.distinct()
    fun isModelLoaded(): Boolean = isOpenCVReady && faceNetModel.isReady()
    fun resetTracking() {
        faceTrackers.clear()
        nextFaceId = 0
    }

    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        var dotProduct = 0f
        var normA = 0f
        var normB = 0f

        for (i in a.indices) {
            dotProduct += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }

        return if (normA > 0 && normB > 0) {
            dotProduct / (sqrt(normA) * sqrt(normB))
        } else 0f
    }

    fun checkHaarCascade(): Boolean {
        return try {
            val cascadeFile = File(context.filesDir, HAAR_CASCADE_FILE)
            if (!cascadeFile.exists()) {
                println("❌ Файл каскада не найден: ${cascadeFile.absolutePath}")
                return false
            }
            println("✅ Файл каскада найден, размер: ${cascadeFile.length()} байт")
            true
        } catch (e: Exception) {
            println("❌ Ошибка проверки каскада: ${e.message}")
            false
        }
    }

    fun close() {
        faceNetModel.close()
    }
}