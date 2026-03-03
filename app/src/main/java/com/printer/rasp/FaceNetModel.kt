package com.printer.rasp

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

class FaceNetModel(private val context: Context) {

    companion object {
        private const val MODEL_FILE = "facenet.tflite"
        private const val INPUT_IMAGE_SIZE = 160
        private const val EMBEDDING_SIZE = 512  // ВНИМАНИЕ: в твоей модели может быть 128 или 512!
    }

    private var interpreter: Interpreter? = null
    private var isModelReady = false
    private var actualEmbeddingSize = EMBEDDING_SIZE

    init {
        try {
            println("📁 Загрузка FaceNet модели из assets...")
            interpreter = Interpreter(FileUtil.loadMappedFile(context, MODEL_FILE))

            // Определяем реальный размер эмбеддинга из модели
            val outputShape = interpreter?.getOutputTensor(0)?.shape()
            if (outputShape != null && outputShape.size >= 2) {
                actualEmbeddingSize = outputShape[1]
                println("✅ FaceNet модель загружена! Размер эмбеддинга: $actualEmbeddingSize")
            } else {
                println("✅ FaceNet модель загружена! Размер эмбеддинга: $EMBEDDING_SIZE (предположительно)")
            }

            isModelReady = true

        } catch (e: Exception) {
            println("❌ Ошибка загрузки FaceNet: ${e.message}")
            e.printStackTrace()
            isModelReady = false
        }
    }

    fun getEmbedding(bitmap: Bitmap): FloatArray? {
        if (!isModelReady || interpreter == null) {
            println("❌ FaceNet модель не готова")
            return null
        }

        try {
            // 1. Масштабируем до нужного размера
            val resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, true)

            // 2. Создаем ByteBuffer для входа
            val byteBuffer = ByteBuffer.allocateDirect(4 * INPUT_IMAGE_SIZE * INPUT_IMAGE_SIZE * 3)
            byteBuffer.order(ByteOrder.nativeOrder())

            val intValues = IntArray(INPUT_IMAGE_SIZE * INPUT_IMAGE_SIZE)
            resizedBitmap.getPixels(intValues, 0, INPUT_IMAGE_SIZE, 0, 0, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)

            var pixel = 0
            for (i in 0 until INPUT_IMAGE_SIZE) {
                for (j in 0 until INPUT_IMAGE_SIZE) {
                    val currentPixel = intValues[pixel++]
                    // Нормализация в [0, 1]
                    byteBuffer.putFloat(((currentPixel shr 16) and 0xFF) / 255.0f)
                    byteBuffer.putFloat(((currentPixel shr 8) and 0xFF) / 255.0f)
                    byteBuffer.putFloat((currentPixel and 0xFF) / 255.0f)
                }
            }

            // 3. Получаем размер выходного тензора
            val outputShape = interpreter?.getOutputTensor(0)?.shape()
            val embeddingSize = if (outputShape != null && outputShape.size >= 2) outputShape[1] else actualEmbeddingSize

            // 4. Создаем выходной массив
            val embeddings = Array(1) { FloatArray(embeddingSize) }

            // 5. Запускаем модель
            interpreter?.run(byteBuffer, embeddings)

            // Проверяем, что эмбеддинг не пустой
            if (embeddings[0].all { it == 0f }) {
                println("⚠️ Предупреждение: получен нулевой эмбеддинг")
            }

            return embeddings[0]

        } catch (e: Exception) {
            println("❌ Ошибка получения эмбеддинга: ${e.message}")
            e.printStackTrace()
            return null
        }
    }

    fun getEmbeddingSize(): Int {
        return try {
            val outputShape = interpreter?.getOutputTensor(0)?.shape()
            if (outputShape != null && outputShape.size >= 2) outputShape[1] else actualEmbeddingSize
        } catch (e: Exception) {
            actualEmbeddingSize
        }
    }

    fun isReady(): Boolean = isModelReady

    fun close() {
        interpreter?.close()
    }
}