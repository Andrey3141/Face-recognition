package com.printer.rasp

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import kotlin.math.*

data class FaceDetectionResult(
    val x: Float,
    val y: Float,
    val width: Float,
    val height: Float,
    val name: String,
    val confidence: Float,
    val recognized: Boolean
)

class FaceOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f // Уменьшил для нескольких лиц
        color = Color.GREEN
    }

    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 28f // Уменьшил для компактности
        typeface = Typeface.DEFAULT_BOLD
        isAntiAlias = true
    }

    private val backgroundPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.GREEN
        isAntiAlias = true
    }

    var detectedFaces: List<FaceDetectionResult> = emptyList()
        set(value) {
            field = value
            postInvalidate()
        }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        if (detectedFaces.isEmpty()) {
            return
        }

        val viewWidth = width.toFloat()
        val viewHeight = height.toFloat()
        val imageWidth = 480f
        val imageHeight = 640f

        val scale = minOf(viewWidth / imageWidth, viewHeight / imageHeight)
        val offsetX = (viewWidth - imageWidth * scale) / 2
        val offsetY = (viewHeight - imageHeight * scale) / 2

        // Рисуем все лица
        for (face in detectedFaces) {
            val left = face.x * scale + offsetX
            val top = face.y * scale + offsetY
            val right = (face.x + face.width) * scale + offsetX
            val bottom = (face.y + face.height) * scale + offsetY

            val boxColor = if (face.recognized) Color.GREEN else Color.RED

            // Рисуем прямоугольник
            paint.color = boxColor
            canvas.drawRect(left, top, right, bottom, paint)

            // Текст с именем
            val text = if (face.recognized) {
                "${face.name} (${(face.confidence * 100).toInt()}%)"
            } else {
                "?"
            }

            backgroundPaint.color = boxColor
            val textWidth = textPaint.measureText(text)
            val textHeight = textPaint.textSize

            // Фон для текста (сверху)
            canvas.drawRect(
                left - 5,
                top - textHeight - 15,
                left + textWidth + 10,
                top - 5,
                backgroundPaint
            )

            textPaint.color = Color.WHITE
            canvas.drawText(text, left, top - 10, textPaint)

            // Если лиц не узнано, рисуем вопросительный знак побольше
            if (!face.recognized) {
                textPaint.textSize = 42f
                canvas.drawText("?", left + 10, top + 50, textPaint)
                textPaint.textSize = 28f // Возвращаем обратно
            }
        }

        // Рисуем счетчик лиц в углу
        textPaint.textSize = 24f
        textPaint.color = Color.WHITE
        backgroundPaint.color = Color.BLUE
        val countText = "👥 ${detectedFaces.size}"
        val countWidth = textPaint.measureText(countText)

        canvas.drawRect(
            20f,
            20f,
            20f + countWidth + 20f,
            20f + textPaint.textSize + 20f,
            backgroundPaint
        )

        textPaint.color = Color.WHITE
        canvas.drawText(countText, 30f, 20f + textPaint.textSize, textPaint)
        textPaint.textSize = 28f
    }
}