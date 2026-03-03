package com.printer.rasp

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Base64
import com.google.gson.Gson
import io.ktor.http.*
import io.ktor.server.application.*
import io.ktor.server.engine.*
import io.ktor.server.netty.*
import io.ktor.server.plugins.cors.routing.*
import io.ktor.server.plugins.contentnegotiation.*
import io.ktor.server.request.*
import io.ktor.server.response.*
import io.ktor.server.routing.*
import io.ktor.serialization.gson.*
import kotlinx.coroutines.*
import java.net.NetworkInterface
import java.net.InetAddress
import java.util.*

data class TrainRequest(
    val name: String,
    val image: String
)

data class RecognizeRequest(
    val image: String
)

data class ApiResponse(
    val success: Boolean,
    val message: String? = null,
    val faces: List<FaceDetectionResult>? = null,
    val count: Int? = null,
    val users: List<String>? = null,
    val total_encodings: Int? = null,
    val error: String? = null
)

class FaceRecognitionServer(private val context: Context, private val engine: FaceRecognitionEngine) {

    private var server: NettyApplicationEngine? = null
    private val gson = Gson()

    fun start(port: Int = 5000) {
        server = embeddedServer(Netty, port = port) {
            install(ContentNegotiation) {
                gson {
                    setPrettyPrinting()
                    serializeNulls()
                }
            }

            install(CORS) {
                anyHost()
                allowHeader(HttpHeaders.ContentType)
                allowMethod(HttpMethod.Get)
                allowMethod(HttpMethod.Post)
                allowMethod(HttpMethod.Options)
            }

            routing {
                get("/") {
                    val clientIp = call.request.local.remoteHost
                    println("🌐 Подключение с $clientIp")

                    call.respond(
                        ApiResponse(
                            success = true,
                            message = "Face Recognition Server",
                            users = engine.getUsers(),
                            total_encodings = engine.getUsers().size
                        )
                    )
                }

                get("/ping") {
                    call.respond(ApiResponse(success = true, message = "pong"))
                }

                post("/recognize") {
                    val request = try {
                        call.receive<RecognizeRequest>()
                    } catch (e: Exception) {
                        call.respond(HttpStatusCode.BadRequest, ApiResponse(success = false, error = "Invalid request format"))
                        return@post
                    }

                    withContext(Dispatchers.IO) {
                        try {
                            val imageData = request.image.substringAfter(",")
                            val imageBytes = Base64.decode(imageData, Base64.NO_WRAP)

                            val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)

                            val (encoding, rect) = engine.getFaceEncoding(bitmap)

                            val faces = mutableListOf<FaceDetectionResult>()

                            if (encoding != null && rect != null) {
                                val (name, confidence) = engine.recognizeFace(encoding)

                                // ✅ ИСПРАВЛЕНО: используем expandRect из engine
                                val expandedRect = expandRect(rect, bitmap.width, bitmap.height)

                                faces.add(
                                    FaceDetectionResult(
                                        x = expandedRect.x.toFloat(),
                                        y = expandedRect.y.toFloat(),
                                        width = expandedRect.width.toFloat(),
                                        height = expandedRect.height.toFloat(),
                                        name = name,
                                        confidence = confidence,
                                        recognized = name != "Неизвестный" && confidence > 0.65f
                                    )
                                )

                                println("📸 Распознано: $name (${"%.2f".format(confidence)})")
                            } else {
                                println("📸 Лиц не найдено")
                            }

                            call.respond(
                                ApiResponse(
                                    success = true,
                                    faces = faces,
                                    count = faces.size,
                                    users = engine.getUsers(),
                                    total_encodings = engine.getUsers().size
                                )
                            )

                        } catch (e: Exception) {
                            e.printStackTrace()
                            call.respond(
                                HttpStatusCode.InternalServerError,
                                ApiResponse(
                                    success = false,
                                    error = e.message
                                )
                            )
                        }
                    }
                }

                get("/users") {
                    call.respond(
                        ApiResponse(
                            success = true,
                            users = engine.getUsers(),
                            total_encodings = engine.getUsers().size
                        )
                    )
                }
            }
        }

        GlobalScope.launch {
            server!!.start(wait = false)

            val localIp = getLocalIpAddress()
            println("\n" + "=".repeat(50))
            println("🤖 СЕРВЕР РАСПОЗНАВАНИЯ ЛИЦ ЗАПУЩЕН")
            println("=".repeat(50))
            println("\n📱 АДРЕС ДЛЯ ПОДКЛЮЧЕНИЯ:")
            println("   http://$localIp:$port")
            println("\n✅ Сервер готов к работе")
            println("=".repeat(50) + "\n")
        }
    }

    fun stop() {
        server?.stop(1000, 5000)
    }

    // ✅ ДОБАВЛЕН МЕТОД expandRect
    private fun expandRect(rect: FaceRect, maxWidth: Int, maxHeight: Int): FaceRect {
        val scaleFactor = 1.5f
        val newWidth = minOf((rect.width * scaleFactor).toInt(), maxWidth - rect.x)
        val newHeight = minOf((rect.height * scaleFactor).toInt(), maxHeight - rect.y)
        val newX = maxOf(0, rect.x - ((newWidth - rect.width) / 2))
        val newY = maxOf(0, rect.y - ((newHeight - rect.height) / 2))

        return FaceRect(
            x = newX,
            y = newY,
            width = minOf(newWidth, maxWidth - newX),
            height = minOf(newHeight, maxHeight - newY)
        )
    }

    private fun getLocalIpAddress(): String {
        return try {
            val interfaces = NetworkInterface.getNetworkInterfaces()
            while (interfaces.hasMoreElements()) {
                val intf = interfaces.nextElement()
                val addresses = intf.inetAddresses
                while (addresses.hasMoreElements()) {
                    val addr = addresses.nextElement()
                    if (!addr.isLoopbackAddress && addr is InetAddress) {
                        val hostAddress = addr.hostAddress ?: continue
                        if (hostAddress.contains(".") && !hostAddress.startsWith("127")) {
                            return hostAddress
                        }
                    }
                }
            }
            "localhost"
        } catch (e: Exception) {
            "localhost"
        }
    }
}