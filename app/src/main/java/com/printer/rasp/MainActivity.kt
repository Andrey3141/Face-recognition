package com.printer.rasp

import android.Manifest
import android.animation.AnimatorSet
import android.animation.ObjectAnimator
import android.app.AlertDialog
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.util.Log
import android.view.View
import android.view.ViewGroup
import android.view.animation.DecelerateInterpolator
import android.widget.*
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.nio.ByteBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.*

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var faceOverlay: FaceOverlayView
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var faceEngine: FaceRecognitionEngine

    // UI элементы
    private lateinit var tvStatus: TextView
    private lateinit var tvMode: TextView
    private lateinit var tvModelInfo: TextView
    private lateinit var tvSamples: TextView
    private lateinit var tvServerIp: TextView
    private lateinit var tvFps: TextView
    private lateinit var etName: EditText
    private lateinit var spinnerUsers: Spinner
    private lateinit var btnMenu: ImageButton
    private lateinit var controlPanel: LinearLayout
    private lateinit var btnStartServer: Button
    private lateinit var btnSwitchCamera: Button
    private lateinit var btnRetrain: Button
    private lateinit var btnRecognize: Button
    private lateinit var btnStop: Button
    private lateinit var progressBar: ProgressBar

    private var cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA
    private var currentMode = "idle"
    private var isFrontCamera = true
    private var isMenuOpen = false
    private var isAnalyzing = AtomicBoolean(false)

    private var faceServer: FaceRecognitionServer? = null
    private var isServerRunning = false

    // Для дообучения
    private var trainingSamples = mutableListOf<FloatArray>()
    private var currentTrainingUser = ""
    private var isTraining = false
    private var trainingCount = 0
    private val maxSamples = 12

    // Список пользователей
    private lateinit var usersAdapter: ArrayAdapter<String>
    private var usersList = mutableListOf<String>()

    // Для измерения FPS
    private var lastFpsUpdate = 0L
    private var frameCount = 0
    private var currentFps = 0

    // Пропуск кадров
    private var frameSkip = 0
    private val processEveryNFrames = 1

    // Для первого запуска
    private var isFirstStart = true

    private val requiredPermissions = arrayOf(
        Manifest.permission.CAMERA,
        Manifest.permission.WRITE_EXTERNAL_STORAGE,
        Manifest.permission.READ_EXTERNAL_STORAGE
    )

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val allGranted = permissions.values.all { it }
        if (allGranted) {
            startCamera()
        } else {
            Toast.makeText(this, "Необходимы все разрешения", Toast.LENGTH_LONG).show()
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        hideSystemUI()
        initViews()
        setupListeners()

        cameraExecutor = Executors.newSingleThreadExecutor()
        faceEngine = FaceRecognitionEngine(this)

        // Проверяем первый запуск
        checkFirstStart()

        loadModelAndUpdateUI()
        checkPermissions()

        // Запускаем обновление FPS
        lifecycleScope.launch {
            while (true) {
                delay(1000)
                updateFpsDisplay()
            }
        }
    }

    private fun initViews() {
        previewView = findViewById(R.id.previewView)
        faceOverlay = findViewById(R.id.faceOverlay)
        tvStatus = findViewById(R.id.tvStatus)
        tvMode = findViewById(R.id.tvMode)
        tvModelInfo = findViewById(R.id.tvModelInfo)
        tvSamples = findViewById(R.id.tvSamples)
        tvServerIp = findViewById(R.id.tvServerIp)
        tvFps = findViewById(R.id.tvFps)
        etName = findViewById(R.id.etName)
        spinnerUsers = findViewById(R.id.spinnerUsers)
        btnMenu = findViewById(R.id.btnMenu)
        controlPanel = findViewById(R.id.controlPanel)
        btnStartServer = findViewById(R.id.btnStartServer)
        btnSwitchCamera = findViewById(R.id.btnSwitchCamera)
        btnRetrain = findViewById(R.id.btnRetrain)
        btnRecognize = findViewById(R.id.btnRecognize)
        btnStop = findViewById(R.id.btnStop)
        progressBar = findViewById(R.id.progressBar)

        previewView.scaleType = PreviewView.ScaleType.FILL_CENTER
        btnStop.isEnabled = false

        usersAdapter = object : ArrayAdapter<String>(this, android.R.layout.simple_spinner_item, usersList) {
            override fun getView(position: Int, convertView: View?, parent: ViewGroup): View {
                val view = super.getView(position, convertView, parent)
                val textView = view as TextView
                textView.setTextColor(Color.WHITE)
                if (usersList[position] == "➕ Новый пользователь") {
                    textView.setTextColor(Color.parseColor("#4CAF50"))
                }
                return view
            }

            override fun getDropDownView(position: Int, convertView: View?, parent: ViewGroup): View {
                val view = super.getDropDownView(position, convertView, parent)
                val textView = view as TextView
                textView.setTextColor(Color.WHITE)
                view.setBackgroundColor(Color.parseColor("#333333"))
                if (usersList[position] == "➕ Новый пользователь") {
                    textView.setTextColor(Color.parseColor("#4CAF50"))
                }
                return view
            }
        }
        usersAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        spinnerUsers.adapter = usersAdapter
    }

    private fun setupListeners() {
        btnMenu.setOnClickListener {
            if (isMenuOpen) closeMenu() else openMenu()
        }

        btnStartServer.setOnClickListener {
            if (isServerRunning) stopServer() else startServer()
            closeMenu()
        }

        btnSwitchCamera.setOnClickListener {
            isFrontCamera = !isFrontCamera
            cameraSelector = if (isFrontCamera)
                CameraSelector.DEFAULT_FRONT_CAMERA
            else
                CameraSelector.DEFAULT_BACK_CAMERA
            startCamera()
            Toast.makeText(this,
                if (isFrontCamera) "Фронтальная камера" else "Задняя камера",
                Toast.LENGTH_SHORT).show()
            closeMenu()
        }

        btnRetrain.setOnClickListener {
            val selectedItem = spinnerUsers.selectedItem?.toString() ?: "➕ Новый пользователь"

            val selectedUser = if (selectedItem == "➕ Новый пользователь") {
                etName.text.toString().trim()
            } else {
                selectedItem
            }

            if (selectedUser.isEmpty()) {
                Toast.makeText(this, "Введите имя нового пользователя", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            val existingUsers = faceEngine.getUsers()
            if (existingUsers.contains(selectedUser) && selectedItem != selectedUser) {
                AlertDialog.Builder(this)
                    .setTitle("Пользователь уже существует")
                    .setMessage("Пользователь '$selectedUser' уже есть в базе. Хотите добавить новые образцы?")
                    .setPositiveButton("Да, добавить") { _, _ ->
                        startRetraining(selectedUser)
                    }
                    .setNegativeButton("Отмена") { _, _ -> }
                    .show()
            } else {
                startRetraining(selectedUser)
            }
            closeMenu()
        }

        btnRecognize.setOnClickListener {
            startRecognition()
            closeMenu()
        }

        btnStop.setOnClickListener {
            stopAll()
            closeMenu()
        }
    }

    private fun checkFirstStart() {
        val prefs = getSharedPreferences("app_prefs", MODE_PRIVATE)
        isFirstStart = prefs.getBoolean("is_first_start", true)

        if (isFirstStart) {
            prefs.edit().putBoolean("is_first_start", false).apply()

            runOnUiThread {
                Toast.makeText(this, "👋 Добро пожаловать! Добавьте первого пользователя через меню", Toast.LENGTH_LONG).show()
                tvModelInfo.text = "📁 База пуста. Нажмите меню и добавьте пользователя"
                tvModelInfo.setTextColor(Color.YELLOW)
            }
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            try {
                val cameraProvider = cameraProviderFuture.get()

                val preview = Preview.Builder()
                    .build()
                preview.setSurfaceProvider(previewView.surfaceProvider)

                val imageAnalysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .build()

                imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                    if (!isAnalyzing.get() && currentMode != "idle") {
                        isAnalyzing.set(true)
                        processFrame(imageProxy)
                    } else {
                        imageProxy.close()
                    }
                }

                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis)

            } catch (e: Exception) {
                Log.e("Camera", "Ошибка запуска камеры", e)
                Toast.makeText(this, "Ошибка запуска камеры", Toast.LENGTH_SHORT).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun processFrame(imageProxy: ImageProxy) {
        frameSkip++
        if (frameSkip % processEveryNFrames != 0) {
            imageProxy.close()
            isAnalyzing.set(false)
            return
        }

        updateFps()

        try {
            val bitmap = imageProxyToBitmap(imageProxy)

            val rotatedBitmap = rotateBitmap(bitmap, imageProxy.imageInfo.rotationDegrees, isFrontCamera)

            if (rotatedBitmap != null) {
                when (currentMode) {
                    "retraining" -> {
                        lifecycleScope.launch(Dispatchers.Default) {
                            processRetrainingFrame(rotatedBitmap)
                            rotatedBitmap.recycle()
                            if (bitmap != rotatedBitmap) bitmap?.recycle()
                            isAnalyzing.set(false)
                        }
                    }
                    "recognizing" -> {
                        lifecycleScope.launch(Dispatchers.Default) {
                            processRecognitionFrame(rotatedBitmap)
                            rotatedBitmap.recycle()
                            if (bitmap != rotatedBitmap) bitmap?.recycle()
                            isAnalyzing.set(false)
                        }
                    }
                    else -> {
                        rotatedBitmap.recycle()
                        if (bitmap != rotatedBitmap) bitmap?.recycle()
                        isAnalyzing.set(false)
                    }
                }
            } else {
                isAnalyzing.set(false)
            }
        } catch (e: Exception) {
            Log.e("Camera", "Ошибка обработки кадра: ${e.message}")
            isAnalyzing.set(false)
        } finally {
            imageProxy.close()
        }
    }

    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
        return try {
            val buffer = imageProxy.planes[0].buffer
            val bytes = ByteArray(buffer.remaining())
            buffer.get(bytes)

            Bitmap.createBitmap(
                imageProxy.width,
                imageProxy.height,
                Bitmap.Config.ARGB_8888
            ).apply {
                copyPixelsFromBuffer(ByteBuffer.wrap(bytes))
            }
        } catch (e: Exception) {
            Log.e("Camera", "Ошибка конвертации: ${e.message}")
            null
        }
    }

    private fun rotateBitmap(bitmap: Bitmap?, rotationDegrees: Int, isFrontCamera: Boolean): Bitmap? {
        if (bitmap == null) return null

        return try {
            val matrix = Matrix()
            matrix.postRotate(rotationDegrees.toFloat())

            if (isFrontCamera) {
                matrix.postScale(-1f, 1f, bitmap.width / 2f, bitmap.height / 2f)
            }

            Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        } catch (e: Exception) {
            Log.e("Camera", "Ошибка поворота: ${e.message}")
            bitmap
        }
    }

    private suspend fun processRecognitionFrame(bitmap: Bitmap) {
        val facesWithEncoding = faceEngine.getFaceEncodings(bitmap)

        if (facesWithEncoding.isNotEmpty()) {
            val results = facesWithEncoding.map { face ->
                FaceDetectionResult(
                    x = face.rect.x.toFloat(),
                    y = face.rect.y.toFloat(),
                    width = face.rect.width.toFloat(),
                    height = face.rect.height.toFloat(),
                    name = face.name,
                    confidence = face.confidence,
                    recognized = face.name != "Неизвестный" && face.confidence > 0.65f
                )
            }

            withContext(Dispatchers.Main) {
                faceOverlay.detectedFaces = results

                if (results.isNotEmpty()) {
                    val recognizedCount = results.count { it.recognized }
                    tvMode.text = "👥 Лиц: ${results.size} (узнано: $recognizedCount)"
                } else {
                    tvMode.text = "❓ Лиц не найдено"
                }
            }
        } else {
            withContext(Dispatchers.Main) {
                faceOverlay.detectedFaces = emptyList()
                tvMode.text = "❓ Лиц не найдено"
            }
        }
    }

    private suspend fun processRetrainingFrame(bitmap: Bitmap) {
        val facesWithEncoding = faceEngine.getFaceEncodings(bitmap)

        if (facesWithEncoding.isNotEmpty() && isTraining) {
            val face = facesWithEncoding[0]

            if (face.encoding.all { it == 0f }) return

            if (trainingSamples.size < maxSamples) {
                trainingSamples.add(face.encoding)
                trainingCount = trainingSamples.size

                withContext(Dispatchers.Main) {
                    progressBar.progress = trainingCount
                    tvSamples.text = "Образцов: $trainingCount/$maxSamples"
                }

                if (trainingSamples.size >= maxSamples) {
                    finishRetraining()
                    return
                }
            }

            val result = FaceDetectionResult(
                x = face.rect.x.toFloat(),
                y = face.rect.y.toFloat(),
                width = face.rect.width.toFloat(),
                height = face.rect.height.toFloat(),
                name = "Обучение: $trainingCount/$maxSamples",
                confidence = 0f,
                recognized = false
            )

            withContext(Dispatchers.Main) {
                faceOverlay.detectedFaces = listOf(result)
            }
        }
    }

    private fun startRetraining(userName: String) {
        isTraining = true
        currentTrainingUser = userName
        trainingSamples.clear()
        trainingCount = 0
        currentMode = "retraining"

        faceEngine.resetTracking()

        updateModeStatus("retraining", userName)

        btnRetrain.isEnabled = false
        btnRecognize.isEnabled = false
        btnStop.isEnabled = true
        btnStartServer.isEnabled = false
        btnSwitchCamera.isEnabled = false

        progressBar.visibility = View.VISIBLE
        tvSamples.visibility = View.VISIBLE
        progressBar.max = maxSamples
        progressBar.progress = 0
        tvSamples.text = "Образцов: 0/$maxSamples"

        Toast.makeText(this, "📸 Дообучение для $userName", Toast.LENGTH_LONG).show()
    }

    private fun finishRetraining() {
        isTraining = false
        currentMode = "idle"

        if (trainingSamples.size >= 8) {
            val embeddingSize = trainingSamples[0].size
            val avgEncoding = FloatArray(embeddingSize)

            for (sample in trainingSamples) {
                for (i in sample.indices) {
                    avgEncoding[i] += sample[i]
                }
            }
            for (i in avgEncoding.indices) {
                avgEncoding[i] = avgEncoding[i] / trainingSamples.size
            }

            var norm = 0f
            for (i in avgEncoding.indices) {
                norm += avgEncoding[i] * avgEncoding[i]
            }
            norm = sqrt(norm)
            if (norm > 0) {
                for (i in avgEncoding.indices) {
                    avgEncoding[i] /= norm
                }
            }

            val success = faceEngine.addFace(currentTrainingUser, avgEncoding)

            runOnUiThread {
                if (success) {
                    updateUsersList()
                    showModelInfo()
                    Toast.makeText(this, "✅ Дообучение завершено для $currentTrainingUser!", Toast.LENGTH_LONG).show()
                } else {
                    Toast.makeText(this, "❌ Ошибка дообучения", Toast.LENGTH_SHORT).show()
                }
                stopAll()
            }
        } else {
            runOnUiThread {
                Toast.makeText(this, "❌ Недостаточно образцов (нужно минимум 8)", Toast.LENGTH_SHORT).show()
                stopAll()
            }
        }
    }

    private fun startRecognition() {
        if (!faceEngine.isModelLoaded()) {
            Toast.makeText(this, "❌ Модель не загружена!", Toast.LENGTH_LONG).show()
            return
        }

        if (isServerRunning) stopServer()

        currentMode = "recognizing"

        faceEngine.resetTracking()

        updateModeStatus("recognizing")

        btnRecognize.isEnabled = false
        btnStop.isEnabled = true
        btnRetrain.isEnabled = false
        btnStartServer.isEnabled = false
        btnSwitchCamera.isEnabled = false

        faceOverlay.detectedFaces = emptyList()
    }

    private fun stopAll() {
        currentMode = "idle"
        isTraining = false
        trainingSamples.clear()

        updateModeStatus("idle")

        btnRecognize.isEnabled = true
        btnStop.isEnabled = false
        btnRetrain.isEnabled = true
        btnStartServer.isEnabled = true
        btnSwitchCamera.isEnabled = true

        progressBar.visibility = View.GONE
        tvSamples.visibility = View.GONE
        faceOverlay.detectedFaces = emptyList()
    }

    private fun startServer() {
        if (!faceEngine.isModelLoaded()) {
            Toast.makeText(this, "❌ Модель не загружена!", Toast.LENGTH_LONG).show()
            return
        }

        if (currentMode == "recognizing" || currentMode == "retraining") {
            stopAll()
        }

        Toast.makeText(this, "🚀 Запуск сервера...", Toast.LENGTH_SHORT).show()
        faceServer = FaceRecognitionServer(this, faceEngine)
        faceServer!!.start()
        isServerRunning = true
        updateServerStatus(true)

        val ip = getLocalIpAddress()
        tvServerIp.text = "http://$ip:5000"
        updateModeStatus("server")
    }

    private fun stopServer() {
        faceServer?.stop()
        faceServer = null
        isServerRunning = false
        updateServerStatus(false)
        tvServerIp.text = "Сервер остановлен"

        if (currentMode == "recognizing" || currentMode == "retraining") {
            stopAll()
        }

        Toast.makeText(this, "⏹️ Сервер остановлен", Toast.LENGTH_SHORT).show()
    }

    private fun updateServerStatus(running: Boolean) {
        if (running) {
            btnStartServer.text = "⏹️ Остановить сервер"
            btnStartServer.setBackgroundColor(ContextCompat.getColor(this, R.color.red))
        } else {
            btnStartServer.text = "🚀 Запустить сервер"
            btnStartServer.setBackgroundColor(ContextCompat.getColor(this, R.color.purple))
        }
    }

    private fun updateFps() {
        frameCount++
        val currentTime = System.currentTimeMillis()
        if (currentTime - lastFpsUpdate >= 1000) {
            currentFps = frameCount
            frameCount = 0
            lastFpsUpdate = currentTime
        }
    }

    private fun updateFpsDisplay() {
        runOnUiThread {
            tvFps.text = "$currentFps FPS"
        }
    }

    private fun updateModeStatus(mode: String, userName: String = "") {
        runOnUiThread {
            when (mode) {
                "retraining" -> {
                    tvStatus.text = "📸 ДООБУЧЕНИЕ"
                    tvMode.text = "Режим: Дообучение - $userName"
                    tvStatus.setBackgroundColor(ContextCompat.getColor(this, R.color.green))
                }
                "recognizing" -> {
                    tvStatus.text = "🔍 РАСПОЗНАВАНИЕ"
                    tvMode.text = "Режим: Распознавание"
                    tvStatus.setBackgroundColor(ContextCompat.getColor(this, R.color.orange))
                }
                "server" -> {
                    tvStatus.text = "🌐 СЕРВЕР"
                    tvMode.text = "Режим: Сервер"
                    tvStatus.setBackgroundColor(ContextCompat.getColor(this, R.color.purple))
                }
                else -> {
                    tvStatus.text = "⚡ ГОТОВ"
                    tvMode.text = "Режим: Ожидание"
                    tvStatus.setBackgroundColor(ContextCompat.getColor(this, R.color.gray))
                }
            }
        }
    }

    private fun loadModelAndUpdateUI() {
        lifecycleScope.launch(Dispatchers.IO) {
            val success = faceEngine.loadModel()
            launch(Dispatchers.Main) {
                if (success) {
                    updateUsersList()
                    showModelInfo()
                    updateModeStatus("idle")
                } else {
                    tvModelInfo.text = "❌ Ошибка загрузки модели"
                    tvModelInfo.setTextColor(Color.RED)
                }
            }
        }
    }

    private fun updateUsersList() {
        usersList.clear()
        usersList.add("➕ Новый пользователь")

        val existingUsers = faceEngine.getUsers()
        if (existingUsers.isNotEmpty()) {
            usersList.addAll(existingUsers)
        }

        usersAdapter.notifyDataSetChanged()

        if (existingUsers.isNotEmpty()) {
            showModelInfo()
        }
    }

    private fun showModelInfo() {
        val users = faceEngine.getUsers()
        if (users.isEmpty()) {
            tvModelInfo.text = "📁 База пуста. Добавьте пользователя через меню"
            tvModelInfo.setTextColor(Color.YELLOW)
        } else {
            tvModelInfo.text = "📁 Модель: ${users.size} пользователей\n${users.joinToString(", ")}"
            tvModelInfo.setTextColor(Color.GREEN)
        }
    }

    private fun openMenu() {
        isMenuOpen = true
        controlPanel.visibility = View.VISIBLE

        val slideUp = ObjectAnimator.ofFloat(controlPanel, "translationY", controlPanel.height.toFloat(), 0f)
        slideUp.duration = 300
        slideUp.interpolator = DecelerateInterpolator()

        val fadeIn = ObjectAnimator.ofFloat(controlPanel, "alpha", 0f, 1f)
        fadeIn.duration = 300

        val rotate = ObjectAnimator.ofFloat(btnMenu, "rotation", 0f, 90f)
        rotate.duration = 300

        val animSet = AnimatorSet()
        animSet.playTogether(slideUp, fadeIn, rotate)
        animSet.start()
    }

    private fun closeMenu() {
        isMenuOpen = false

        val slideDown = ObjectAnimator.ofFloat(controlPanel, "translationY", 0f, controlPanel.height.toFloat())
        slideDown.duration = 250
        slideDown.interpolator = DecelerateInterpolator()

        val fadeOut = ObjectAnimator.ofFloat(controlPanel, "alpha", 1f, 0f)
        fadeOut.duration = 250

        val rotate = ObjectAnimator.ofFloat(btnMenu, "rotation", 90f, 0f)
        rotate.duration = 250

        val animSet = AnimatorSet()
        animSet.playTogether(slideDown, fadeOut, rotate)
        animSet.start()

        animSet.doOnEnd {
            controlPanel.visibility = View.GONE
        }
    }

    private fun getLocalIpAddress(): String {
        return try {
            val interfaces = java.net.NetworkInterface.getNetworkInterfaces()
            while (interfaces.hasMoreElements()) {
                val intf = interfaces.nextElement()
                val addresses = intf.inetAddresses
                while (addresses.hasMoreElements()) {
                    val addr = addresses.nextElement()
                    if (!addr.isLoopbackAddress && addr is java.net.InetAddress) {
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

    private fun hideSystemUI() {
        window.decorView.systemUiVisibility = (
                View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
                        or View.SYSTEM_UI_FLAG_FULLSCREEN
                        or View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                )
    }

    private fun checkPermissions() {
        val missingPermissions = requiredPermissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }

        if (missingPermissions.isNotEmpty()) {
            requestPermissionLauncher.launch(missingPermissions.toTypedArray())
        } else {
            startCamera()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        stopServer()
        cameraExecutor.shutdown()
        faceEngine.close()
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        if (hasFocus) {
            hideSystemUI()
        }
    }

    // Extension function for AnimatorSet
    private fun AnimatorSet.doOnEnd(action: () -> Unit) {
        this.addListener(object : android.animation.AnimatorListenerAdapter() {
            override fun onAnimationEnd(animation: android.animation.Animator) {
                action()
            }
        })
    }
}