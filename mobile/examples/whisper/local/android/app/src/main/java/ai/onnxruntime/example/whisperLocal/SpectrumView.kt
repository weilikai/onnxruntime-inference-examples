package ai.onnxruntime.example.whisperLocal

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import demo.weilikai.simpleasr.mfcc.FFT
import java.io.BufferedInputStream
import kotlin.concurrent.thread
import java.io.DataInputStream
import java.io.EOFException
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

class SpectrumView(context: Context, attrs: AttributeSet) : View(context, attrs) {
    private var amplitudes = mutableListOf<DoubleArray>()
    private val paint = Paint()

    init {
        paint.strokeWidth = 4f
    }

    fun readWavFromStream(input: InputStream) = sequence {
        val input2 = BufferedInputStream(input)
        val din = DataInputStream(input)
        din.use {
            val window = ByteArray(4 * 512)
            while (true) {
                try {
                    input2.mark(4 * 512*10)
                    val bytesRead = din.read(window)
                    if (bytesRead < window.size) break

                    val floatWindow = FloatArray(512)
                    ByteBuffer.wrap(window).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(floatWindow)
                    yield(floatWindow)
                    input2.reset()
                    input2.skip(4*160)
                } catch (e: EOFException) {
                    break
                }
            }
        }
    }

    fun loadWav(input: InputStream) {
        thread(start = true) {
            input.use {
                it.skip(44)
                val windows = readWavFromStream(it)
                amplitudes.clear()
                for (window in windows) {
                    val fftResult = FFT.fft(window)
                    val energies = FFT.computeEnergies(fftResult)
                    val half = energies.size / 2 + 1
                    val positiveEnergies = energies.copyOfRange(0, half)
                    amplitudes.add(positiveEnergies)
                }
                postInvalidate()
            }
        }
    }

    private fun getEnergyColor(energy: Double, maxEnergy: Double): Int {
        val ratio = energy / maxEnergy
        val hue = (1 - ratio) * 240 // Hue goes from 0 to 240 (from red to blue in HSV color space)
        val hsv = floatArrayOf(hue.toFloat(), 1f, 1f)
        return Color.HSVToColor(hsv)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawColor(Color.BLACK)
    
        if (amplitudes.isEmpty()) return

        val timeStep = width.toFloat() / amplitudes.size.toFloat()
        val freqStep = height.toFloat() / amplitudes.first().size.toFloat()

        for (time in amplitudes.indices) {
            val energies = amplitudes[time]
            for (freq in energies.indices) {
                paint.color = getEnergyGrayColor(energies[freq], 10.0)
                val x = time * timeStep
                val y = height - freq * freqStep
//                canvas.drawPoint(x, y, paint)

                val rectLeft = time * timeStep
                val rectTop = y
                val rectRight = rectLeft + timeStep
                val rectBottom = rectTop + freqStep

                canvas.drawRect(rectLeft, rectTop, rectRight, rectBottom, paint)
            }
        }
    }

    fun getEnergyGrayColor(energy: Double, maxEnergy: Double): Int {
        val minLogEnergy = 1e-10
        val normalizedEnergy = (energy / maxEnergy).coerceAtLeast(minLogEnergy)
        val logEnergy = Math.pow(normalizedEnergy, logBase)
        val adjustedLogEnergy = (logEnergy).coerceIn(0.0, 1.0)
        val intensity = (adjustedLogEnergy * 200).toInt().coerceIn(0, 200)
        return Color.rgb(intensity, 0, 0)
    }

    private var initialTouchY = 0f // 初始触控点的Y坐标
    private var currentTouchY = 0f // 当前触控点的Y坐标
    private var logBase = 0.2      // 对数的初始底数
    private var deltaY = 0.0
    private var lastUpdate: Long = 0

    override fun onTouchEvent(event: MotionEvent): Boolean {
        when (event.action and MotionEvent.ACTION_MASK) {
            MotionEvent.ACTION_POINTER_DOWN -> {
                if (event.pointerCount == 2) { // 当有两个手指按下时
                    initialTouchY = (event.getY(0) + event.getY(1)) / 2 // 记录初始Y坐标
                    currentTouchY = initialTouchY
                    deltaY = 0.0
                }
            }
            MotionEvent.ACTION_MOVE -> {
                if (event.pointerCount == 2) { // 当有两个手指移动时
                    currentTouchY = (event.getY(0) + event.getY(1)) / 2
                    deltaY = 0.1 * ((currentTouchY - initialTouchY) / height)
                    adjustLogBase()
                    if (System.currentTimeMillis() - lastUpdate > 50) {
                        invalidate()
                        lastUpdate = System.currentTimeMillis()
                    }
                }
            }
            MotionEvent.ACTION_POINTER_UP -> {
                if (event.pointerCount == 2) { // 当一个手指抬起时
                    initialTouchY = 0f
                    currentTouchY = 0f
                    deltaY = 0.0
                }
            }
        }
        return true
    }

    private fun adjustLogBase() {
        logBase += deltaY
        logBase = logBase.coerceIn(0.1, 1.0)
    }
}