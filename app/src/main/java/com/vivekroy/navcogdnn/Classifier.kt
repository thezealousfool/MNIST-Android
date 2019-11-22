package com.vivekroy.navcogdnn

import android.app.Activity
import android.graphics.Bitmap
import android.os.SystemClock
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel.MapMode.READ_ONLY


class Result(probs: FloatArray, val timeCost: Long) {

    val number: Int
    val probability: Float

    init {
        number = argmax(probs)
        probability = probs[number]
    }

    private fun argmax(probs: FloatArray): Int {
        var maxIdx = -1
        var maxProb = 0.0f
        for (i in probs.indices) {
            if (probs[i] > maxProb) {
                maxProb = probs[i]
                maxIdx = i
            }
        }
        return maxIdx
    }
}


class Classifier(activity: Activity) {
    private val tflite : Interpreter
    private val options = Interpreter.Options()
    private val imageData = ByteBuffer.allocateDirect(byteCount)

    companion object {
        const val MODEL_NAME = "mnist.tflite"
        const val byteCount = 28 * 28 * 4
        const val NUM_CLASSES = 10
    }
    init {
        tflite = Interpreter(loadModelFile(activity), options)
        imageData.order(ByteOrder.nativeOrder())
    }

    private fun loadModelFile(activity : Activity) : MappedByteBuffer {
        val fileDescriptor = activity.assets.openFd(MODEL_NAME)
        return FileInputStream(fileDescriptor.fileDescriptor).channel.map(
            READ_ONLY, fileDescriptor.startOffset, fileDescriptor.declaredLength)
    }

    fun classify(bitmap: Bitmap): Result {
        convertBitmapToByteBuffer(bitmap)
        val startTime = SystemClock.uptimeMillis()
        val result = Array(1) { FloatArray(NUM_CLASSES) }
        tflite.run(imageData, result)
        val endTime = SystemClock.uptimeMillis()
        val timeCost = endTime - startTime
        return Result(result[0], timeCost)
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap) {
        imageData.rewind()

        for (i in 0 until 28) {
            for (j in 0 until 28) {
                val value = bitmap.getPixel(j,i)
                imageData.putFloat((value and 0xFF)/1.0f)
            }
        }
    }
}