package ai.onnxruntime.example.whisperLocal

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import android.annotation.SuppressLint
import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import com.konovalov.vad.silero.Vad
import com.konovalov.vad.silero.VadSilero
import com.konovalov.vad.silero.config.FrameSize
import com.konovalov.vad.silero.config.Mode
import com.konovalov.vad.silero.config.SampleRate
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.LinkedList
import java.util.concurrent.atomic.AtomicBoolean

class AudioTensorSource {
    companion object {
        private const val bytesPerFloat = 4
        private const val sampleRate = 16000
        private const val maxAudioLengthInSeconds = 30

        fun fromRawPcmBytes(rawBytes: ByteArray): OnnxTensor {
            val rawByteBuffer = ByteBuffer.wrap(rawBytes)
            // TODO handle big-endian native order...
            if (ByteOrder.nativeOrder() != ByteOrder.LITTLE_ENDIAN) {
                throw NotImplementedError("Reading PCM data is only supported when native byte order is little-endian.")
            }
            rawByteBuffer.order(ByteOrder.nativeOrder())
            val floatBuffer = rawByteBuffer.asFloatBuffer()
            val numSamples = minOf(floatBuffer.capacity(), maxAudioLengthInSeconds * sampleRate)
            val env = OrtEnvironment.getEnvironment()
            return OnnxTensor.createTensor(
                env, floatBuffer, tensorShape(1, numSamples.toLong())
            )
        }

        @SuppressLint("MissingPermission")
        fun fromRecording(context: Context, stopRecordingFlag: AtomicBoolean) = sequence {
            val audioRecord = createAudioRecord()
            val vad = createVad(context)

            try {
                audioRecord.startRecording()
                val chunkList = LinkedList<FloatArray>()
                var isPreviousSpeech = false

                while (!stopRecordingFlag.get()) {
                    val audioBuffer = FloatArray(FrameSize.FRAME_SIZE_512.value)
                    var length = 0;
                    while (length < audioBuffer.size) {
                        val result = audioRecord.read(
                            audioBuffer,
                            length,
                            audioBuffer.size - length,
                            AudioRecord.READ_BLOCKING
                        )
                        if (result < 0) {
                            throw RuntimeException("AudioRecord.read() error with code $result")
                        }
                        length += result

                        if (stopRecordingFlag.get()) {
                            break
                        }
                    }

                    val isCurrentSpeech = vad.isSpeech(audioBuffer)
                    var inputTensor: OnnxTensor? = null
                    chunkList.add(audioBuffer)
                    if (isCurrentSpeech) {
                        // speaking
                    } else if (isPreviousSpeech) {
                        inputTensor = processSpeechEnd(chunkList, context)
                        chunkList.clear()
                    } else {
                        val limitSize = 400f / ((FrameSize.FRAME_SIZE_512.value / sampleRate.toFloat()) * 1000)
                        while (chunkList.size > limitSize) {
                            chunkList.removeFirst()
                        }
                    }
                    isPreviousSpeech = isCurrentSpeech
                    if (inputTensor != null) {
                        yield(inputTensor)
                    }
                }
            } finally {
                if (audioRecord.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                    audioRecord.stop()
                }
                audioRecord.release()
            }
        }

        @SuppressLint("MissingPermission")
        private fun createAudioRecord(): AudioRecord {
            var recordingChunkLengthInSeconds = 1

            val minBufferSize = maxOf(
                AudioRecord.getMinBufferSize(
                    sampleRate,
                    AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_FLOAT
                ),
                2 * recordingChunkLengthInSeconds * sampleRate * bytesPerFloat
            )
            return AudioRecord.Builder()
                .setAudioSource(MediaRecorder.AudioSource.MIC)
                .setAudioFormat(
                    AudioFormat.Builder()
                        .setSampleRate(sampleRate)
                        .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                        .setChannelMask(AudioFormat.CHANNEL_IN_MONO)
                        .build()
                )
                .setBufferSizeInBytes(minBufferSize)
                .build()
        }

        private fun createVad(context: Context): VadSilero {
            val vad = Vad.builder()
                .setContext(context)
                .setSampleRate(SampleRate.SAMPLE_RATE_16K)
                .setFrameSize(FrameSize.FRAME_SIZE_512)
                .setMode(Mode.NORMAL)
                .setSilenceDurationMs(300)
                .setSpeechDurationMs(50)
                .build()
            return vad
        }

        private fun processSpeechEnd(chunkList: MutableList<FloatArray>, context: Context): OnnxTensor {
            Log.d(MainActivity.TAG, "== SPEECH END ==: ${chunkList.size}")
            writePcmDataToFile(chunkList, context)
            val fb = FloatBuffer.allocate(maxAudioLengthInSeconds * sampleRate)
            chunkList.stream().forEach(fb::put)
            fb.flip().limit(fb.capacity())
            val env = OrtEnvironment.getEnvironment()
            return OnnxTensor.createTensor(env, fb, tensorShape(1, fb.capacity().toLong()))
        }

        private fun writePcmDataToFile(chunkList: MutableList<FloatArray>, context: Context) {
            val SAMPLE_RATE = 16000 // Hz
            val CHANNELS = 1 // Mono audio
            val BYTES_PER_FLOAT = 4
            val BITS_PER_SAMPLE = 32
            val AUDIO_FORMAT_FLOAT = 3 // IEEE float
            val HEADER_SIZE = 44 // WAV standard header size

            val byteRate = SAMPLE_RATE * CHANNELS * BYTES_PER_FLOAT
            val dataSize = chunkList.size * FrameSize.FRAME_SIZE_512.value * BYTES_PER_FLOAT

            val totalDataLen = dataSize + HEADER_SIZE - 8
            val header = ByteBuffer.allocate(HEADER_SIZE).order(ByteOrder.LITTLE_ENDIAN)

            header.put("RIFF".toByteArray(Charsets.US_ASCII))           // ChunkID
            header.putInt(totalDataLen)                                 // ChunkSize
            header.put("WAVE".toByteArray(Charsets.US_ASCII))           // Format
            header.put("fmt ".toByteArray(Charsets.US_ASCII))           // Subchunk1ID
            header.putInt(16)                                     // Subchunk1Size (16 for PCM)
            header.putShort(AUDIO_FORMAT_FLOAT.toShort())               // AudioFormat (3 for IEEE float)
            header.putShort(CHANNELS.toShort())                         // NumChannels
            header.putInt(SAMPLE_RATE)                                  // SampleRate
            header.putInt(byteRate)                                     // ByteRate
            header.putShort((CHANNELS * BYTES_PER_FLOAT).toShort())     // BlockAlign
            header.putShort(BITS_PER_SAMPLE.toShort())                  // BitsPerSample
            header.put("data".toByteArray(Charsets.US_ASCII))           // Subchunk2ID
            header.putInt(dataSize)                                     // Subchunk2Size

            val fileOutputStream = context.openFileOutput("debug.wav", AppCompatActivity.MODE_PRIVATE)
            fileOutputStream.use { fos ->
                fos.write(header.array())
                for (chunk in chunkList) {
                    for (value in chunk) {
                        val bytes = ByteBuffer
                            .allocate(4)
                            .order(ByteOrder.LITTLE_ENDIAN)
                            .putFloat(value)
                            .array()
                        fos.write(bytes)
                    }
                }
            }
        }
    }
}