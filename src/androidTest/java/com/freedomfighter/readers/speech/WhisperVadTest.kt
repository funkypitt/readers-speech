package com.freedomfighter.readers.speech

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.freedomfighter.readers.speech.whisper.Vad
import com.freedomfighter.readers.speech.whisper.WhisperSession
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * The native path, on a real Android: that the JNI signatures still match, that the detector
 * comes out of the assets, and that a passage with silence in it comes back as words.
 *
 * It needs two files the repository does not carry, put in the test app's own files directory
 * (they are large, and one of them is a model):
 *
 *     adb push ggml-small-q5_1.bin /data/local/tmp/
 *     adb push speech.f32 /data/local/tmp/            # 16 kHz mono float32
 *     adb shell "run-as <package> sh -c 'mkdir -p files/models;
 *         cat /data/local/tmp/ggml-small-q5_1.bin > files/models/ggml-small-q5_1.bin;
 *         cat /data/local/tmp/speech.f32 > files/speech.f32'"
 *
 * Without them the test says so and passes: it is a check one runs deliberately, not a gate.
 */
@RunWith(AndroidJUnit4::class)
class WhisperVadTest {
    private val ctx = InstrumentationRegistry.getInstrumentation().targetContext

    @Test fun detectorComesOutOfTheAssets() {
        val path = Vad.modelPath(ctx)
        assertNotNull("the detector should be unpacked", path)
        assertEquals(885_098L, File(path!!).length())
        // Asked twice, it is not written twice.
        assertEquals(path, Vad.modelPath(ctx))
    }

    @Test fun aPassageWithSilenceComesBackAsWords() {
        val model = File(ctx.filesDir, "models/ggml-small-q5_1.bin")
        val audio = File(ctx.filesDir, "speech.f32")
        assumeTrue("push the model and a piece of speech first (see the file's head)", model.isFile && audio.isFile)
        val pcm = FloatArray(audio.length().toInt() / 4)
        java.io.DataInputStream(audio.inputStream().buffered()).use { input ->
            val bytes = ByteArray(pcm.size * 4)
            input.readFully(bytes)
            java.nio.ByteBuffer.wrap(bytes).order(java.nio.ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(pcm)
        }
        WhisperSession(model.path, Vad.modelPath(ctx)).use { session ->
            val segments = session.run(pcm, "fr", "Bonjour. Voici une transcription soignée.") {}
            assertNotNull("the detector must not cost the transcription", segments)
            assertTrue("something should have been heard", segments!!.joinToString(" ") { it.text }.split(" ").size > 5)
        }
    }
}
