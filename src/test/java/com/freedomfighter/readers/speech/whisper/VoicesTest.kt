package com.freedomfighter.readers.speech.whisper

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Test
import java.io.File

/**
 * The detector against thirty seconds of a real interview (a question, then the guest's
 * answer), with what the Python reference of the study found on the same samples — the model
 * in the assets, run by the same engine on the JVM.
 */
class VoicesTest {
    private fun resource(name: String) = File("src/test/resources/$name").takeIf { it.isFile } ?: File("speech/src/test/resources/$name")
    private fun model() = (File("src/main/assets/${Voices.FILE}").takeIf { it.isFile } ?: File("speech/src/main/assets/${Voices.FILE}")).readBytes()

    /** 16 kHz mono 16-bit wav, as the fixture is: the samples after the 44-byte header. */
    private fun wav(name: String): FloatArray {
        val bytes = resource(name).readBytes()
        val n = (bytes.size - 44) / 2
        return FloatArray(n) { i -> (((bytes[44 + 2 * i + 1].toInt() shl 8) or (bytes[44 + 2 * i].toInt() and 0xff)).toShort() / 32768f) }
    }

    @Test fun theAnswerStartsANewVoiceAndNothingElseDoes() {
        val fixture = JSONObject(resource("fixture.json").readText())
        val arr = fixture.getJSONArray("lines")
        val lines = (0 until arr.length()).map { i -> arr.getJSONObject(i).let { Segment(it.getLong("start_ms"), it.getLong("end_ms"), it.getString("text")) } }
        val expected = fixture.getJSONObject("expected_gate").getJSONArray("changes").let { e -> (0 until e.length()).map { e.getBoolean(it) } }
        val pcm = wav("fixture.wav")
        val t0 = System.nanoTime()
        val marked = Voices.fromBytes(model()).use { it.mark(pcm, lines) }
        val ms = (System.nanoTime() - t0) / 1_000_000
        assertEquals(expected, marked.map { it.newVoice })
        assertEquals(true, marked[3].text.startsWith("Yeah, so I think"))
        println("voices: ${lines.size - 1} gaps over ${pcm.size / 16_000} s in $ms ms")
    }

    @Test fun tooLittleSoundSaysNothing() {
        Voices.fromBytes(model()).use { assertEquals(null, it.changeAt(FloatArray(8_000), 4_000)) }
    }
}
