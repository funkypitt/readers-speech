package com.freedomfighter.readers.speech.whisper

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Test
import java.io.File

/**
 * The detector's window and dominant-speaker logic against thirty seconds of a real interview
 * (a question, then the guest's answer), with what the Python reference of the study found on
 * the same samples. The network itself is C and cannot run on the JVM: it is proved on the host
 * by tools/pyannet-check.sh against onnxruntime (log-probabilities within 1e-3, every frame's
 * class the same); here it is replaced by the frames that reference computed on the fixture's
 * windows (fixture-frames.json), so that the Kotlin layer is tested on the real model's answers.
 */
class VoicesTest {
    private fun resource(name: String) = File("src/test/resources/$name").takeIf { it.isFile } ?: File("speech/src/test/resources/$name")

    /** 16 kHz mono 16-bit wav, as the fixture is: the samples after the 44-byte header. */
    private fun wav(name: String): FloatArray {
        val bytes = resource(name).readBytes()
        val n = (bytes.size - 44) / 2
        return FloatArray(n) { i -> (((bytes[44 + 2 * i + 1].toInt() shl 8) or (bytes[44 + 2 * i].toInt() and 0xff)).toShort() / 32768f) }
    }

    /** The recorded answers of the model, window by window, as the log-probabilities the C side would return. */
    private fun recorded(): FrameRunner {
        val arr = JSONObject(resource("fixture-frames.json").readText()).getJSONArray("windows")
        val windows = (0 until arr.length()).map { arr.getJSONObject(it) }.associate { w ->
            val classes = w.getJSONArray("classes")
            (w.getInt("offset") to w.getInt("n")) to IntArray(classes.length()) { classes.getInt(it) }
        }
        return { _, offset, n ->
            val classes = windows[offset to n] ?: throw AssertionError("the detector looked at a window the reference did not: $offset+$n")
            FloatArray(classes.size * 7) { i -> if (i % 7 == classes[i / 7]) -0.01f else -9f }
        }
    }

    @Test fun theAnswerStartsANewVoiceAndNothingElseDoes() {
        val fixture = JSONObject(resource("fixture.json").readText())
        val arr = fixture.getJSONArray("lines")
        val lines = (0 until arr.length()).map { i -> arr.getJSONObject(i).let { Segment(it.getLong("start_ms"), it.getLong("end_ms"), it.getString("text")) } }
        val expected = fixture.getJSONObject("expected_gate").getJSONArray("changes").let { e -> (0 until e.length()).map { e.getBoolean(it) } }
        val pcm = wav("fixture.wav")
        val marked = Voices(recorded()).use { it.mark(pcm, lines) }
        assertEquals(expected, marked.map { it.newVoice })
        assertEquals(true, marked[3].text.startsWith("Yeah, so I think"))
    }

    @Test fun everyGatedGapIsLookedAtAndTheOthersAreNot() {
        val fixture = JSONObject(resource("fixture.json").readText())
        val arr = fixture.getJSONArray("lines")
        val lines = (0 until arr.length()).map { i -> arr.getJSONObject(i).let { Segment(it.getLong("start_ms"), it.getLong("end_ms"), it.getString("text")) } }
        val gaps = fixture.getJSONObject("expected_gate").getJSONArray("gaps").let { g -> (0 until g.length()).map { g.getJSONObject(it).getInt("gap_sample") } }
        val seen = mutableListOf<Int>()
        val runner = recorded()
        val spy: FrameRunner = { pcm, offset, n -> seen += offset; runner(pcm, offset, n) }
        Voices(spy).use { it.mark(wav("fixture.wav"), lines) }
        assertEquals(gaps.map { maxOf(0, it - 80_000) }, seen)
    }

    @Test fun tooLittleSoundSaysNothing() {
        val never: FrameRunner = { _, _, _ -> throw AssertionError("the network must not run on under a second of sound") }
        Voices(never).use { assertEquals(null, it.changeAt(FloatArray(8_000), 4_000)) }
    }

    @Test fun silenceOnOneSideSaysNothing() {
        val nobody: FrameRunner = { _, _, n -> FloatArray(n / 270 * 7) { i -> if (i % 7 == 0) -0.01f else -9f } }
        Voices(nobody).use { assertEquals(null, it.changeAt(FloatArray(160_000), 80_000)) }
    }
}
