package com.freedomfighter.readers.speech.whisper

import android.content.Context
import com.freedomfighter.readers.speech.share.Cpu
import kotlin.math.max
import kotlin.math.min

/** The JNI surface of pyannet.c, the segmentation network written in C (in the whisper library). */
object VoicesLib {
    init { Cpu.load("whisper") }
    /** The model from the bytes of the .bin asset; 0 when they are not it. */
    @JvmStatic external fun initModel(bytes: ByteArray): Long
    @JvmStatic external fun freeModel(ptr: Long)
    /** The network over pcm[offset, offset + n): frames × 7 log-probabilities, flat; null when it failed. */
    @JvmStatic external fun run(ptr: Long, pcm: FloatArray, offset: Int, n: Int): FloatArray?
}

/** What runs the network: over `pcm[offset, offset + n)`, the frames × 7 log-probabilities, flat. */
typealias FrameRunner = (pcm: FloatArray, offset: Int, n: Int) -> FloatArray

/**
 * Tells, line by line, where another person starts speaking — so that a question and its
 * answer become two paragraphs — without the full speaker diarization a phone cannot afford.
 *
 * The model is pyannote's segmentation network alone (`pyannote/segmentation-3.0`, 1.5 million
 * parameters, six megabytes, MIT), carried in the assets as plain weights and run by a C
 * implementation of the network in this module (`src/main/cpp/pyannet.c`), with no inference
 * engine: given ten seconds of sound it says, every 17 ms, which of up to three local speakers
 * are talking. It is run once per gap between two lines of whisper, on a window starting five
 * seconds before the gap, and the speaker who dominates the two seconds before is compared with
 * the one who dominates the two seconds after; a different one marks the line. Local is enough:
 * the paragraphs need a change, not an identity.
 *
 * Only the gaps where a paragraph may begin are looked at — the previous line ends a sentence,
 * or there is a pause of 200 ms — which halves the work. Measured on 2026-10-11 (study in
 * docs/etudes/paragraphes-2026-10-11, reproduced bit for bit by this engine): against pauses
 * alone, the changes of voice found on an interview go from a quarter to four fifths, for about
 * 50 ms of one desktop core per window — a minute or two per hour on a phone.
 *
 * The [runner] is the network; the real one is [VoicesLib], the tests hand in recorded frames.
 */
class Voices(private val runner: FrameRunner, private val release: () -> Unit = {}) : AutoCloseable {

    /** The lines with `newVoice` set where another person starts; the sound is the piece the lines belong to. */
    fun mark(pcm16k: FloatArray, segments: List<Segment>): List<Segment> {
        if (segments.size < 2) return segments
        val out = segments.toMutableList()
        for (i in 1 until segments.size) {
            val before = segments[i - 1]
            val after = segments[i]
            if (!(Paragraphs.endsSentence(before.text) || after.startMs - before.endMs >= GATE_MS)) continue
            val gap = ((before.endMs + after.startMs) * SAMPLE_RATE / 2000).toInt()
            if (changeAt(pcm16k, gap) == true) out[i] = after.copy(newVoice = true)
        }
        return out
    }

    /**
     * Whether the voice dominating the two seconds before the sample [gap] differs from the one
     * dominating the two seconds after. Null when there is too little sound, or nobody speaks on
     * one side (then the pauses alone decide).
     */
    fun changeAt(pcm16k: FloatArray, gap: Int): Boolean? {
        val a = max(0, gap - WINDOW / 2)
        val b = min(pcm16k.size, a + WINDOW)
        if (b - a < SAMPLE_RATE) return null
        val active = run(pcm16k, a, b)                       // per frame, which local speakers talk
        val frames = active.size
        if (frames == 0) return null
        fun frame(sample: Int) = ((sample - a).toLong() * frames / (b - a)).toInt()
        val before = dominant(active, frame(max(a, gap - MARGIN)), frame(gap)) ?: return null
        val after = dominant(active, frame(gap), min(frames, frame(gap + MARGIN))) ?: return null
        return before != after
    }

    /** The model over the samples [a, b), as the set of speakers active in each frame (the likeliest class). */
    private fun run(pcm16k: FloatArray, a: Int, b: Int): Array<BooleanArray> {
        val y = runner(pcm16k, a, b - a)                     // [frames][7 classes], log-probabilities, flat
        return Array(y.size / CLASSES.size) { f ->
            val base = f * CLASSES.size
            var best = 0
            for (k in 1 until CLASSES.size) if (y[base + k] > y[base + best]) best = k
            BooleanArray(SPEAKERS) { it in CLASSES[best] }
        }
    }

    /** The speaker active in the most frames of [from, to); null when nobody speaks there. */
    private fun dominant(active: Array<BooleanArray>, from: Int, to: Int): Int? {
        val counts = IntArray(SPEAKERS)
        for (f in from until max(to, from + 1).coerceAtMost(active.size)) for (s in 0 until SPEAKERS) if (active[f][s]) counts[s]++
        val best = counts.indices.maxBy { counts[it] }
        return if (counts[best] > 0) best else null
    }

    override fun close() = release()

    companion object {
        const val FILE = "pyannote-segmentation-3.0.bin"
        private const val SAMPLE_RATE = 16_000
        /** The model's window: ten seconds, 589 frames of 270 samples. */
        private const val WINDOW = 160_000
        /** Compared: the two seconds before the gap and the two seconds after. */
        private const val MARGIN = 2 * SAMPLE_RATE
        /** A gap is looked at when the line before ends a sentence or this pause is there. */
        private const val GATE_MS = 200L
        private const val SPEAKERS = 3
        /** The seven classes the model answers with: nobody, each speaker alone, each pair. */
        private val CLASSES = listOf(emptySet(), setOf(0), setOf(1), setOf(2), setOf(0, 1), setOf(0, 2), setOf(1, 2))

        /** The detector, from the model in the assets; null when it cannot be had (then the pauses decide alone). */
        fun load(ctx: Context): Voices? = runCatching {
            fromBytes(ctx.assets.open(FILE).use { it.readBytes() })
        }.getOrNull()

        /** The detector on the C network, from the bytes of the .bin file. */
        fun fromBytes(bytes: ByteArray): Voices {
            val ptr = VoicesLib.initModel(bytes)
            require(ptr != 0L) { "not the segmentation model" }
            return Voices(
                runner = { pcm, offset, n -> VoicesLib.run(ptr, pcm, offset, n) ?: error("segmentation failed") },
                release = { VoicesLib.freeModel(ptr) },
            )
        }
    }
}
