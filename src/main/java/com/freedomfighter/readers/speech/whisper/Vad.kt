package com.freedomfighter.readers.speech.whisper

import android.content.Context
import java.io.File

/**
 * Silero, the voice activity detector: it finds where there is speech, and whisper is given only
 * that. Eight hundred and eighty-five kilobytes, carried in the assets, so nothing is downloaded.
 *
 * Measured on 2026-09-18 (a LibriVox reading of Daudet against Gutenberg's own text, three
 * minutes, the same sources as the module and the style prompt the apps use): on speech with
 * pauses it saves a tenth of the time and, with the ordinary model, two points of word error
 * (12.1 → 10.0 %); on speech with almost no silence it changes nothing either way. The padding
 * is what makes it safe — at 30 ms the detector eats the first words of a sentence and the error
 * rate goes the wrong way (12.8 %), at 250 ms it does not.
 */
object Vad {
    const val FILE = "ggml-silero-v5.1.2.bin"

    /** Kept before and after each stretch of speech. Below about 200 ms, words are lost. */
    const val PAD_MS = 250

    /**
     * The detector as a file whisper can open, unpacked once; null if it cannot be had.
     *
     * The name carries the version, so a new detector lands beside the old one rather than
     * over it — and the old one is swept away. An asset is compressed in the package, which is
     * why it is copied out rather than opened where it lies.
     */
    fun modelPath(ctx: Context): String? = runCatching {
        val dir = File(ctx.filesDir, "vad").apply { mkdirs() }
        val target = File(dir, FILE)
        if (!target.isFile) {
            val part = File(dir, "$FILE.part")
            ctx.assets.open(FILE).use { input -> part.outputStream().use { input.copyTo(it) } }
            part.renameTo(target)
            dir.listFiles()?.forEach { if (it.name != FILE) it.delete() }
        }
        target.path
    }.getOrNull()
}
