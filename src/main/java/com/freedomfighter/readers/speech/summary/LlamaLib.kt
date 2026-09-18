package com.freedomfighter.readers.speech.summary

import com.freedomfighter.readers.speech.share.Cpu
import com.freedomfighter.readers.speech.whisper.preferredThreads

/**
 * The JNI surface of the vendored llama.cpp. One model loaded at a time, one answer at a time
 * (the context is not thread-safe), and nothing to configure: the model is only ever asked for
 * a list of points, so there is no sampling setting to get wrong.
 */
object LlamaLib {
    init { Cpu.load("llama") }

    @JvmStatic external fun initContext(modelPath: String, threads: Int, nCtx: Int, keepInRam: Boolean): Long
    @JvmStatic external fun freeContext(ptr: Long)
    /** One prompt, one answer; "" when cancelled, when the prompt does not fit, or on failure. */
    @JvmStatic external fun generate(ptr: Long, prompt: String, maxTokens: Int): String
    @JvmStatic external fun cancel()
    @JvmStatic external fun progress(): Int
    /** Why the last call gave nothing, in one short sentence, or "" when nothing went wrong. */
    @JvmStatic external fun lastError(): String
}

/**
 * A model loaded once and used for every piece of a long transcript, then closed.
 * Failures are returned, never thrown: a recording must never lose its transcript because
 * the summary went wrong.
 */
/**
 * [keepInRam] pins the weights instead of leaving them file-backed. It is the difference between
 * a model that answers and one that crawls: measured on a Pixel on 2026-09-18, a four-billion
 * model left file-backed produced under a word a second, because two and a half gigabytes were
 * read back from storage for every token. Only ask for it when the phone has the room to spare.
 */
class LlamaSession(
    modelPath: String,
    private val nCtx: Int = CONTEXT,
    keepInRam: Boolean = false,
) : AutoCloseable {
    private val ptr = LlamaLib.initContext(modelPath, preferredThreads(), nCtx, keepInRam)

    val loaded: Boolean get() = ptr != 0L

    /** The answer, or null when cancelled or impossible. [onProgress] gets 0–100 while it runs. */
    fun run(prompt: String, maxTokens: Int = 512, onProgress: (Int) -> Unit = {}): String? {
        if (ptr == 0L) return null
        val done = java.util.concurrent.atomic.AtomicBoolean(false)
        val poll = Thread {
            while (!done.get()) {
                onProgress(LlamaLib.progress())
                Thread.sleep(300)
            }
        }.apply { isDaemon = true; start() }
        return try {
            LlamaLib.generate(ptr, prompt, maxTokens).takeIf { it.isNotBlank() }
        } finally {
            done.set(true)
            runCatching { poll.join(500) }
        }
    }

    /** Stops the answer being written; the session stays usable for the next piece. */
    fun cancel() = LlamaLib.cancel()

    /** Why the last answer was empty, for the log — never shown to anyone as it is. */
    fun why(): String = runCatching { LlamaLib.lastError() }.getOrDefault("")

    override fun close() {
        if (ptr != 0L) LlamaLib.freeContext(ptr)
    }

    companion object {
        /** Enough for a piece of transcript and its answer, small enough to leave the phone alone. */
        const val CONTEXT = 4096
    }
}
