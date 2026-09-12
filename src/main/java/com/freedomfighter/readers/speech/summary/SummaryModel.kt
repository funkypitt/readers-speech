package com.freedomfighter.readers.speech.summary

import android.app.ActivityManager
import android.content.Context
import com.freedomfighter.readers.speech.share.Download
import com.freedomfighter.readers.speech.share.ModelFiles
import com.freedomfighter.readers.speech.share.ModelHandle
import kotlinx.coroutines.flow.MutableStateFlow
import java.io.File

/**
 * The language model that writes the summary, fetched once and shared between the two apps.
 *
 * Only one choice is offered, on purpose. Measured on 2026-09-12 on the same interview: a
 * 1.5-billion model misses the subject entirely — it turned the psychiatrist's critique of
 * prescribing into a description of it, and promoted the closing thanks to a main point — while
 * a 3-billion model got the thesis, the argument about the industry, and the lasting side
 * effects. An 8-billion one would be three times slower and hold five gigabytes, which a phone
 * cannot spare. Offering a model that produces a plausible but wrong summary would be worse
 * than offering none, so this is the only size on the menu.
 */
object SummaryModel {
    const val FILE = "qwen2.5-3b-instruct-q4_k_m.gguf"
    const val URL_STR = "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/$FILE"
    const val MB = 2105                     // 2 104 932 768 bytes, as the server reports it
    private const val NEEDED_BYTES = 2_200_000_000L
    private const val MIN_BYTES = 1_500_000_000L

    /**
     * A phone small enough that loading two gigabytes of weights would get the application killed
     * mid-answer is refused the option outright, rather than offered a feature that cannot work.
     * Measured on the emulator: three gigabytes of system memory is not enough — the model loads,
     * then the process dies and the queue starts again.
     */
    private const val MIN_TOTAL_RAM = 5_000_000_000L    // a "6 GB" phone reports about 5.6
    private const val MIN_FREE_RAM = 900_000_000L       // right before loading

    fun file(ctx: Context): File = ModelFiles.own(ctx, FILE)

    /** Whether the model can be read, from this app's files or the sibling's. */
    fun isDownloaded(ctx: Context): Boolean = ModelFiles.available(ctx, FILE, MIN_BYTES)

    /** The sibling package that holds the model when this app does not, else null. */
    fun sharedFrom(ctx: Context): String? = ModelFiles.holder(ctx, FILE, MIN_BYTES)?.takeIf { it != ctx.packageName }

    /** The model open for llama, from wherever it is; null when nobody has it. */
    fun open(ctx: Context): ModelHandle? = ModelFiles.open(ctx, FILE, MIN_BYTES)

    private fun memory(ctx: Context): ActivityManager.MemoryInfo =
        ActivityManager.MemoryInfo().also {
            (ctx.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager).getMemoryInfo(it)
        }

    /** Whether this phone can hold the model at all. Checked before the setting is even offered. */
    fun phoneCanHoldIt(ctx: Context): Boolean {
        val am = ctx.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        return !am.isLowRamDevice && memory(ctx).totalMem >= MIN_TOTAL_RAM
    }

    /** Total system memory in whole gigabytes, for the sentence that explains a refusal. */
    fun phoneMemoryGb(ctx: Context): Int = ((memory(ctx).totalMem + 500_000_000L) / 1_000_000_000L).toInt()

    /** Whether there is room right now: another application may have taken it since. */
    fun roomRightNow(ctx: Context): Boolean = memory(ctx).let { !it.lowMemory && it.availMem >= MIN_FREE_RAM }

    /** 0–100 while the download runs, −1 otherwise. */
    val downloading = MutableStateFlow(-1)

    /** Delete our copy to get the space back; the setting then simply offers to fetch it again. */
    fun remove(ctx: Context): Boolean = file(ctx).delete().also { ModelFiles.refresh() }

    /** Fetch the model into this app's files, carrying on from an interrupted attempt. */
    fun download(ctx: Context, onProgress: (Int) -> Unit = {}, cancelled: () -> Boolean = { false }) {
        if (isDownloaded(ctx)) return
        val target = file(ctx)
        val have = Download.partBytes(target)
        if (ModelFiles.dir(ctx).usableSpace < NEEDED_BYTES - have)
            throw IllegalStateException("not enough space: ${(NEEDED_BYTES - have) / 1_000_000} MB needed")
        downloading.value = 0
        try {
            Download.fetch(URL_STR, target, { downloading.value = it; onProgress(it) }, cancelled)
        } finally { downloading.value = -1; ModelFiles.refresh() }
    }

    /** How much of the model an interrupted attempt already brought down, 0 when there is none. */
    fun partPercent(ctx: Context): Int = (Download.partBytes(file(ctx)) * 100 / (MB * 1_000_000L)).toInt().coerceIn(0, 99)
}
