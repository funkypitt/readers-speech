package com.freedomfighter.readers.speech.translate

import android.app.ActivityManager
import android.content.Context
import com.freedomfighter.readers.speech.share.Download
import com.freedomfighter.readers.speech.share.ModelFiles
import com.freedomfighter.readers.speech.share.ModelHandle
import kotlinx.coroutines.flow.MutableStateFlow
import java.io.File

/**
 * The model that translates, fetched once and shared between the apps like the other two.
 *
 * Gemma 3 4B, and not something smaller: measured on 2026-09-18 over five passages in four
 * languages, a 3-billion model copied the source out instead of translating it in one German
 * passage out of five — a failure that reads as a translation and is not one. Gemma is 3.6
 * chrF++ ahead of the best 4-billion alternative (p = 0.0001), which is why it is the only
 * choice offered here.
 *
 * Its weights come from `ggml-org`, not from Google's own repository, which is gated and cannot
 * be fetched without an account; that mirror also answers Range requests, so an interrupted
 * download carries on where it stopped.
 */
object TranslateModel {
    const val FILE = "gemma-3-4b-it-Q4_K_M.gguf"
    const val URL_STR = "https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF/resolve/main/$FILE"
    const val MB = 2490
    private const val NEEDED_BYTES = 2_600_000_000L
    private const val MIN_BYTES = 2_000_000_000L

    /**
     * Eight gigabytes of phone, which is what its 2.9 GB footprint needs in practice. Written as
     * 6.5 and not 8: a phone sold as "8 GB" reports about 7.4, and a check against 8e9 would
     * refuse the very phones the measurement was made on.
     */
    private const val MIN_TOTAL_RAM = 6_500_000_000L
    private const val MIN_FREE_RAM = 1_200_000_000L

    fun file(ctx: Context): File = ModelFiles.own(ctx, FILE)

    fun isDownloaded(ctx: Context): Boolean = ModelFiles.available(ctx, FILE, MIN_BYTES)

    /** The sibling package that holds the model when this app does not, else null. */
    fun sharedFrom(ctx: Context): String? = ModelFiles.holder(ctx, FILE, MIN_BYTES)?.takeIf { it != ctx.packageName }

    /** The model open for llama, from wherever it is; null when nobody has it. */
    fun open(ctx: Context): ModelHandle? = ModelFiles.open(ctx, FILE, MIN_BYTES)

    private fun memory(ctx: Context): ActivityManager.MemoryInfo =
        ActivityManager.MemoryInfo().also {
            (ctx.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager).getMemoryInfo(it)
        }

    /** Whether this phone can hold it at all. Checked before the option is even offered. */
    fun phoneCanHoldIt(ctx: Context): Boolean {
        val am = ctx.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        return !am.isLowRamDevice && memory(ctx).totalMem >= MIN_TOTAL_RAM
    }

    fun phoneMemoryGb(ctx: Context): Int = ((memory(ctx).totalMem + 500_000_000L) / 1_000_000_000L).toInt()

    fun roomRightNow(ctx: Context): Boolean = memory(ctx).let { !it.lowMemory && it.availMem >= MIN_FREE_RAM }

    /** 0–100 while the download runs, −1 otherwise. */
    val downloading = MutableStateFlow(-1)

    fun remove(ctx: Context): Boolean = file(ctx).delete().also { ModelFiles.refresh() }

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

    fun partPercent(ctx: Context): Int =
        (Download.partBytes(file(ctx)) * 100 / (MB * 1_000_000L)).toInt().coerceIn(0, 99)
}
