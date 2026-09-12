package com.freedomfighter.readers.speech.whisper

import android.content.Context
import com.freedomfighter.readers.speech.share.Download
import com.freedomfighter.readers.speech.share.ModelFiles
import com.freedomfighter.readers.speech.share.ModelHandle
import kotlinx.coroutines.flow.MutableStateFlow
import java.io.File

/** A ggml Whisper model from whisper.cpp's Hugging Face repository (quantised, multilingual). */
data class Model(val key: String, val file: String, val mb: Int) {
    val url: String get() = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/$file"
}

/**
 * Two qualities. Normal is Whisper small: quick, decent punctuation. High is large-v3-turbo:
 * clearly better punctuation and accuracy, but its encoder is as heavy as large-v3's, so it is
 * much slower on a phone. Shared between the two apps like the summary model.
 */
object Models {
    val NORMAL = Model("normal", "ggml-small-q5_1.bin", 190)
    val HIGH = Model("high", "ggml-large-v3-turbo-q5_0.bin", 574)
    val ALL = listOf(NORMAL, HIGH)
    const val DEFAULT = "normal"
    private const val MIN_BYTES = 1_000_000L

    /** Older settings said base / small / medium: medium means high, the rest normal. */
    fun byKey(key: String?): Model = if (key == "high" || key == "medium") HIGH else NORMAL
    fun dir(ctx: Context): File = ModelFiles.dir(ctx)
    fun file(ctx: Context, m: Model): File = ModelFiles.own(ctx, m.file)

    /** Whether [m] can be read, from this app's files or the sibling's. */
    fun isDownloaded(ctx: Context, m: Model): Boolean = ModelFiles.available(ctx, m.file, MIN_BYTES)

    /** [m] open for whisper, from wherever it is; null when nobody has it. */
    fun open(ctx: Context, m: Model): ModelHandle? = ModelFiles.open(ctx, m.file, MIN_BYTES)

    /** 0–100 while a download runs, −1 otherwise. */
    val downloading = MutableStateFlow(-1)

    /** Fetch [m] into this app's files, carrying on from an interrupted attempt. */
    fun download(ctx: Context, m: Model, onProgress: (Int) -> Unit = {}, cancelled: () -> Boolean = { false }) {
        if (isDownloaded(ctx, m)) return
        downloading.value = 0
        try {
            Download.fetch(m.url, file(ctx, m), { downloading.value = it; onProgress(it) }, cancelled)
        } finally { downloading.value = -1; ModelFiles.refresh() }
    }

    /** Remove model files neither quality uses any more (the old base and medium). */
    fun cleanup(ctx: Context) {
        val keep = ALL.map { it.file }.toSet()
        dir(ctx).listFiles()?.forEach { if (it.name.endsWith(".bin") && it.name !in keep) it.delete() }
    }
}
