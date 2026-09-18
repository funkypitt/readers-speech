package com.freedomfighter.readers.speech.share

import android.content.Context
import android.net.Uri
import android.os.ParcelFileDescriptor
import java.io.Closeable
import java.io.File

/**
 * A model file open for whisper or llama: our own copy on disk, or a sibling app's copy handed
 * over as a file descriptor. `/proc/self/fd/N` is a path both libraries read like any other
 * (llama maps it, whisper streams it), so nothing is copied and the two gigabytes exist once.
 * Keep it open for as long as the model is in use.
 */
class ModelHandle(val path: String, private val pfd: ParcelFileDescriptor?) : Closeable {
    /** True when the file belongs to the other app. */
    val shared: Boolean get() = pfd != null
    override fun close() { pfd?.close() }
}

/**
 * Where the models live, and how one app finds a model the other already downloaded.
 *
 * Both apps keep theirs in `files/models`; each exposes that folder read-only through
 * [ModelProvider] to apps signed with the same key. Before downloading, an app asks its sibling.
 */
object ModelFiles {
    /** The apps that share models, by package name. */
    val SIBLINGS = listOf(
        "com.freedomfighter.readersaudio",
        "com.freedomfighter.readersrecorder",
        "com.freedomfighter.readerspodcasts",
    )

    fun dir(ctx: Context): File = File(ctx.filesDir, "models").apply { mkdirs() }
    fun own(ctx: Context, name: String): File = File(dir(ctx), name)

    /** [name] readable here, or from a sibling; null when nobody has it. */
    fun open(ctx: Context, name: String, minBytes: Long): ModelHandle? {
        own(ctx, name).let { if (it.isFile && it.length() >= minBytes) return ModelHandle(it.absolutePath, null) }
        for (pkg in SIBLINGS) {
            if (pkg == ctx.packageName) continue
            val pfd = runCatching { ctx.contentResolver.openFileDescriptor(uri(pkg, name), "r") }.getOrNull() ?: continue
            if (pfd.statSize >= minBytes) return ModelHandle("/proc/self/fd/${pfd.fd}", pfd)
            pfd.close()
        }
        return null
    }

    /** Whether [name] can be read at all, here or from a sibling. Remembered a few seconds: the screens ask often. */
    fun available(ctx: Context, name: String, minBytes: Long): Boolean = holder(ctx, name, minBytes) != null

    /** The package that holds [name]: ours, a sibling's, or null. */
    fun holder(ctx: Context, name: String, minBytes: Long): String? {
        own(ctx, name).let { if (it.isFile && it.length() >= minBytes) return ctx.packageName }
        val now = System.currentTimeMillis()
        synchronized(cache) {
            cache[name]?.let { (at, who) -> if (now - at < CACHE_MS) return who }
        }
        val who = SIBLINGS.firstOrNull { pkg ->
            pkg != ctx.packageName && runCatching {
                ctx.contentResolver.openFileDescriptor(uri(pkg, name), "r")?.use { it.statSize >= minBytes } == true
            }.getOrDefault(false)
        }
        synchronized(cache) { cache[name] = now to who }
        return who
    }

    /** Forget what the siblings answered: after a download, or a deletion. */
    fun refresh() = synchronized(cache) { cache.clear() }

    private fun uri(pkg: String, name: String): Uri = Uri.parse("content://$pkg.models/$name")
    private val cache = HashMap<String, Pair<Long, String?>>()
    private const val CACHE_MS = 5_000L
}
