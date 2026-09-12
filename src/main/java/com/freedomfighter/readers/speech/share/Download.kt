package com.freedomfighter.readers.speech.share

import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL

/**
 * One resumable download, for every model: what an interrupted attempt already brought down is
 * kept in `<target>.part` and asked for again with a Range header, because a download of
 * hundreds of megabytes over a phone connection gets cut, and one that always restarts from zero
 * is one that never finishes. Only a complete file is renamed into place.
 */
object Download {
    fun fetch(url: String, target: File, onProgress: (Int) -> Unit, cancelled: () -> Boolean) {
        val tmp = File(target.parentFile, target.name + ".part")
        var have = if (tmp.exists()) tmp.length() else 0L
        val c = URL(url).openConnection() as HttpURLConnection
        c.instanceFollowRedirects = true; c.connectTimeout = 20_000; c.readTimeout = 60_000
        c.setRequestProperty("User-Agent", "readers-speech")
        if (have > 0) c.setRequestProperty("Range", "bytes=$have-")
        if (c.responseCode >= 400) throw IllegalStateException("download: HTTP ${c.responseCode}")
        // 206: the server picks up where we stopped. Anything else means it sent the whole file
        // again, so what is already on disk is worthless and the part starts over.
        val resuming = c.responseCode == 206
        if (!resuming) have = 0L
        val total = c.contentLengthLong.let { if (it > 0) it + have else -1L }
        var done = have
        c.inputStream.use { i ->
            FileOutputStream(tmp, resuming).use { o ->
                val buf = ByteArray(512 * 1024); var last = -1
                while (true) {
                    if (cancelled()) return
                    val n = i.read(buf); if (n < 0) break
                    o.write(buf, 0, n); done += n
                    val pct = if (total > 0) (done * 100 / total).toInt() else 0
                    if (pct != last) { last = pct; onProgress(pct) }
                }
            }
        }
        if (total > 0 && done != total) throw IllegalStateException("download cut short")
        if (!tmp.renameTo(target)) { target.delete(); tmp.renameTo(target) }
    }

    /** Bytes an interrupted attempt already brought down. */
    fun partBytes(target: File): Long = File(target.parentFile, target.name + ".part").let { if (it.exists()) it.length() else 0L }
}
