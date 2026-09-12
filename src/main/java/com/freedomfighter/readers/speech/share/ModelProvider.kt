package com.freedomfighter.readers.speech.share

import android.content.ContentProvider
import android.content.ContentValues
import android.content.pm.PackageManager
import android.database.Cursor
import android.database.MatrixCursor
import android.net.Uri
import android.os.ParcelFileDescriptor
import java.io.File
import java.io.FileNotFoundException

/**
 * Read-only window on this app's `files/models`, for the sibling app signed with the same key.
 * `content://<package>.models/<file>` opens the model; a query lists what is here with sizes.
 * Nothing is ever written through it, and a half-downloaded `.part` is never handed out.
 */
class ModelProvider : ContentProvider() {
    override fun onCreate(): Boolean = true

    private fun allowed(): Boolean {
        val ctx = context ?: return false
        val caller = callingPackage ?: return false
        return ctx.packageManager.checkSignatures(caller, ctx.packageName) == PackageManager.SIGNATURE_MATCH
    }

    override fun query(uri: Uri, projection: Array<String>?, selection: String?, args: Array<String>?, sort: String?): Cursor? {
        if (!allowed()) return null
        val c = MatrixCursor(arrayOf("name", "size"))
        ModelFiles.dir(context!!).listFiles()
            ?.filter { it.isFile && !it.name.endsWith(".part") }
            ?.forEach { c.addRow(arrayOf<Any>(it.name, it.length())) }
        return c
    }

    override fun openFile(uri: Uri, mode: String): ParcelFileDescriptor {
        if (!allowed() || mode != "r") throw FileNotFoundException()
        val name = uri.lastPathSegment?.takeIf { '/' !in it && !it.endsWith(".part") && it != "." && it != ".." }
            ?: throw FileNotFoundException()
        val f = File(ModelFiles.dir(context!!), name)
        if (!f.isFile) throw FileNotFoundException(name)
        return ParcelFileDescriptor.open(f, ParcelFileDescriptor.MODE_READ_ONLY)
    }

    override fun getType(uri: Uri): String = "application/octet-stream"
    override fun insert(uri: Uri, values: ContentValues?): Uri? = null
    override fun delete(uri: Uri, selection: String?, args: Array<String>?): Int = 0
    override fun update(uri: Uri, values: ContentValues?, selection: String?, args: Array<String>?): Int = 0
}
