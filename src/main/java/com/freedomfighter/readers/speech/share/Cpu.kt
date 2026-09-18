package com.freedomfighter.readers.speech.share

import android.os.Build
import java.io.File

/**
 * Which build of whisper or llama this processor deserves.
 *
 * Above fp16 there are two instructions that decide the speed of a quantised model: the int8
 * dot product and the int8 matrix multiply. A phone that has them and is given the plain build
 * answers a four-billion model at under a word a second; measured on a Pixel on 2026-09-18, a
 * forty-minute talk was going to take an hour and a quarter to translate.
 *
 * The names are tried in order and the first that loads wins, so a library that was not built,
 * or a processor that lied, costs nothing but a fallback.
 */
object Cpu {
    private val features: String by lazy {
        runCatching { File("/proc/cpuinfo").readText() }.getOrDefault("")
    }

    private val arm64: Boolean get() = Build.SUPPORTED_ABIS.firstOrNull() == "arm64-v8a"

    /** The variants of [base], best first. */
    fun variants(base: String): List<String> = when {
        !arm64 -> listOf(base)
        features.contains("asimddp") && features.contains("i8mm") && features.contains("fphp") ->
            listOf("${base}_v8_2_dp_i8mm", "${base}_v8fp16_va", base)
        features.contains("fphp") -> listOf("${base}_v8fp16_va", base)
        else -> listOf(base)
    }

    /** Loads the best build of [base] that this phone can run. */
    fun load(base: String) {
        val names = variants(base)
        names.forEachIndexed { i, name ->
            val ok = runCatching { System.loadLibrary(name) }.isSuccess
            if (ok) return
            if (i == names.lastIndex) System.loadLibrary(base)
        }
    }

    /** What was chosen, for the log and for a settings line: "dotprod+i8mm", "fp16", "baseline". */
    fun flavour(): String = when (variants("x").first()) {
        "x_v8_2_dp_i8mm" -> "dotprod+i8mm"
        "x_v8fp16_va" -> "fp16"
        else -> "baseline"
    }
}
