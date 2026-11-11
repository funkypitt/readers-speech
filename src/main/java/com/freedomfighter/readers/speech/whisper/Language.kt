package com.freedomfighter.readers.speech.whisper

/**
 * Which language is spoken, found before transcribing — and whether the language chosen by hand
 * is plainly not it.
 *
 * Whisper left to itself guesses from the first thirty seconds of each piece. That guess fails
 * where recordings often begin: music, a jingle, a short greeting, an accent. And when it fails,
 * whisper does not write nonsense: it writes the wrong language fluently — a French talk comes
 * back as a neat English translation, which nobody notices until reading it. So the guess is made
 * here over several passages spread through the recording, each one chosen because Silero hears
 * speech in it, and the probabilities are averaged. The language found is then told to whisper
 * for every piece, so all pieces agree and the style sentence can be used.
 *
 * The same passages check a language chosen by hand: when the recording plainly speaks another
 * one, the app asks before spending minutes transcribing in the wrong language.
 */
object Language {
    /** Thirty seconds at 16 kHz: what whisper looks at for one guess. */
    const val WINDOW = 30 * 16_000

    /** Passages whose guesses are averaged. */
    const val PASSAGES = 3

    /** Places looked at to find those passages (Silero is quick; whisper's encoder is not). */
    const val CANDIDATES = 9

    /** A passage counts as speech above this share (Silero's frames). */
    const val SPEECH = 0.4f

    /** For the check: what detection must be sure of, and how unlikely the chosen language must be. */
    const val SURE = 0.7f
    const val UNLIKELY = 0.1f

    /** What was heard: the language, its averaged probability, all averaged probabilities, passages used. */
    data class Heard(val language: String, val confidence: Float, val probs: Map<String, Float>, val passages: Int) {
        fun prob(code: String): Float = probs[code] ?: 0f
    }

    /**
     * Where to look, in samples, for a recording of [total] samples: [CANDIDATES] windows spread
     * evenly, the first and last kept clear of the very edges (jingles live there). A recording
     * shorter than a window is one window.
     */
    fun candidates(total: Long, count: Int = CANDIDATES): List<Long> {
        if (total <= WINDOW) return listOf(0L)
        val room = total - WINDOW
        val n = count.coerceAtMost(((total / WINDOW).toInt()).coerceAtLeast(1)).coerceAtLeast(1)
        return (0 until n).map { i -> (room * (2 * i + 1) / (2L * n)) }.distinct()
    }

    /**
     * From each candidate's share of speech (null when Silero could not be had), the passages to
     * ask whisper about: the most spoken ones, up to [PASSAGES]. With no detector, passages spread
     * over the recording. None when nothing in the recording sounds like speech.
     */
    fun choose(shares: List<Pair<Long, Float?>>): List<Long> {
        if (shares.isEmpty()) return emptyList()
        if (shares.any { it.second == null }) {
            val step = (shares.size.toDouble() / PASSAGES).coerceAtLeast(1.0)
            return (0 until minOf(PASSAGES, shares.size)).map { shares[(it * step + step / 2).toInt().coerceAtMost(shares.size - 1)].first }.distinct()
        }
        val ranked = shares.sortedByDescending { it.second!! }
        val spoken = ranked.filter { it.second!! >= SPEECH }
        if (spoken.isNotEmpty()) return spoken.take(PASSAGES).map { it.first }.sorted()
        // Little speech anywhere: the best place, if there is any speech in it at all.
        return ranked.first().takeIf { it.second!! > 0.05f }?.let { listOf(it.first) } ?: emptyList()
    }

    /** The average of each language's probability over the passages; null when there were none. */
    fun combine(guesses: List<Map<String, Float>>): Heard? {
        val usable = guesses.filter { it.isNotEmpty() }
        if (usable.isEmpty()) return null
        val sum = HashMap<String, Float>()
        for (g in usable) for ((k, v) in g) sum[k] = (sum[k] ?: 0f) + v
        val avg = sum.mapValues { it.value / usable.size }
        val best = avg.maxBy { it.value }
        return Heard(best.key, best.value, avg, usable.size)
    }

    /**
     * Whether the language chosen by hand is plainly not the one spoken: the language to suggest
     * instead, or null. Plainly means detection is sure ([SURE]) and gives the chosen one almost
     * no chance ([UNLIKELY]) — an accent, or two languages mixed, must not raise the question.
     */
    fun suggest(chosen: String, heard: Heard?): String? {
        if (heard == null || chosen.isBlank() || heard.language == chosen) return null
        return heard.language.takeIf { heard.confidence >= SURE && heard.prob(chosen) < UNLIKELY }
    }

    /**
     * Detection over a whole recording of [total] samples at 16 kHz, read through [read]
     * (offset, count → samples). Null when no speech was heard or whisper could not tell.
     * [onStep] is told 0–100.
     */
    fun detect(session: WhisperSession, total: Long, read: (Long, Int) -> FloatArray, onStep: (Int) -> Unit = {}): Heard? {
        val places = candidates(total)
        val shares = places.mapIndexed { i, at ->
            onStep(i * 30 / places.size)
            at to session.speechShare(read(at, minOf(WINDOW.toLong(), total - at).toInt()))
        }
        val chosen = choose(shares)
        val guesses = chosen.mapIndexed { i, at ->
            onStep(30 + i * 70 / chosen.size.coerceAtLeast(1))
            session.languageProbs(read(at, minOf(WINDOW.toLong(), total - at).toInt()))
        }
        onStep(100)
        return combine(guesses)
    }

    /** The same over samples already in memory. */
    fun detect(session: WhisperSession, pcm16k: FloatArray, onStep: (Int) -> Unit = {}): Heard? =
        detect(session, pcm16k.size.toLong(), { at, n -> pcm16k.copyOfRange(at.toInt(), (at + n).toInt()) }, onStep)
}
