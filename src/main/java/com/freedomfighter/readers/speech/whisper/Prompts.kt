package com.freedomfighter.readers.speech.whisper

/**
 * Whisper imitates the style of the text it is given as a prompt: a short, well-punctuated
 * sentence in the language spoken nudges it towards full sentences, commas and capitals.
 * The sentences say nothing about any subject, so they bias no vocabulary.
 */
object Prompts {
    private val STYLE = mapOf(
        "en" to "Hello. This is a careful transcript, with full sentences, commas and full stops.",
        "fr" to "Bonjour. Voici une transcription soignée, avec des phrases complètes, des virgules et des points.",
        "de" to "Guten Tag. Dies ist eine sorgfältige Abschrift, mit ganzen Sätzen, Kommas und Punkten.",
        "es" to "Hola. Esta es una transcripción cuidada, con frases completas, comas y puntos.",
        "pt" to "Olá. Esta é uma transcrição cuidada, com frases completas, vírgulas e pontos finais.",
        "ru" to "Здравствуйте. Это аккуратная расшифровка, с полными предложениями, запятыми и точками.",
        "it" to "Buongiorno. Questa è una trascrizione accurata, con frasi complete, virgole e punti."
    )

    /** Languages offered by hand: the phone's, English, detected (""), then the Reader's languages. */
    fun choices(device: String): List<String> = listOf(device, "en", "", "fr", "de", "es", "pt", "ru", "it").distinct()

    /** The style sentence; none when the language is left to detection. */
    fun style(language: String?): String = language?.let { STYLE[it] } ?: ""

    /**
     * For a piece after the first: the style sentence, then the end of what came before, so the
     * context runs on — unless that end is whisper repeating itself, in which case handing it
     * over would carry the stutter into the next five minutes. Then the style sentence alone.
     */
    fun forPiece(language: String?, previous: String): String {
        val tail = if (previous.length <= 240) previous else previous.takeLast(240).substringAfter(' ')
        return listOf(style(language), if (stutters(tail)) "" else tail.trim()).filter { it.isNotBlank() }.joinToString(" ")
    }

    /** Whether a text ends by saying the same sentence twice. */
    fun stutters(text: String): Boolean {
        val sentences = text.split(Regex("(?<=[.!?…])\\s+")).map { it.trim().lowercase() }.filter { it.length > 8 }
        if (sentences.size < 2) return false
        val last = sentences.last()
        return sentences.dropLast(1).any { it == last }
    }
}

/**
 * Paragraphs from timed segments, the same rule as the desktop apps so that a transcript reads
 * the same wherever it was made: a new paragraph where another voice starts (a question, then
 * its answer), after a pause of two seconds, after a shorter pause (0.9 s) once a sentence has
 * ended, and, one voice going on, at the end of a sentence once the paragraph has grown past
 * about 900 characters. A runaway sentence is cut at about 1,200 characters all the same.
 */
object Paragraphs {
    const val LONG_PAUSE_MS = 2_000L
    const val SENTENCE_PAUSE_MS = 900L
    const val LONG_PARAGRAPH = 900
    const val BREATH_MS = 250L
    private const val RUNAWAY = 1_200

    /** For each segment, whether a paragraph begins with it (the first always does). */
    fun starts(segments: List<Segment>): BooleanArray {
        val out = BooleanArray(segments.size)
        var lastEnd = -1L
        var length = 0
        var ended = false
        segments.forEachIndexed { i, s ->
            val gap = if (lastEnd < 0) 0L else s.startMs - lastEnd
            val start = i == 0 || s.newVoice || gap >= LONG_PAUSE_MS || (ended && gap >= SENTENCE_PAUSE_MS) ||
                (ended && length >= LONG_PARAGRAPH && gap >= BREATH_MS) || length >= RUNAWAY
            out[i] = start
            val text = s.text.trim()
            length = if (start) text.length else length + 1 + text.length
            if (text.isNotEmpty()) ended = endsSentence(text)
            lastEnd = s.endMs
        }
        return out
    }

    fun build(segments: List<Segment>, noSpeech: String): String {
        // whisper, once it repeats a sentence, may repeat it for minutes: once is enough
        var previous = ""
        val kept = segments.filter { s ->
            val key = s.text.trim().lowercase().replace(Regex("[^\\p{L}\\p{N}]+"), " ").trim()
            (key.isNotEmpty() && key != previous).also { if (it) previous = key }
        }
        val starts = starts(kept)
        val out = ArrayList<String>()
        val cur = StringBuilder()
        kept.forEachIndexed { i, s ->
            if (starts[i] && cur.isNotEmpty()) { out.add(cur.toString().trim()); cur.setLength(0) }
            if (cur.isNotEmpty()) cur.append(' ')
            cur.append(s.text.trim())
        }
        if (cur.isNotEmpty()) out.add(cur.toString().trim())
        return if (out.isEmpty()) noSpeech + "\n" else out.joinToString("\n\n") + "\n"
    }

    fun endsSentence(s: CharSequence): Boolean {
        var i = s.length - 1
        while (i >= 0 && (s[i].isWhitespace() || s[i] in "\"'»”’)")) i--
        return i >= 0 && s[i] in ".!?…"
    }
}
