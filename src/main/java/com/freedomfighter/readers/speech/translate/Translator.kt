package com.freedomfighter.readers.speech.translate

import android.util.Log
import com.freedomfighter.readers.speech.summary.LlamaSession
import com.freedomfighter.readers.speech.whisper.Segment

/**
 * A transcript translated on the phone, in blocks that stay tied to the audio.
 *
 * Not sentence by sentence: measured on 2026-09-18, matching a translated sentence back to the
 * source sentence only holds three times out of four, so a translation aligned that way would
 * drift against the sound and put the wrong line under the finger. A block of some forty
 * seconds is translated as one passage and keeps that passage's start and end, which is an
 * alignment the reader can trust.
 *
 * The instruction names the target language twice and ends on a primer in that language: of the
 * shapes measured, that is the only one the model never answered in the wrong language.
 */
object Translator {
    private const val TAG = "ReadersTranslate"

    /** Characters of source text per block. Small enough to keep the model on the passage. */
    private const val BLOCK_CHARS = 700
    private const val MAX_TOKENS = 640

    data class Block(val startMs: Long, val endMs: Long, val text: String)

    /**
     * [segments] rendered into [target]. Returns null if it was cancelled; a block whose
     * translation could not be had keeps the source text, so the reading never has a hole.
     */
    fun translate(
        modelPath: String,
        segments: List<Segment>,
        target: String,
        onProgress: (Int) -> Unit = {},
        cancelled: () -> Boolean = { false },
        /** Pin the weights: a whole talk is hundreds of answers, not one. */
        keepInRam: Boolean = true,
    ): List<Block>? {
        val blocks = group(segments)
        if (blocks.isEmpty()) return emptyList()
        LlamaSession(modelPath, keepInRam = keepInRam).use { session ->
            if (!session.loaded) throw IllegalStateException(session.why().ifBlank { "llama: model not loaded" })
            val out = ArrayList<Block>(blocks.size)
            blocks.forEachIndexed { i, block ->
                if (cancelled()) return null
                onProgress(i * 100 / blocks.size)
                val done = once(session, block.text, target)
                    ?: once(session, block.text, target, strict = true)
                out.add(block.copy(text = (done ?: block.text).trim()))
            }
            onProgress(100)
            return out
        }
    }

    /** One passage. Returns null when the answer is missing, empty, or plainly not a translation. */
    private fun once(session: LlamaSession, text: String, target: String, strict: Boolean = false): String? {
        val template = (if (strict) STRICT else ASK)[target] ?: (if (strict) STRICT else ASK)["en"]!!
        val answer = session.run(template.format(text), MAX_TOKENS)?.trim()
        if (answer.isNullOrBlank()) {
            Log.w(TAG, "empty answer: " + session.why())
            return null
        }
        val clean = strip(answer)
        // A model that hands the source back has not translated it — the failure the choice of
        // model was made to avoid, and worth catching all the same.
        if (clean.equals(text.trim(), ignoreCase = true)) return null
        // A length wildly out of proportion means it answered something else entirely.
        val ratio = clean.length.toDouble() / text.length.coerceAtLeast(1)
        if (ratio < 0.4 || ratio > 2.5) {
            Log.w(TAG, "length out of proportion ($ratio)")
            return null
        }
        return clean
    }

    /** Models like to announce themselves; the primer they were given comes back too. */
    fun strip(answer: String): String {
        var s = answer.trim()
        PRIMERS.forEach { p -> if (s.startsWith(p, ignoreCase = true)) s = s.removeRange(0, p.length).trim() }
        s = s.removePrefix("\"").removeSuffix("\"")
        return s.trim()
    }

    /** Consecutive segments gathered into passages of about [BLOCK_CHARS]. */
    fun group(segments: List<Segment>): List<Block> {
        val out = ArrayList<Block>()
        var start = -1L
        var end = 0L
        val text = StringBuilder()
        for (s in segments) {
            val piece = s.text.trim()
            if (piece.isEmpty()) continue
            if (start < 0) start = s.startMs
            if (text.isNotEmpty()) text.append(' ')
            text.append(piece)
            end = s.endMs
            if (text.length >= BLOCK_CHARS) {
                out.add(Block(start, end, text.toString()))
                text.setLength(0); start = -1L
            }
        }
        if (text.isNotEmpty()) out.add(Block(start.coerceAtLeast(0), end, text.toString()))
        return out
    }

    private val PRIMERS = listOf(
        "TRADUCTION EN FRANÇAIS :", "TRADUCTION EN FRANÇAIS:", "TRANSLATION INTO ENGLISH:",
        "ÜBERSETZUNG AUF DEUTSCH:", "TRADUCCIÓN AL ESPAÑOL:", "TRADUÇÃO PARA PORTUGUÊS:",
        "ПЕРЕВОД НА РУССКИЙ:",
    )

    private val ASK = mapOf(
        "en" to "Translate the passage below into English. Give the English translation only: no comment, no note, no source text.\n\nPASSAGE:\n%s\n\nTRANSLATION INTO ENGLISH:",
        "fr" to "Traduis en français le passage ci-dessous. Rends uniquement la traduction française : pas de commentaire, pas de note, pas le texte d'origine.\n\nPASSAGE :\n%s\n\nTRADUCTION EN FRANÇAIS :",
        "de" to "Übersetze den folgenden Abschnitt auf Deutsch. Gib nur die deutsche Übersetzung: kein Kommentar, keine Anmerkung, nicht den Ausgangstext.\n\nABSCHNITT:\n%s\n\nÜBERSETZUNG AUF DEUTSCH:",
        "es" to "Traduce al español el pasaje siguiente. Da solo la traducción española: sin comentario, sin nota, sin el texto de origen.\n\nPASAJE:\n%s\n\nTRADUCCIÓN AL ESPAÑOL:",
        "pt" to "Traduz para português o trecho abaixo. Dá apenas a tradução portuguesa: sem comentário, sem nota, sem o texto de origem.\n\nTRECHO:\n%s\n\nTRADUÇÃO PARA PORTUGUÊS:",
        "ru" to "Переведи следующий отрывок на русский язык. Дай только русский перевод: без комментариев, без примечаний, без исходного текста.\n\nОТРЫВОК:\n%s\n\nПЕРЕВОД НА РУССКИЙ:",
    )

    /** The second attempt, for a passage the first one fumbled: the same, said harder. */
    private val STRICT = mapOf(
        "en" to "You are translating into English. Write the English of the passage below, sentence for sentence, and nothing else — no preamble, no explanation, not one word of the original.\n\nPASSAGE:\n%s\n\nTRANSLATION INTO ENGLISH:",
        "fr" to "Tu traduis en français. Écris le français du passage ci-dessous, phrase après phrase, et rien d'autre — pas de préambule, pas d'explication, pas un mot de l'original.\n\nPASSAGE :\n%s\n\nTRADUCTION EN FRANÇAIS :",
        "de" to "Du übersetzt auf Deutsch. Schreibe das Deutsche des folgenden Abschnitts, Satz für Satz, und sonst nichts — keine Vorrede, keine Erklärung, kein Wort des Originals.\n\nABSCHNITT:\n%s\n\nÜBERSETZUNG AUF DEUTSCH:",
        "es" to "Estás traduciendo al español. Escribe el español del pasaje siguiente, frase por frase, y nada más: sin preámbulo, sin explicación, ni una palabra del original.\n\nPASAJE:\n%s\n\nTRADUCCIÓN AL ESPAÑOL:",
        "pt" to "Estás a traduzir para português. Escreve o português do trecho abaixo, frase a frase, e nada mais: sem preâmbulo, sem explicação, nem uma palavra do original.\n\nTRECHO:\n%s\n\nTRADUÇÃO PARA PORTUGUÊS:",
        "ru" to "Ты переводишь на русский язык. Напиши русский текст отрывка ниже, предложение за предложением, и больше ничего — без предисловий, без объяснений, ни слова из оригинала.\n\nОТРЫВОК:\n%s\n\nПЕРЕВОД НА РУССКИЙ:",
    )
}
