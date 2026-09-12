package com.freedomfighter.readers.speech.summary

import android.util.Log

/**
 * The main points of a transcript, written on the phone by a small model.
 *
 * What a 3-billion model can and cannot do was measured before this was written, and again on
 * 2026-09-12 on three talks of an hour (two in English, one in French), with the very weights the
 * phone runs. It understands a talk well, but cannot follow a composed instruction — asked for
 * "a summary then bullet points" it echoed the instruction and dumped its own notes — so every
 * step below asks for ONE thing. And it cannot hold a whole talk in view: the first version of
 * this file took bullet points from every piece and asked the model to merge seventy of them,
 * and the ten it returned all came from the first ten minutes, every time, on every talk. Bullets
 * lose the thread; a paragraph keeps it.
 *
 * So the transcript is read in pieces, each piece is turned into a short paragraph of what is
 * said (not "the speaker says", the ideas themselves), the paragraphs — in order — give the theme
 * of the whole talk in two sentences, and then, with the theme in front of it, the model writes
 * the points of the parts a few at a time, in order — asked for all at once it went back to
 * the opening on the phone. A talk too long for its paragraphs to fit the context has them
 * folded two into one for the theme only. The result is the theme, then the points, and it
 * covers the talk instead of its opening.
 */
object Summariser {
    /** Words per piece. A piece plus its instruction must sit well inside the context. */
    private const val WORDS_PER_PIECE = 900
    /** Words of notes the points step may be given; past that the notes are folded first. */
    private const val NOTES_BUDGET = 1_500
    private const val MAX_POINTS = 12
    /** Points of a recording read whole, in one piece: fewer than of an hour's talk. */
    private const val POINTS_SHORT = 8
    /** Notes per points call. Each call sees the theme and a few parts, and covers them by construction. */
    private const val NOTES_PER_CALL = 4
    /** Below this, a recording is a note to self, not something to summarise. */
    private const val MIN_WORDS = 120
    private const val TAG = "ReadersLlama"

    private val PART = mapOf(
        "en" to "Here is part %1\$d of %2\$d of the transcript of a talk. Note what is said in this part, in one paragraph of at most 120 words, as plain statements of the ideas: not \"the speaker says\", just the ideas themselves, and what they lead to. Write nothing else.\n\nTRANSCRIPT:\n%3\$s",
        "fr" to "Voici la partie %1\$d sur %2\$d de la transcription d'une causerie. Note ce qui y est dit, en un paragraphe de 120 mots au plus, sous forme d'affirmations simples : pas « l'orateur dit », seulement les idées elles-mêmes et où elles mènent. N'écris rien d'autre.\n\nTRANSCRIPTION :\n%3\$s",
        "de" to "Hier ist Teil %1\$d von %2\$d der Abschrift eines Vortrags. Halte fest, was in diesem Teil gesagt wird, in einem Absatz von höchstens 120 Wörtern, als schlichte Aussagen: nicht „der Redner sagt“, nur die Gedanken selbst und wohin sie führen. Schreibe sonst nichts.\n\nABSCHRIFT:\n%3\$s",
        "es" to "Esta es la parte %1\$d de %2\$d de la transcripción de una charla. Anota lo que se dice en esta parte, en un párrafo de 120 palabras como máximo, como afirmaciones simples: no «el orador dice», solo las ideas mismas y adónde llevan. No escribas nada más.\n\nTRANSCRIPCIÓN:\n%3\$s",
        "pt" to "Esta é a parte %1\$d de %2\$d da transcrição de uma palestra. Anota o que é dito nesta parte, num parágrafo de 120 palavras no máximo, como afirmações simples: não «o orador diz», apenas as ideias em si e aonde levam. Não escrevas mais nada.\n\nTRANSCRIÇÃO:\n%3\$s",
        "ru" to "Вот часть %1\$d из %2\$d расшифровки беседы. Запиши, что в ней говорится, одним абзацем не более 120 слов, в виде простых утверждений: не «докладчик говорит», а сами мысли и к чему они ведут. Больше ничего не пиши.\n\nРАСШИФРОВКА:\n%3\$s",
    )
    private val GROUP = mapOf(
        "en" to "Here are, in order, notes on two consecutive parts of a talk. Merge them into one paragraph of at most 160 words that keeps the thread of the argument. Write nothing else.\n\nNOTES:\n%s",
        "fr" to "Voici, dans l'ordre, des notes sur deux parties consécutives d'une causerie. Fusionne-les en un paragraphe de 160 mots au plus qui garde le fil du raisonnement. N'écris rien d'autre.\n\nNOTES :\n%s",
        "de" to "Hier sind, der Reihe nach, Notizen zu zwei aufeinanderfolgenden Teilen eines Vortrags. Fasse sie zu einem Absatz von höchstens 160 Wörtern zusammen, der den Faden der Argumentation behält. Schreibe sonst nichts.\n\nNOTIZEN:\n%s",
        "es" to "Estas son, en orden, notas sobre dos partes consecutivas de una charla. Fúndelas en un párrafo de 160 palabras como máximo que conserve el hilo del razonamiento. No escribas nada más.\n\nNOTAS:\n%s",
        "pt" to "Estas são, por ordem, notas sobre duas partes consecutivas de uma palestra. Funde-as num parágrafo de 160 palavras no máximo que mantenha o fio do raciocínio. Não escrevas mais nada.\n\nNOTAS:\n%s",
        "ru" to "Вот по порядку заметки о двух следующих друг за другом частях беседы. Объедини их в один абзац не более 160 слов, сохранив ход рассуждения. Больше ничего не пиши.\n\nЗАМЕТКИ:\n%s",
    )
    private val THEME = mapOf(
        "en" to "Here are, in order, notes on the parts of a talk. In two sentences, say what the talk is about and what the speaker wants the listener to understand. Write nothing else.\n\nNOTES:\n%s",
        "fr" to "Voici, dans l'ordre, des notes sur les parties d'une causerie. En deux phrases, dis de quoi parle la causerie et ce que l'orateur veut faire comprendre. N'écris rien d'autre.\n\nNOTES :\n%s",
        "de" to "Hier sind, der Reihe nach, Notizen zu den Teilen eines Vortrags. Sage in zwei Sätzen, worum es in dem Vortrag geht und was der Redner dem Zuhörer verständlich machen will. Schreibe sonst nichts.\n\nNOTIZEN:\n%s",
        "es" to "Estas son, en orden, notas sobre las partes de una charla. En dos frases, di de qué trata la charla y qué quiere el orador que el oyente entienda. No escribas nada más.\n\nNOTAS:\n%s",
        "pt" to "Estas são, por ordem, notas sobre as partes de uma palestra. Em duas frases, diz de que trata a palestra e o que o orador quer que o ouvinte compreenda. Não escrevas mais nada.\n\nNOTAS:\n%s",
        "ru" to "Вот по порядку заметки о частях беседы. В двух предложениях скажи, о чём беседа и что докладчик хочет донести до слушателя. Больше ничего не пиши.\n\nЗАМЕТКИ:\n%s",
    )
    private val POINTS = mapOf(
        "en" to "Here is what a talk is about:\n%1\$s\n\nAnd here are, in order, notes on parts %2\$s of that talk:\n%3\$s\n\nWrite the %4\$d main points of these parts, in order, one per line, each line starting with \"- \". Each point states an idea as a plain sentence, as the speaker would put it — never \"the talk discusses\" or \"the speaker emphasizes\". Write nothing else.",
        "fr" to "Voici de quoi parle une causerie :\n%1\$s\n\nEt voici, dans l'ordre, des notes sur les parties %2\$s de cette causerie :\n%3\$s\n\nÉcris les %4\$d points principaux de ces parties, dans l'ordre, un par ligne, chaque ligne commençant par « - ». Chaque point énonce une idée en une phrase complète, comme l'orateur la dirait — jamais « la causerie aborde » ni « l'orateur insiste ». N'écris rien d'autre.",
        "de" to "Darum geht es in einem Vortrag:\n%1\$s\n\nUnd hier sind, der Reihe nach, Notizen zu den Teilen %2\$s dieses Vortrags:\n%3\$s\n\nSchreibe die %4\$d wichtigsten Punkte dieser Teile, der Reihe nach, einen pro Zeile, jede Zeile beginnt mit „- “. Jeder Punkt formuliert einen Gedanken als ganzen Satz, so wie der Redner ihn sagen würde — nie „der Vortrag behandelt“ oder „der Redner betont“. Schreibe sonst nichts.",
        "es" to "De esto trata una charla:\n%1\$s\n\nY estas son, en orden, notas sobre las partes %2\$s de esa charla:\n%3\$s\n\nEscribe los %4\$d puntos principales de estas partes, en orden, uno por línea, cada línea empezando por «- ». Cada punto enuncia una idea como una frase completa, tal como la diría el orador — nunca «la charla trata» ni «el orador insiste». No escribas nada más.",
        "pt" to "É disto que trata uma palestra:\n%1\$s\n\nE estas são, por ordem, notas sobre as partes %2\$s dessa palestra:\n%3\$s\n\nEscreve os %4\$d pontos principais destas partes, por ordem, um por linha, cada linha a começar por «- ». Cada ponto enuncia uma ideia numa frase completa, como o orador a diria — nunca «a palestra aborda» nem «o orador sublinha». Não escrevas mais nada.",
        "ru" to "Вот о чём беседа:\n%1\$s\n\nА вот по порядку заметки о частях %2\$s этой беседы:\n%3\$s\n\nНапиши %4\$d основных мыслей этих частей, по порядку, по одной в строке, каждая строка начинается с «- ». Каждая мысль — законченное предложение, как сказал бы сам докладчик, никогда «в беседе говорится» или «докладчик подчёркивает». Больше ничего не пиши.",
    )
    /** "1 to 4" in each language, for the parts a points call is given. */
    private val TO = mapOf("en" to "%d to %d", "fr" to "%d à %d", "de" to "%d bis %d", "es" to "%d a %d", "pt" to "%d a %d", "ru" to "%d–%d")
    /** The word before a note's number, in the notes handed to the theme and points steps. */
    private val PART_WORD = mapOf("en" to "Part", "fr" to "Partie", "de" to "Teil", "es" to "Parte", "pt" to "Parte", "ru" to "Часть")

    /**
     * A point that narrates the talk instead of stating an idea. The instruction forbids it and
     * the model writes one anyway now and then, usually first; it is dropped here, cheaply.
     */
    private val META = Regex(
        "^(the talk|the speaker|this talk|this part|la causerie|le discours|l'orateur|l'oratrice|la conférence|" +
        "der vortrag|der redner|die rednerin|la charla|el orador|la oradora|a palestra|o orador|a oradora|доклад|лектор|оратор)\\b",
        RegexOption.IGNORE_CASE
    )

    /** The instruction follows the language of the recording, and falls back to English. */
    private fun tongue(language: String): String = language.lowercase().take(2).takeIf { it in PART } ?: "en"

    /** Whole paragraphs, grouped into pieces of at most [WORDS_PER_PIECE] words. */
    fun pieces(text: String): List<String> {
        val out = mutableListOf<String>()
        val cur = StringBuilder()
        var words = 0
        for (para in text.split("\n\n").filter { it.isNotBlank() }) {
            val n = para.split(Regex("\\s+")).size
            if (words > 0 && words + n > WORDS_PER_PIECE) { out += cur.toString().trim(); cur.clear(); words = 0 }
            cur.append(para).append("\n\n"); words += n
        }
        if (cur.isNotBlank()) out += cur.toString().trim()
        return out
    }

    /**
     * Keeps only what was asked for. A small model sometimes adds a preamble or repeats itself;
     * rather than trusting it, the answer is filtered here — deterministic, and cheap.
     */
    fun keepPoints(raw: String, limit: Int = MAX_POINTS): List<String> =
        raw.lines()
            .map { it.trim() }
            .filter { it.startsWith("-") || it.startsWith("•") || it.startsWith("*") }
            .map { it.removePrefix("-").removePrefix("•").removePrefix("*").trim().trim('*') }
            .filter { it.length > 3 && !META.containsMatchIn(it) }
            .distinctBy { it.lowercase() }
            .take(limit)

    private fun words(s: String) = s.split(Regex("\\s+")).count { it.isNotBlank() }

    /** The notes with their part numbers, [from] onwards, as the theme and points steps read them. */
    private fun labelled(notes: List<String>, from: Int, partWord: String): String =
        notes.mapIndexed { i, n -> "$partWord ${from + i}: $n" }.joinToString("\n\n")

    /** A note without the preamble a small model sometimes opens with ("Here are the ideas of this part:"). */
    private fun unprefaced(answer: String): String {
        val t = answer.trim()
        val first = t.lineSequence().first().trim()
        return if (first.endsWith(":") && first.length < 120) t.removePrefix(first).trim() else t
    }

    /**
     * The theme and the points of [transcript] — the two sentences, a blank line, then one point
     * per line — or null when cancelled, when the model cannot be loaded, or when nothing usable
     * came back. Never throws: a recording keeps its transcript whatever happens here.
     */
    fun summarise(
        modelPath: String,
        transcript: String,
        language: String,
        onProgress: (Int) -> Unit = {},
        cancelled: () -> Boolean = { false },
    ): String? = runCatching {
        val lang = tongue(language)
        if (words(transcript) < MIN_WORDS) return null
        val parts = pieces(transcript)
        val partWord = PART_WORD[lang]!!

        LlamaSession(modelPath).use { session ->
            if (!session.loaded) { Log.e(TAG, "no model: ${session.why()}"); return null }
            // Progress over the calls to come: one per part, then the theme and the points.
            var total = (if (parts.size > 1) parts.size else 0) + 2
            var done = 0
            fun step(it: Int) = onProgress((100 * (done + it / 100f) / total).toInt().coerceIn(0, 99))

            // One piece: nothing to fold, the transcript itself is the note.
            var notes: List<String> = if (parts.size == 1) listOf(transcript.trim()) else {
                val out = mutableListOf<String>()
                parts.forEachIndexed { i, part ->
                    if (cancelled()) { session.cancel(); return null }
                    val answer = session.run(PART[lang]!!.format(i + 1, parts.size, part), maxTokens = 256, onProgress = ::step)
                    done++
                    if (answer == null) { Log.e(TAG, "part ${i + 1}/${parts.size}: ${session.why()}"); return@forEachIndexed }
                    out += unprefaced(answer)
                    Log.i(TAG, "note ${i + 1}/${parts.size}, ${words(out.last())} words: ${out.last().take(160).replace('\n', ' ')}")
                }
                out
            }
            if (notes.isEmpty()) return null

            // The theme needs every note at once. Too many to fit: two into one, until they fit
            // — two at a time, not three; measured on a two-hour recording, three into one lost its
            // second half. The points below are given the notes as written, never the folded ones.
            var forTheme = notes
            while (forTheme.sumOf { words(it) } > NOTES_BUDGET && forTheme.size > 2) {
                val groups = forTheme.chunked(2)
                total += groups.size
                forTheme = groups.map { g ->
                    if (cancelled()) { session.cancel(); return null }
                    val a = session.run(GROUP[lang]!!.format(g.joinToString("\n\n")), maxTokens = 320, onProgress = ::step)
                    done++
                    a?.trim() ?: g.joinToString(" ").take(900)
                }
            }
            Log.i(TAG, "${notes.size} notes, ${notes.sumOf { words(it) }} words (${forTheme.size} for the theme)")
            if (cancelled()) { session.cancel(); return null }
            val theme = session.run(THEME[lang]!!.format(labelled(forTheme, 1, partWord)), maxTokens = 160, onProgress = ::step)?.trim().orEmpty()
            done++
            Log.i(TAG, "theme: ${theme.take(200).replace('\n', ' ')}")

            // The points, a few parts at a time with the theme in front: the first version asked
            // for the points of the whole talk in one call, and the model — on the phone, not on
            // the workstation — kept giving twelve points on the first half. Asked group by group,
            // every part is covered by construction, in order, and the theme keeps them one talk.
            val groups = notes.chunked(NOTES_PER_CALL)
            val wanted = if (groups.size == 1) POINTS_SHORT else MAX_POINTS
            val perGroup = maxOf(3, (wanted + groups.size - 1) / groups.size)
            total += groups.size - 1
            val points = mutableListOf<String>()
            groups.forEachIndexed { g, group ->
                if (cancelled()) { session.cancel(); return null }
                val first = g * NOTES_PER_CALL + 1
                val last = first + group.size - 1
                val range = if (first == last) "$first" else TO[lang]!!.format(first, last)
                val raw = session.run(POINTS[lang]!!.format(theme, range, labelled(group, first, partWord), perGroup), maxTokens = 320, onProgress = ::step)
                done++
                if (raw == null) { Log.e(TAG, "points ${first}–$last: ${session.why()}"); return@forEachIndexed }
                points += keepPoints(raw, perGroup)
            }
            val kept = points.distinctBy { it.lowercase() }
            if (kept.isEmpty()) return null
            listOf(theme, kept.joinToString("\n") { "- $it" }).filter { it.isNotBlank() }.joinToString("\n\n")
        }
    }.getOrElse {
        Log.e(TAG, "summary failed: ${it.message}")
        null
    }
}
