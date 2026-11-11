package com.freedomfighter.readers.speech.whisper

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class LanguageTest {
    private val W = Language.WINDOW.toLong()

    @Test fun shortRecordingIsOneWindow() {
        assertEquals(listOf(0L), Language.candidates(W / 2))
        assertEquals(listOf(0L), Language.candidates(W))
    }

    @Test fun candidatesStayInsideAndSpread() {
        val total = 60 * 60 * 16_000L
        val c = Language.candidates(total)
        assertEquals(Language.CANDIDATES, c.size)
        assertTrue(c.all { it >= 0 && it + W <= total })
        assertTrue(c.first() > 0)                      // not the very start: jingles live there
        assertTrue(c.zipWithNext().all { (a, b) -> b > a })
        // two minutes: four windows, not nine overlapping ones
        assertEquals(4, Language.candidates(120 * 16_000L).size)
    }

    @Test fun musicIntroIsSkipped() {
        val shares = listOf(0L to 0.02f, 100L to 0.9f, 200L to 0.85f, 300L to 0.1f, 400L to 0.7f, 500L to 0.95f)
        assertEquals(listOf(100L, 200L, 500L), Language.choose(shares))
    }

    @Test fun noSpeechNoPassage() {
        assertEquals(emptyList<Long>(), Language.choose(listOf(0L to 0.0f, 1L to 0.01f)))
        assertEquals(listOf(1L), Language.choose(listOf(0L to 0.0f, 1L to 0.2f)))
    }

    @Test fun withoutDetectorPassagesAreSpread() {
        val shares = (0 until 9).map { it.toLong() to null }
        assertEquals(listOf(1L, 4L, 7L), Language.choose(shares))
    }

    @Test fun averageOutvotesOneWrongPassage() {
        val h = Language.combine(listOf(
            mapOf("en" to 0.8f, "fr" to 0.2f),       // the English jingle
            mapOf("fr" to 0.95f, "en" to 0.05f),
            mapOf("fr" to 0.9f, "en" to 0.1f),
        ))!!
        assertEquals("fr", h.language)
        assertEquals(3, h.passages)
        assertNull(Language.combine(listOf(emptyMap())))
    }

    @Test fun suggestOnlyWhenPlainlyWrong() {
        val english = Language.Heard("en", 0.92f, mapOf("en" to 0.92f, "fr" to 0.03f), 3)
        assertEquals("en", Language.suggest("fr", english))
        assertNull(Language.suggest("en", english))
        assertNull(Language.suggest("", english))
        // a French speaker's English: detection unsure, no question
        val accent = Language.Heard("en", 0.6f, mapOf("en" to 0.6f, "fr" to 0.35f), 3)
        assertNull(Language.suggest("fr", accent))
        // sure, but the chosen language is still plausible (two languages mixed)
        val mixed = Language.Heard("en", 0.75f, mapOf("en" to 0.75f, "fr" to 0.2f), 3)
        assertNull(Language.suggest("fr", mixed))
    }
}
