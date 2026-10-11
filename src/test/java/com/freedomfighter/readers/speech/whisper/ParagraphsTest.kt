package com.freedomfighter.readers.speech.whisper

import org.junit.Assert.assertEquals
import org.junit.Test

/** The paragraph rule, the desktop's: voices, long pauses, shorter pauses after a sentence, long paragraphs. */
class ParagraphsTest {
    private fun seg(a: Long, b: Long, t: String, voice: Boolean = false) = Segment(a, b, t, voice)
    private fun starts(vararg s: Segment) = Paragraphs.starts(s.toList()).toList()

    @Test fun shortPausesInsideASentenceDoNotCut() {
        assertEquals(listOf(true, false, false), starts(seg(0, 1000, "Today we talk"), seg(1500, 2500, "about sleep,"), seg(2700, 4000, "and more.")))
    }

    @Test fun aLongPauseCutsEvenMidSentence() {
        assertEquals(listOf(true, true), starts(seg(0, 1000, "and then"), seg(3100, 4000, "nothing happened.")))
    }

    @Test fun aShorterPauseCutsOnlyAfterASentence() {
        assertEquals(listOf(true, true), starts(seg(0, 1000, "That is all."), seg(2000, 3000, "Next question.")))
        assertEquals(listOf(true, false), starts(seg(0, 1000, "That is all"), seg(2000, 3000, "I had to say.")))
        assertEquals(listOf(true, false), starts(seg(0, 1000, "That is all."), seg(1800, 3000, "Next question.")))
    }

    @Test fun anotherVoiceAlwaysOpensAParagraph() {
        assertEquals(listOf(true, true, false), starts(seg(0, 2000, "What do you think?"), seg(2100, 4000, "Well, I think it works.", voice = true), seg(4300, 5000, "Mostly.")))
    }

    @Test fun aLongParagraphBreaksAtASentenceEndWithABreath() {
        val long = "x".repeat(450) + "."
        assertEquals(listOf(true, false, true, false), starts(seg(0, 1000, long), seg(1100, 2000, long), seg(2300, 3000, "New one."), seg(3100, 4000, "More.")))
    }

    @Test fun buildJoinsLinesAndDropsRepeats() {
        val text = Paragraphs.build(listOf(seg(0, 1000, "Hello there."), seg(1100, 2000, "Hello there."), seg(2000, 3000, "Yes.", voice = true)), "—")
        assertEquals("Hello there.\n\nYes.\n", text)
        assertEquals("—\n", Paragraphs.build(emptyList(), "—"))
    }
}
