package io.github.yizhiru.toynlp.common

import io.github.yizhiru.toynlp.util.IDIOM_BIN_PATH
import io.github.yizhiru.toynlp.util.IDIOM_DICT_PATH
import org.junit.jupiter.api.Assertions.*
import java.nio.file.Files
import java.nio.file.Paths
import java.util.stream.Collectors

internal class DoubleArrayTrieTest {

    @org.junit.jupiter.api.Test
    fun isWordMatched() {
        val dat = DoubleArrayTrie.loadDat(IDIOM_BIN_PATH)
//        assertTrue(dat.isPrefixMatched("郑虔三"))
//        assertTrue(dat.isWordMatched("郑虔三绝"))
        assertTrue(dat.isWordMatched("动人幽意"))
//        assertTrue(dat.isWordMatched("齐齐哈尔"))
//        assertTrue(dat.isWordMatched("名古屋"))
//        assertTrue(dat.isWordMatched("克拉约瓦"))
//        assertTrue(dat.isWordMatched("１０月９日街"))
//        assertTrue(dat.isWordMatched("鸡公？"))
//        assertTrue(dat.isWordMatched("齐白石纪念馆"))
//        assertTrue(dat.isWordMatched("龙格伦吉里"))
//        assertTrue(dat.isWordMatched("特德本－圣玛丽"))
//        assertFalse(dat.isWordMatched("首乌"))
    }

    @org.junit.jupiter.api.Test
    fun serialize() {
        val expect = DoubleArrayTrieMaker.make(IDIOM_DICT_PATH)
        expect.serialize(IDIOM_BIN_PATH)
        val actual = DoubleArrayTrie.loadDat(IDIOM_BIN_PATH)

        assertEquals(expect.size(), actual.size())
        for (j in 0 until expect.size()) {
            assertEquals(expect.getBaseByIndex(j), actual.getBaseByIndex(j))
            assertEquals(expect.getCheckByIndex(j), actual.getCheckByIndex(j))
        }
    }

    @org.junit.jupiter.api.Test
    fun restore() {
        val dat = DoubleArrayTrie.loadDat(IDIOM_BIN_PATH)
        val actualLexicon = dat.restoreLexicon()
                .stream()
                .collect(Collectors.toSet())
        val expectedLexicon = Files.lines(Paths.get(IDIOM_DICT_PATH))
                .map { line -> line.trim() }
                .collect(Collectors.toSet())
        for (word in expectedLexicon) {
            assertTrue(actualLexicon.contains(word))
        }
        assertEquals(expectedLexicon.size, actualLexicon.size)
    }
}