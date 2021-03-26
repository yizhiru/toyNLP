package io.github.yizhiru.toynlp.common


import io.github.yizhiru.toynlp.util.readLines
import io.github.yizhiru.toynlp.util.toIntArray
import java.io.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.*


/**
 * 匹配失败标注
 */
internal const val MATCH_FAILURE_MARK = -1

/**
 * 基类
 */
abstract class BaseDoubleArrayTrie {

    /**
     * base 数组
     */
    abstract val baseArray: IntArray

    /**
     * check 数组
     */
    abstract val checkArray: IntArray

    /**
     * DAT的长度，等同于base数组的长度.
     */
    abstract val size: Int

    /**
     * 按照DAT的转移方程进行转移:
     * base(r) + c = s
     * check(s) = r
     */
    fun transition(fromIndex: Int, charValue: Int): Int {
        if (fromIndex < 0 || fromIndex >= size) {
            return MATCH_FAILURE_MARK
        }
        val index = baseArray[fromIndex] + charValue
        return if (index >= size || checkArray[index] != fromIndex) {
            MATCH_FAILURE_MARK
        } else index
    }

    /**
     * 匹配词 [word], 默认开始DAT索引 [startIndex]为0
     * 若匹配上，则为转移后DAT索引的负值；否则，则返回已匹配上的字符数
     * 其中，转移后DAT索引不为0. 若返回0，则未匹配上首字符.
     */
    protected fun match(word: String, startIndex: Int = 0): Int {
        var index = startIndex
        for (i in 0 until word.length) {
            index = transition(index, word[i].toInt())
            if (index == MATCH_FAILURE_MARK) {
                return i
            }
        }
        return -index
    }
}

/**
 * 双数组Trie (Double Array Trie, DAT).
 */
class DoubleArrayTrie(
        override val baseArray: IntArray,
        override val checkArray: IntArray
) : BaseDoubleArrayTrie(), Serializable {

    override val size: Int = baseArray.size

    init {
        if (baseArray.size != checkArray.size) {
            throw IllegalArgumentException("The length of base array ${baseArray.size} != the " +
                    "length of check array ${checkArray.size}")
        }
    }

    /**
     * 双数组Trie 长度
     */
    fun size(): Int {
        return size
    }

    /**
     * 确保索引没有越界.
     *
     * @param index 索引.
     */
    private fun ensureValidIndex(index: Int) {
        if (index >= size) {
            throw RuntimeException("The index $index is out of bound [$size].")
        }
    }

    /**
     * 根据索引 [index]得到base数组的值.
     */
    fun getBaseByIndex(index: Int): Int {
        ensureValidIndex(index)
        return baseArray[index]
    }

    /**
     * 根据索引 [index]得到check数组的值.
     */
    fun getCheckByIndex(index: Int): Int {
        ensureValidIndex(index)
        return checkArray[index]
    }

    /**
     * 序列化成二进制文件.
     */
    @Throws(IOException::class)
    fun serialize(path: String) {
        val channel = FileOutputStream(path).channel
        val byteBuffer = ByteBuffer.allocateDirect(4 * (2 * size + 1))
        val intBuffer = byteBuffer.order(ByteOrder.LITTLE_ENDIAN)
                .asIntBuffer()
        intBuffer.put(size)
        intBuffer.put(baseArray)
        intBuffer.put(checkArray)
        channel.write(byteBuffer)
        channel.close()
    }

    /**
     * 找到词 [word] 在词典中的索引
     */
    fun findLexiconIndex(word: String): Int {
        val matchedIndex = -match(word)
        if (matchedIndex <= 0) {
            return MATCH_FAILURE_MARK
        }
        val base = baseArray[matchedIndex]
        if (base >= size || checkArray[base] != matchedIndex) {
            return MATCH_FAILURE_MARK
        }
        return baseArray[base]
    }

    /**
     * 词[word]是否在DAT
     */
    fun isWordMatched(word: String): Boolean {
        return findLexiconIndex(word) >= 0
    }

    /**
     * 前缀[prefix]是否在trie中
     */
    fun isPrefixMatched(prefix: String): Boolean {
        return match(prefix) < 0
    }

    /**
     * 将DAT还原成词典.
     */
    fun restoreLexicon(): List<String> {
        val list = LinkedList<String>()
        for (i in 0 until size) {
            if (checkArray[i] >= 0) {
                val word = restoreWord(i)
                if (isWordMatched(word)) {
                    list.add(word)
                }
            }
        }
        return list
    }

    /**
     * 从词最后转移的索引 [lastDATIndex] 回溯，重建词
     */
    private fun restoreWord(lastDATIndex: Int): String {
        var index = lastDATIndex
        val sb = StringBuilder()
        while (index in 0 until size) {
            val priorIndex = checkArray[index]
            val priorBaseVal = baseArray[priorIndex]
            if (priorIndex == index || priorBaseVal >= index) {
                break
            }
            sb.insert(0, (index - priorBaseVal).toChar())
            index = priorIndex
        }
        return sb.toString()
    }

    companion object {

        /**
         * 加载序列化DAT模型
         *
         * @param path 文件目录
         * @return DAT模型
         */
        @Throws(IOException::class)
        fun loadDat(path: String): DoubleArrayTrie {
            return loadDat(FileInputStream(path))
        }

        /**
         * 加载序列化双数组Trie文件
         *
         * @param inputStream 文件输入流
         * @return 双数组Trie
         */
        fun loadDat(inputStream: InputStream): DoubleArrayTrie {
            val array = inputStream.toIntArray()
            val arrayLength = array[0]
            val baseArray = Arrays.copyOfRange(array, 1, arrayLength + 1)
            val checkArray = Arrays.copyOfRange(array, arrayLength + 1, 2 * arrayLength + 1)
            return DoubleArrayTrie(baseArray, checkArray)
        }
    }
}

/**
 * 双数组Trie构建
 */
class DoubleArrayTrieMaker : BaseDoubleArrayTrie() {

    /**
     * base数组初始值.
     */
    private val initialBaseValue = 0

    /**
     * check数组初始值.
     */
    private val initialCheckValue = -1

    override var baseArray: IntArray = intArrayOf(initialBaseValue, initialBaseValue)
    override var checkArray: IntArray = intArrayOf(initialCheckValue, initialCheckValue)
    override var size: Int = 2

    /**
     * 标记可用的索引.
     */
    private var availableIndexCursor: Int = 0

    /**
     * 双数组扩张两倍.
     */
    private fun expandTwo() {
        val oldCapacity = size
        val newCapacity = oldCapacity shl 1
        baseArray = Arrays.copyOf(baseArray, newCapacity)
        Arrays.fill(baseArray, oldCapacity, newCapacity, initialBaseValue)
        checkArray = Arrays.copyOf(checkArray, newCapacity)
        Arrays.fill(checkArray, oldCapacity, newCapacity, initialCheckValue)
        size = newCapacity
    }

    /**
     * 缩减操作，删掉尾部空闲的双数组
     */
    private fun shrink() {
        for (i in checkArray.indices.reversed()) {
            if (checkArray[i] == initialCheckValue) {
                size--
            } else {
                break
            }
        }
    }

    /**
     * 找到满足可填充后一字符集合 [nextChars]条件的索引
     */
    private fun findAvailableIndex(nextChars: IntArray): Int {
        var index = availableIndexCursor
        while (true) {
            if (nextChars.isNotEmpty()) {
                while (index + nextChars.last() >= size) {
                    expandTwo()
                }
            }
            // 所有应满足条件：
            // 1. 未被使用
            // 2. 满足所有后一字符跳转到的节点也未被使用
            if (checkArray[index] != initialCheckValue) {
                index++
                continue
            }
            var isValid = true
            for (c in nextChars) {
                if (checkArray[index + c] != initialCheckValue) {
                    isValid = false
                    break
                }
            }
            if (isValid) {
                return index
            }
            index++
        }
    }

    /**
     * 从前缀对应的双数组索引 [prefixDATIndex] 开始, 插入前缀的后一字符集合 [nextChars]
     */
    private fun insertNextChars(prefixDATIndex: Int, nextChars: IntArray, prefixIsWord: Boolean) {
        val index = findAvailableIndex(nextChars)
        baseArray[prefixDATIndex] = index
        if (prefixIsWord) {
            checkArray[index] = prefixDATIndex
            availableIndexCursor = index + 1
        }
        for (c in nextChars) {
            baseArray[index + c] = initialBaseValue
            checkArray[index + c] = prefixDATIndex
        }
    }

    /**
     * 基于按字典序排序后的词典 [sortedLexicon], 给定开始词典索引 [startLexiconIndex]
     * 以及前缀 [prefix]生成后一字符集合
     */
    private fun generateNextChars(sortedLexicon: Array<String>,
                                  startLexiconIndex: Int,
                                  prefix: String): IntArray {
        val list = LinkedList<Int>()
        val prefixLength = prefix.length
        for (i in startLexiconIndex until sortedLexicon.size) {
            val word = sortedLexicon[i]
            // 停止循环条件：
            // 1. 词的长度小于前缀长度
            // 2. 词的前缀与给定前缀不一致
            if (word.length < prefixLength || word.substring(0, prefixLength) != prefix) {
                break
            } else if (word.length > prefixLength) {
                val charValue = word[prefixLength].toInt()
                if (charValue != list.lastOrNull()) {
                    list.add(charValue)
                }
            }
        }
        return list.toIntArray()
    }

    companion object {

        /**
         * 根据词典文件构建双数组Trie.
         */
        fun make(lexicon: List<String>): DoubleArrayTrie {
            val maker = DoubleArrayTrieMaker()
            val sortedLexicon = lexicon.sortedWith(compareBy { it })
                    .toTypedArray()
            for (i in sortedLexicon.indices) {
                val word = sortedLexicon[i]
                val wordLength = word.length
                val matched = maker.match(word)
                val matchedLength = if (matched < 0) wordLength else matched
                for (j in matchedLength..wordLength) {
                    val prefix = word.substring(0, j)
                    val prefixDATIndex = -maker.match(prefix)
                    val nextChars = maker.generateNextChars(sortedLexicon, i, prefix)
                    maker.insertNextChars(prefixDATIndex, nextChars, j == wordLength)
                }
                val datIndex = -maker.match(word)
                maker.baseArray[maker.baseArray[datIndex]] = i
            }
            maker.shrink()
            return DoubleArrayTrie(Arrays.copyOf(maker.baseArray, maker.size),
                    Arrays.copyOf(maker.checkArray, maker.size))
        }

        /**
         * 根据词典文件构建双数组Trie.
         */
        @Throws(FileNotFoundException::class)
        fun make(path: String): DoubleArrayTrie {
            return make(FileInputStream(path))
        }

        /**
         * 根据词典文件构建双数组Trie.
         */
        @Throws(IOException::class)
        fun make(inputStream: InputStream): DoubleArrayTrie {
            val lexicon = inputStream.readLines()
                    .map { line -> line.trim() }
            return make(lexicon)
        }
    }
}
