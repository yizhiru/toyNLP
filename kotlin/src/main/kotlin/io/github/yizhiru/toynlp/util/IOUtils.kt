package io.github.yizhiru.toynlp.util

import java.io.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.Charset
import java.util.ArrayList

/**
 * Represents the end-of-file (or stream).
 */
private const val EOF = -1

/**
 * Gets the contents of an `InputStream` as a `byte[]`.
 *
 *
 * This method buffers the input internally, so there is no need to use a
 * `BufferedInputStream`.
 *
 * @return the requested byte array
 * @throws NullPointerException if the input is null
 * @throws IOException          if an I/O error occurs
 */
@Throws(NullPointerException::class, IOException::class)
fun InputStream.toByteArray(): ByteArray {
    ByteArrayOutputStream().use { output ->
        val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
        var n = this.read(buffer)
        while (EOF != n) {
            output.write(buffer, 0, n)
            n = this.read(buffer)
        }
        return output.toByteArray()
    }
}

/**
 * Gets the contents of an `InputStream` as a int array.
 *
 *
 * This method buffers the input internally, so there is no need to use a
 * `BufferedInputStream`.
 *
 * @return the requested int array
 * @throws NullPointerException if the input is null
 * @throws IOException          if an I/O error occurs
 */
@Throws(NullPointerException::class, IOException::class)
fun InputStream.toIntArray(): IntArray {
    val bytes = this.toByteArray()
    val intBuffer = ByteBuffer.wrap(bytes)
            .order(ByteOrder.LITTLE_ENDIAN)
            .asIntBuffer()
    val array = IntArray(intBuffer.remaining())
    intBuffer.get(array)
    return array
}

/**
 * Gets the contents of an <code>InputStream</code> as a list of Strings,
 * one entry per line.
 *
 * @return the list of Strings, never null
 * @throws NullPointerException if the input is null
 * @throws IOException          if an I/O error occurs
 */
@Throws(NullPointerException::class, IOException::class)
fun InputStream.readLines(charset: Charset = Charsets.UTF_8): List<String> {
    val reader = this.bufferedReader(charset)
    val list = ArrayList<String>()
    var line: String? = reader.readLine()
    while (line != null) {
        list.add(line)
        line = reader.readLine()
    }
    reader.close()
    return list
}