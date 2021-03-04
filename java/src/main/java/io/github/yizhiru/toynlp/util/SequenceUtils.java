package io.github.yizhiru.toynlp.util;

public final class SequenceUtils {

    /**
     * Padding 1维数组，超过最长长度则截断，小于最长长度则padding
     *
     * @param seq          1维数组
     * @param maxLength    最长长度
     * @param paddingValue 用于padding的填充值
     * @return padding后数组
     */
    public static float[] pad1DSequence(float[] seq, int maxLength, float paddingValue) {
        float[] paddedSeq = new float[maxLength];
        int seqLength = seq.length;
        if (maxLength <= seqLength) {
            System.arraycopy(seq, 0, paddedSeq, 0, maxLength);
            return paddedSeq;
        }

        System.arraycopy(seq, 0, paddedSeq, 0, seqLength);
        for (int i = seqLength; i < maxLength; i++) {
            paddedSeq[i] = paddingValue;
        }

        return paddedSeq;

    }

    /**
     * Padding 2维数组，超过最长长度则截断，小于最长长度则padding
     *
     * @param seq          2维数组
     * @param maxLength    最长长度
     * @param paddingValue 用于padding的填充值
     * @return padding后数组
     */
    public static float[][] pad2DSequence(float[][] seq, int maxLength, float paddingValue) {
        int seqLength = seq.length;
        float[][] paddedSeq = new float[seqLength][maxLength];
        for (int i = 0; i < seqLength; i++) {
            float[] padded = new float[maxLength];
            float[] ids = seq[i];
            int length = ids.length;
            if (length <= maxLength) {
                System.arraycopy(ids, 0, padded, 0, length);
                for (int j = length; j < maxLength; j++) {
                    padded[j] = paddingValue;
                }
            } else {
                System.arraycopy(ids, 0, padded, 0, maxLength);
            }
            paddedSeq[i] = padded;
        }

        return paddedSeq;
    }

}
