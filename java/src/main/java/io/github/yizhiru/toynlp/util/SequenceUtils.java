package io.github.yizhiru.toynlp.util;

public final class SequenceUtils {

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
