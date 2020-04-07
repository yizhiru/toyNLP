package io.github.yizhiru.toynlp.util;

public final class ArrayMathUtils {

    /**
     * @return the index of the max value; if max is a tie, returns the first one.
     */
    public static int argmax(float[] a) {
        float max = Float.NEGATIVE_INFINITY;
        int argmax = 0;
        for (int i = 0; i < a.length; i++) {
            if (a[i] > max) {
                max = a[i];
                argmax = i;
            }
        }
        return argmax;
    }
    
    public static int[][] argmax(float[][][] a) {
        int len1 = a.length;
        int len2 = a[0].length;
        int[][] argmaxArray = new int[len1][len2];
        for (int i = 0; i < len1; i++) {
            for (int j = 0; j < len2; j++) {
                argmaxArray[i][j] = argmax(a[i][j]);
            }
        }
        return argmaxArray;
    }
}
