package io.github.yizhiru.toynlp.ner;

/**
 * 实体词类
 */
public final class Entity {
    // 实体词在句子中开始索引值
    private final int startIndex;

    // 实体词在句子中结束索引值
    private final int endIndex;

    // 实体词
    public final String word;

    // 实体词类型
    public final String type;

    public Entity(int startIndex, int endIndex, String word, String type) {
        this.startIndex = startIndex;
        this.endIndex = endIndex;
        this.word = word;
        this.type = type;
    }


    public String toString() {
        return word;
    }

    public int getStartIndex() {
        return startIndex;
    }

    public int getEndIndex() {
        return endIndex;
    }


}
