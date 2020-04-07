package io.github.yizhiru.toynlp.ner;

public final class Entity {
    private int startIndex;
    private int endIndex;

    public String word;
    public String type;

    public Entity(int startIndex, int endIndex, String word, String type) {
        this.startIndex = startIndex;
        this.endIndex = endIndex;
        this.word = word;
        this.type = type;
    }


    public String toString() {
        return word;
    }

}
