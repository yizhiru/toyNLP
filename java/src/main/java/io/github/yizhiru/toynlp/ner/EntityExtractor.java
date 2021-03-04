package io.github.yizhiru.toynlp.ner;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * 提取实体词，仅支持BIOES 标注体系
 */
public final class EntityExtractor {

    /**
     * 用于解析标注类别，一个类别可以解析成BIOES标签、实体词类型
     */
    private static class Chunk {
        // BIOES标签
        public char tag;

        // 实体词类型
        public String type;

        public Chunk(char tag, String type) {
            this.tag = tag;
            this.type = type;
        }
    }

    /**
     * 根据模型输出的已标注类别，解析句子中的实体词
     *
     * @param chars  字符串类型的句子
     * @param labels 模型输出的已标注的标签
     * @return 句子中所有实体词
     */
    public static List<Entity> extractEntities(char[] chars, List<String> labels) {
        int length = Math.min(chars.length, labels.size());
        List<Entity> entities = new ArrayList<>();
        for (int i = 0; i < length; ) {
            Chunk chunk = parseLabelToChunk(labels.get(i));

            // 单独成实体词
            if (chunk.tag == 'S') {
                entities.add(new Entity(i,
                        i,
                        String.valueOf(chars[i]),
                        chunk.type));
                i++;
            }
            // 连续成实体词
            else if (chunk.tag == 'B') {
                // 尾字
                if (i == length - 1) {
                    break;
                }
                for (int j = i + 1; j < length; j++) {
                    Chunk another = parseLabelToChunk(labels.get(j));
                    if (another.tag == 'I' && another.type.equals(chunk.type) && j < length - 1) {
                        continue;
                    } else if (another.tag == 'E' && another.type.equals(chunk.type)) {
                        entities.add(new Entity(i,
                                j,
                                String.valueOf(Arrays.copyOfRange(chars, i, j + 1)),
                                chunk.type));
                        i = j + 1;
                        break;
                    } else if (another.tag == 'B' || another.tag == 'S') {
                        i = j;
                        break;
                    } else {
                        i = j + 1;
                        break;
                    }
                }
            } else {
                i++;
            }
        }
        return entities;
    }

    /**
     * 解析NER 标注类别为Chunk类，比如：B_PROD解析成Chunk(B, PROD)，O解析成Chunk(O,)
     */
    private static Chunk parseLabelToChunk(String label) {
        char tag = label.charAt(0);
        String type;
        if (label.equals("O")) type = "";
        else type = label.substring(2);
        return new Chunk(tag, type);
    }


    /**
     * 检查当前chunk是否为一个实体的结束
     *
     * @param previousTag  前一标签
     * @param tag          当前标签
     * @param previousType 前一实体类型
     * @param type         当前实体类型
     * @return 布尔值，当前chunk为一个实体的结束，则为true；反之则为false
     */
    private static boolean isEndOfChunk(char previousTag, char tag, String previousType, String type) {
        if (tag == 'S') {
            return true;
        }
        if (tag == 'E') {
            if ((previousTag == 'B' || previousTag == 'I') && previousType.equals(type)) {
                return true;
            }
        }
        return false;
    }

}
