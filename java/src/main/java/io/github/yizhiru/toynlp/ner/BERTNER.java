package io.github.yizhiru.toynlp.ner;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.yizhiru.toynlp.util.ArrayMathUtils;
import io.github.yizhiru.toynlp.util.SequenceUtils;
import org.apache.commons.collections.MapUtils;
import org.apache.commons.io.IOUtils;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;


/**
 * BERT NER模型，提取实体词
 */
public class BERTNER {

    // 序列标注类别对应编码值
    private Map<String, Integer> label2idx;

    // 编码值对应序列标注类别
    private Map<Integer, String> idx2Label;

    // BERT模型词典token 对应编码值
    private Map<String, Float> token2idx;

    private Session session;

    // BERT 模型的PAD 标识
    private final String PAD = "[PAD]";

    // BERT 模型的UNK 标识
    private final String UNK = "[UNK]";

    // BERT 模型的CLS 标识
    private final String CLS = "[CLS]";

    // BERT 模型的SEP 标识
    private final String SEP = "[SEP]";


    private final int SEQUENCE_LENGTH = 100;

    private BERTNER(Map<String, Integer> label2idx, Map<String, Float> token2idx, Session session) {
        this.label2idx = label2idx;
        this.idx2Label = MapUtils.invertMap(label2idx);
        this.token2idx = token2idx;
        this.session = session;
    }

    /**
     * 加载BERT NER模型
     */
    public static BERTNER load(String path) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        Map<String, Integer> label2idx = mapper.readValue(new FileInputStream(path + "/labels.json"),
                new TypeReference<Map<String, Integer>>() {
                });
        Map<String, Float> token2idx = mapper.readValue(new FileInputStream(path + "/words.json"),
                new TypeReference<Map<String, Float>>() {
                });

        byte[] pbBytes = IOUtils.toByteArray(new FileInputStream(path + "/model.pb"));
        Graph graph = new Graph();
        graph.importGraphDef(pbBytes);
        Session session = new Session(graph);
        return new BERTNER(label2idx, token2idx, session);
    }

    /**
     * 解析句子为BERT token对应编码值。
     * 将句子分解为字符，作为BERT输入token，查找编码值。
     * <p>
     * TODO: BERT token并非单一字符，也有由多个字符组成，官方tokenize为前缀最长匹配，后期待优化。
     *
     * @param textChars 输入句子
     * @return 解析后token编码值数组
     */
    private float[] tokenize(char[] textChars) {
        int length = Math.min(SEQUENCE_LENGTH, textChars.length + 2);
        float[] ids = new float[length];
        // add BOS
        ids[0] = token2idx.get(CLS);
        // add EOS
        ids[length - 1] = token2idx.get(SEP);
        float unkIndex = token2idx.get(UNK);
        for (int i = 1; i <= length - 2; i++) {
            ids[i] = MapUtils.getFloatValue(token2idx, String.valueOf(textChars[i - 1]), unkIndex);
        }
        return ids;
    }

    /**
     * 预测单个句子中的实体词
     */
    public Set<String> predict(String text) {
        String lowerText = text.toLowerCase();
        char[] textChars = lowerText.toCharArray();

        float[] tokenIds = tokenize(textChars);
        float[][] tokenArray = new float[1][tokenIds.length];
        tokenArray[0] = tokenIds;
        float[][] padded = SequenceUtils.pad2DSequence(tokenArray, SEQUENCE_LENGTH, this.token2idx.get(PAD));
        float[][] another = new float[1][SEQUENCE_LENGTH];

        try (Tensor<Float> x1 = Tensor.create(padded, Float.class);
             Tensor<Float> x2 = Tensor.create(another, Float.class)) {

            Tensor y = session.runner()
                    .feed("Input-Token", x1)
                    .feed("Input-Segment", x2)
                    .fetch("CRF/cond/Merge")
                    .run()
                    .get(0);
            float[][][] scoreMatrix = (float[][][]) y.copyTo(new float[1][SEQUENCE_LENGTH][label2idx.size()]);
            int[][] labelIds = ArrayMathUtils.argmax(scoreMatrix);

            int effectiveLength = Math.min(lowerText.length(), SEQUENCE_LENGTH - 2);
            List<Entity> entities = EntityExtractor.extractEntities(textChars, convertToLabels(labelIds[0], effectiveLength));
            return entities.stream()
                    .map(e -> e.word)
                    .collect(Collectors.toSet());
        }
    }

    /**
     * Batch 预测句子实体词
     *
     * @param sentences 所有句子
     * @param batchSize 用于预测的batch size
     * @return 句子中所有实体词
     */
    public List<Set<String>> predictOnBatch(List<String> sentences, int batchSize) {
        int sentencesSize = sentences.size();
        int n = Math.floorDiv(sentencesSize, batchSize);
        float padValue = token2idx.get(PAD);

        List<Set<String>> entitiesSeq = new ArrayList<>();
        for (int i = 0; i <= n; i++) {
            if (i * batchSize >= sentencesSize) {
                break;
            }
            List<String> subSentences = sentences.subList(i * batchSize, Math.min((i + 1) * batchSize, sentencesSize));
            int subSize = subSentences.size();
            float[][] padded = new float[subSize][SEQUENCE_LENGTH];
            float[][] another = new float[subSize][SEQUENCE_LENGTH];

            char[][] charsSeq = new char[subSize][];
            for (int j = 0; j < subSize; j++) {
                char[] chars = subSentences.get(j).toLowerCase().toCharArray();
                charsSeq[j] = chars;
                float[] tokenIds = tokenize(chars);
                padded[j] = SequenceUtils.pad1DSequence(tokenIds, SEQUENCE_LENGTH, padValue);
            }

            // tensor predict
            try (Tensor<Float> x1 = Tensor.create(padded, Float.class);
                 Tensor<Float> x2 = Tensor.create(another, Float.class)) {
                Tensor y = session.runner()
                        .feed("Input-Token", x1)
                        .feed("Input-Segment", x2)
                        .fetch("CRF/cond/Merge")
                        .run()
                        .get(0);


                float[][][] scoreMatrix = (float[][][]) y.copyTo(new float[subSize][SEQUENCE_LENGTH][label2idx.size()]);
                int[][] labelIdMatrix = ArrayMathUtils.argmax(scoreMatrix);

                List<Set<String>> subResult = new ArrayList<>(subSize);
                for (int j = 0; j < subSize; j++) {
                    char[] chars = charsSeq[j];
                    int effectiveLength = Math.min(chars.length, SEQUENCE_LENGTH - 2);
                    List<Entity> entities = EntityExtractor.extractEntities(chars, convertToLabels(labelIdMatrix[j], effectiveLength));
                    subResult.add(entities.stream()
                            .map(e -> e.word)
                            .collect(Collectors.toSet()));
                }
                entitiesSeq.addAll(subResult);
            }
        }
        return entitiesSeq;
    }

    /**
     * 模型输出的序列标注类别映射为标签。
     *
     * @param labelIds        模型输出的序列标注类别，为编码值数组
     * @param effectiveLength 序列有效长度，用于剔除padding 类别
     * @return 句子每个token 对应的标签
     */
    private List<String> convertToLabels(int[] labelIds, int effectiveLength) {
        List<String> labels = new ArrayList<>(labelIds.length);
        for (int i = 1; i <= effectiveLength; i++) {
            labels.add(idx2Label.get(labelIds[i]));
        }
        return labels;
    }
}