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

public class ProductNER {

    private Map<String, Integer> label2idx;

    private Map<Integer, String> idx2Label;

    private Map<String, Float> token2idx;

    private Session session;

    private final String PAD = "[PAD]";

    private final String UNK = "[UNK]";

    private final String CLS = "[CLS]";

    private final String SEP = "[SEP]";


    private final int SEQUENCE_LENGTH = 100;

    private ProductNER(Map<String, Integer> label2idx, Map<String, Float> token2idx, Session session) {
        this.label2idx = label2idx;
        this.idx2Label = MapUtils.invertMap(label2idx);
        this.token2idx = token2idx;
        this.session = session;
    }

    public static ProductNER load(String path) throws IOException {
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
        return new ProductNER(label2idx, token2idx, session);
    }

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

    private List<String> convertToLabels(int[] labelIds, int effectiveLength) {
        List<String> labels = new ArrayList<>(labelIds.length);
        for (int i = 1; i <= effectiveLength; i++) {
            labels.add(idx2Label.get(labelIds[i]));
        }
        return labels;
    }
}
