package io.github.yizhiru.toynlp.ner;

import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.*;

public class EntityExtractorTest {

    @Test
    public void extractEntities() {
        List<Entity> entities;
        entities = EntityExtractor.extractEntities("黑皮诺干红葡萄酒".toCharArray(),
                Arrays.asList("O", "O", "O", "B-PROD", "E-PROD", "B-PROD", "I-PROD", "E-PROD"));
        assertEquals(entities.toString(), "[干红, 葡萄酒]");

        entities = EntityExtractor.extractEntities("黑皮诺干红葡萄酒".toCharArray(),
                Arrays.asList("O", "O", "O", "B-PROD", "I-PROD", "B-PROD", "E-PROD", "S-PROD"));
        assertEquals(entities.toString(), "[葡萄, 酒]");

        entities = EntityExtractor.extractEntities("向着波士顿湾无声逝去".toCharArray(),
                Arrays.asList("O", "O", "B-LOC", "I-PROD",  "I-LOC", "E-LOC", "B-PROD", "B-PROD", "I-PROD"));
        assertEquals(entities.toString(), "[]");

        entities = EntityExtractor.extractEntities("黑皮诺干红葡萄酒".toCharArray(),
                Arrays.asList("O", "O", "O", "B-PROD", "I-PROD", "B-PROD", "E-PROD", "B-PROD"));
        assertEquals(entities.toString(), "[葡萄]");

        entities = EntityExtractor.extractEntities("黑皮诺干红葡萄酒".toCharArray(),
                Arrays.asList("O", "O", "O", "B-PROD", "I-PROD", "O", "B-PROD", "I-PROD"));
        assertEquals(entities.toString(), "[]");

        entities = EntityExtractor.extractEntities("黑皮诺干红葡萄酒".toCharArray(),
                Arrays.asList("O", "O", "O", "B-PROD", "I-PROD", "O", "O", "E-PROD"));
        assertEquals(entities.toString(), "[]");
    }
}