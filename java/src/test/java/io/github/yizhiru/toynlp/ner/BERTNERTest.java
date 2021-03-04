package io.github.yizhiru.toynlp.ner;

import org.apache.commons.io.IOUtils;
import org.junit.Test;

import java.io.FileReader;
import java.io.IOException;
import java.util.List;
import java.util.Set;

public class BERTNERTest {


    @Test
    public void predictOnBatch() throws IOException {
        BERTNER ner = BERTNER.load("/model");
        List<String> lines = IOUtils.readLines(new FileReader("data/ner/case.txt"));
        List<Set<String>> prodsList = ner.predictOnBatch(lines, 1);
        for (int i = 0; i < lines.size(); i++) {
            System.out.println(lines.get(i) + "\t" + String.join(",", prodsList.get(i)));
        }
    }
}