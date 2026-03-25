# toyNLP

> 工作之余的 NLP 实验项目，涵盖命名实体识别、文本分类、文本相似度计算等经典任务，支持 Python / Java / Kotlin 多语言实现。

## 功能概览

| 任务 | 算法 | 语言 |
|------|------|------|
| 命名实体识别 (NER) | Bi-LSTM + CRF | Python (Keras) |
| 命名实体识别 (NER) | BERT + CRF | Python (Keras) / Java (TensorFlow) |
| 命名实体识别 (NER) | ALBERT + CRF | Python (Keras) |
| 文本分类 | char Bigram + TF-IDF + SVM | Python (scikit-learn) |
| 文本相似度 | MaLSTM (Siamese LSTM) | Python (Keras) |
| 基础组件 | 双数组 Trie 树 (Double Array Trie) | Kotlin |

## 项目结构

```
toyNLP/
├── python/                     # Python 模块
│   ├── toynlp/                 #   核心包
│   │   ├── ner/                #     NER 模型 (BiLSTM-CRF / BERT-CRF / ALBERT-CRF)
│   │   ├── text_classify/      #     文本分类 (char Bigram + SVM)
│   │   ├── text_similarity/    #     文本相似度 (MaLSTM)
│   │   ├── chinese_utils.py    #     中文分字、繁简转换、全角半角转换
│   │   └── helper.py           #     数据加载与词表构建
│   ├── examples/               #   训练与评估示例脚本
│   │   ├── ner/                #     NER 训练脚本
│   │   └── text_classify/      #     文本分类训练脚本
│   └── requirements.txt        #   Python 依赖
├── java/                       # Java 模块 — BERT NER 推理 (TensorFlow Java)
│   └── src/
├── kotlin/                     # Kotlin 模块 — 双数组 Trie 树
│   └── src/
└── data/                       # 数据集
    ├── ner/                    #   CoNLL 格式 NER 标注数据 (train/dev/test)
    └── dicts/                  #   词典资源 (成语词典等)
```

## 快速开始

### Python

```bash
pip install -r python/requirements.txt

# 训练 NER 模型示例
python python/examples/ner/train_bilstm_crf.py
```

### Java

```bash
cd java
mvn clean install
```

- **Java 8+**，Maven 构建
- 基于 TensorFlow Java 1.12 加载 BERT 模型进行 NER 推理
- BIOES 标签序列解码为实体区间

### Kotlin

```bash
cd kotlin
mvn clean install
```

- **Kotlin 1.6**，Maven 构建
- 双数组 Trie 树的构建、序列化与前缀/精确匹配

## 参考文献

1. Lample, G., et al. *Neural Architectures for Named Entity Recognition.* arXiv:1603.01360, 2016.
2. Mueller, J., & Thyagarajan, A. *Siamese Recurrent Architectures for Learning Sentence Similarity.* AAAI, 2016: 2786-2792.

## 许可证

本项目基于 [Apache License 2.0](LICENSE) 开源。
