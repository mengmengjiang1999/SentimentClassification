# SentimentClassification

本项目是 **2019 年春季清华大学《人工智能导论》课程第三次大作业**，主题为中文新闻情感分类。

## 项目简介

项目使用新浪新闻数据集，将 4,570 篇新闻划分为 8 种情感类别，并实现、比较了以下模型：

- CNN 文本分类模型
- 基于双向 LSTM 的 RNN 模型
- MLP baseline

文本表示采用 [Chinese Word Vectors](https://github.com/Embedding/Chinese-Word-Vectors) 提供的 Sogou News 预训练词向量。详细的模型设计、参数设置和实验结果见[课程作业报告](./SentimentClassification-%E9%99%88%E5%BC%A0%E8%90%8C/%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E5%AF%BC%E8%AE%BA-%E7%AC%AC%E4%B8%89%E6%AC%A1%E4%BD%9C%E4%B8%9A.md)。

## 目录说明

- `SentimentClassification-陈张萌/`：模型代码、数据、实验报告及部分已训练权重
- `实验数据/`：课程提供的原始实验数据
- `实验三介绍PPT.pptx`、`实验三说明文档.pdf`：课程作业说明材料

## 运行方式

进入代码目录：

```bash
cd SentimentClassification-陈张萌
```

训练模型：

```bash
python3 main.py -m rnn
python3 main.py -m cnn
python3 main.py -m mlp
```

测试已有模型：

```bash
python3 test.py -m rnn
python3 test.py -m cnn
```

运行前请根据 `models/config.py` 中的配置准备预训练词向量，并安装 PyTorch、NumPy、SciPy 和 scikit-learn 等依赖。
