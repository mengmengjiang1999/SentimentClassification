# SentimentClassification

本项目是 **2019 年春季清华大学《人工智能导论》课程第三次大作业**，主题为中文新闻情感分类。

## 项目简介

项目使用新浪新闻数据集，将 4,570 篇新闻划分为 8 种情感类别，并实现、比较了以下模型：

- CNN 文本分类模型
- 基于双向 LSTM 的 RNN 模型
- MLP baseline

文本表示采用 [Chinese Word Vectors](https://github.com/Embedding/Chinese-Word-Vectors) 提供的 Sogou News 预训练词向量。详细的模型设计、参数设置和实验结果见[课程作业报告](./REPORT.md)。

## 目录说明

- `models/`：CNN、RNN 和 MLP 模型及配置
- `data/`：匿名合成的词表和格式示例（不包含完整课程数据集）
- `images/`：实验报告使用的模型结构图
- `REPORT.md`：课程作业报告
- `main.py`、`test.py`：训练和测试入口

## 运行方式

安装依赖：

```bash
python3 -m pip install -r requirements.txt
```

然后在 `data/` 目录中准备以下本地文件：

- `sinanews.train`
- `sinanews.valid`
- `sinanews.test`
- `reduced.sgns.sogounews.bigram-char`

数据集、预训练词向量和模型权重体积较大，不纳入本仓库版本管理。路径及训练参数可在 `models/config.py` 中调整。
仓库内的 `sinanews.demo` 仅用于展示输入格式，不是原始新闻语料。

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

`test.py` 还需要将对应的本地权重文件（如 `rnn_best.pkl` 或 `cnn_best.pkl`）放在仓库根目录。
