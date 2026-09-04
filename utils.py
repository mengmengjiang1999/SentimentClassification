import string
import torch
from models.config import Config

def load_word2id(length=2000, vocab_path="./data/vocab.txt"):
    word2id = {"<pad>": 0, "<unk>": 1}
    with open(vocab_path, "r") as f:
        words = [line.split(',')[0] for line in f]
    for word in words[:length]:
        word2id[word] = len(word2id)
    return word2id


def load_embeddings(word2id, emb_dim=300,
                    emb_path="./data/embeddings"):
    vocab_size = len(word2id)
    embedding = torch.Tensor(vocab_size, emb_dim)

    word2embstr = {}
    with open(emb_path, 'r') as f:
        for line in f:
            word, embstr = line.split(" ", 1)
            word2embstr[word] = embstr.strip("\n")

    for word, word_id in word2id.items():
        if word in word2embstr:
            embs = list(map(float, word2embstr[word].split()))
            embedding[word_id] = torch.Tensor(embs)
        else:
            embedding[word_id] = torch.randn(emb_dim)

    return embedding


def collate_fn_dl(word2id, max_len, batch):
    batch.sort(key=lambda pair: len(pair[1]), reverse=True)
    labels, sentences = zip(*batch)
    sentences = [sent[:Config.sentence_max_len] for sent in sentences]
    labels = torch.LongTensor(labels)

    pad_id = word2id["<pad>"]
    unk_id = word2id["<unk>"]
    bsize = len(sentences)
    max_len = max(len(sentences[0]), max_len)
    sent_tensor = torch.ones(bsize, max_len).long() * pad_id
    for sent_id, sent in enumerate(sentences):
        for word_id, word in enumerate(sent):
            sent_tensor[sent_id][word_id] = word2id.get(word, unk_id)

    lengths = [len(sent) for sent in sentences]
    return labels, sent_tensor, lengths