import os
import torch


class Config:
    data_root = './data'
    vec_path = os.path.join(data_root, 'sgns.sogounews.bigram-char')
    reduced_vec_path = os.path.join(data_root, 'reduced.sgns.sogounews.bigram-char')
    train_data_path = os.path.join(data_root, 'sinanews.train')
    valid_data_path = os.path.join(data_root, 'sinanews.valid')
    test_data_path = os.path.join(data_root, 'sinanews.test')
    vocab_data_path = os.path.join(data_root, 'vocab.txt')

    label_len = 8
    vec_len = 300
    seq_len = 500
    unk_vec = torch.rand([vec_len])*2 - 1
    filter_num = 4
    train_batch_size = 32
    lr = 0.001
    momentum = 0.9
    epoch_num = 100
    sentence_max_len = 2048


class RNNConfig:
    emb_size = 300
    hidden_size = 20
    label_len = 8
    num_layer = 1
    dropout = 0.5


class CNNConfig:
    emb_size = 300
    num_filters = 180
    label_len = 8
    kernel_size = [3, 4, 5]
    dropout_rate = 0.5


class MLPConfig:
    emb_size = 150
    hidden_size = 20
    label_len = 8
    dropout = 0.3


class RNNTrainingConfig:
    learning_rate = 0.0032
    epoches = 300
    print_step = 6
    factor = 0.5
    patience = 1
    verbose = True


class CNNTrainingConfig:
    learning_rate = 0.001
    epoches = 300
    print_step = 6
    factor = 0.5
    patience = 1
    verbose = True


class MLPTrainingConfig:
    learning_rate = 0.003
    epoches = 300
    print_step = 6
    factor = 0.5
    patience = 1
    verbose = True
