from torch.utils.data import DataLoader
from utils import load_embeddings, load_word2id
from models.cnn import CNN_model
from models.rnn import RNN_model
from models.mlp import MLP_model
from myDataset import MyDataset,load_word2vec
from models.config import Config
import argparse
from sklearn.metrics import f1_score
import torch
import torch
import numpy as np
from scipy.stats import pearsonr
VOCAB_SIZE = 2331
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', help='select a model from  [\'cnn\', \'lstm\', \'mlp\']')
    args = parser.parse_args()
    word2id = load_word2id(length=VOCAB_SIZE)
    vocab_size = len(word2id)
    word2vec = load_word2vec(Config.reduced_vec_path)
    test_data = MyDataset(Config.test_data_path, Config.seq_len, Config.vec_len, Config.label_len, word2vec)
    test_dataloader = DataLoader(dataset=test_data, shuffle=True, num_workers=0, batch_size=Config.train_batch_size)


    if args.m in ['rnn', 'cnn', 'mlp']:
        try:
            embeddings = load_embeddings(word2id)
        except FileNotFoundError:
            embeddings = None
    if args.m == 'rnn':
        print("rnn")
        rnn_model = RNN_model(vocab_size, embedding=embeddings)
        rnn_model.model.load_state_dict(torch.load('rnn_best.pkl'))
        rnn_model.test(test_dataloader,True)

    elif args.m == 'cnn':
        print("CNN")
        cnn_model = CNN_model(vocab_size, embedding=embeddings)
        cnn_model.model.load_state_dict(torch.load('cnn_best.pkl'))
        cnn_model.test(test_dataloader, True)

    elif args.m == 'mlp':
        print("MLP")
        mlp_model = MLP_model(vocab_size, embedding=embeddings)
        mlp_model.model.load_state_dict(torch.load('mlp_best.pkl'))
        mlp_model.test(test_dataloader, True)


if __name__ == "__main__":
    main()