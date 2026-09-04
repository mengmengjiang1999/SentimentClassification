from torch.utils.data import DataLoader
from utils import load_embeddings, load_word2id
from models.cnn import CNN_model
from models.rnn import RNN_model
from models.mlp import MLP_model
from myDataset import MyDataset,load_word2vec
from models.config import Config
import argparse
import torch


VOCAB_SIZE = 2331  # Vocabulary size


def main():
    print("reduced_vec_path is...")
    reduced_vec_path = "./data/reduced.sgns.sogounews.bigram-char"
    print("reduced.sgns.sogounews.bigram-cha")
    word2vec = load_word2vec(reduced_vec_path)
    train_data = MyDataset(Config.train_data_path, Config.seq_len, Config.vec_len, Config.label_len, word2vec)
    # valid_data = MyDataset(Config.valid_data_path, Config.seq_len, Config.vec_len, Config.label_len, word2vec)
    test_data = MyDataset(Config.test_data_path, Config.seq_len, Config.vec_len, Config.label_len, word2vec)
    train_dataloader = DataLoader(dataset=train_data, shuffle=True, num_workers=0, batch_size=Config.train_batch_size)
    # valid_dataloader = DataLoader(dataset=valid_data, shuffle=True, num_workers=0, batch_size=Config.train_batch_size)
    test_dataloader = DataLoader(dataset=test_data, shuffle=True, num_workers=0, batch_size=Config.train_batch_size)

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', help='select a model from  [\'cnn\', \'rnn\', \'mlp\']')
    args = parser.parse_args()

    word2id = load_word2id(length=VOCAB_SIZE)

    vocab_size = len(word2id)
    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    if args.m in ['rnn', 'cnn', 'mlp']:
        try:
            embeddings = load_embeddings(word2id)
        except FileNotFoundError:
            embeddings = None
    if args.m == 'rnn':
        print("rnn")
        rnn_model = RNN_model(vocab_size, embedding=embeddings)
        rnn_acc = rnn_model.train_and_eval(train_dataloader, test_dataloader)
        print("Accuracy of RNN: {:.2f}".format(100 * rnn_acc))

    elif args.m == 'cnn':
        print("CNN")
        cnn_model = CNN_model(vocab_size, embedding=embeddings)
        cnn_acc = cnn_model.train_and_eval(train_dataloader, test_dataloader)
        print("Accuracy of CNN: {:.2f}".format(100 * cnn_acc))
    elif args.m == 'mlp':
        print("MLP")
        mlp_model = MLP_model(Config.seq_len, embedding=embeddings)
        mlp_acc = mlp_model.train_and_eval(train_dataloader, test_dataloader)
        print("Accuracy of MLP: {:.2f}".format(100 * mlp_acc))


if __name__ == "__main__":
    main()
