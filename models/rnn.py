import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.config import RNNConfig,RNNTrainingConfig,Config
from sklearn.metrics import f1_score
from scipy.stats import pearsonr

class textRNN(nn.Module):
    def __init__(self, vocab_size,hidden_size = RNNConfig.hidden_size, num_layer=RNNConfig.num_layer):
        super(textRNN, self).__init__()

        self.emb_size = RNNConfig.emb_size
        self.label_len = RNNConfig.label_len
        self.hidden_size = hidden_size
        self.num_layer = num_layer

        self.rnn = nn.LSTM(input_size=self.emb_size, hidden_size=self.hidden_size, num_layers=num_layer, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.hidden_size * 2, self.label_len)

    def forward(self, input_data):  # (N, seqlen, 300)
        input_data, (hn, cn) = self.rnn(input_data)  # (N, seqlen, 2*Hidden)
        input_data = input_data.permute(0, 2, 1)  # (N, 2*Hidden, seqlen)
        input_data = F.max_pool1d(input_data, input_data.size(2)).squeeze(2)  # (N, 2*Hidden)
        output = self.fc(input_data)

        return output

class RNN_model(object):
    def __init__(self, vocab_size, embedding=None):
        self.device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

        self.model = textRNN(vocab_size).to(self.device)
        self.epoches = RNNTrainingConfig.epoches
        self.learning_rate = RNNTrainingConfig.learning_rate
        self.print_step = RNNTrainingConfig.print_step
        self.lr_decay = RNNTrainingConfig.factor
        self.patience = RNNTrainingConfig.patience
        self.verbose = RNNTrainingConfig.verbose

        if embedding is not None:
            self.model.init_embedding(embedding.to(self.device))

        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=Config.momentum)
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer,
                                              'min',
                                              factor=self.lr_decay,
                                              patience=self.patience,
                                              verbose=self.verbose
                                              )
        # self.loss_fn = nn.BCELoss()
        self.loss_fn = nn.CrossEntropyLoss()
        self.best_acc = 0.

    def train_and_eval(self, train_loader, test_loader):
        for epoch in range(1, self.epoches +1):
            print('Epoch {}: training...'.format(epoch))
            running_loss = 0.0
            step = 0
            for i, data in enumerate(train_loader):
                inputs, _, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, torch.argmax(labels, 1))
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                step += 1

            print('Train ', end='')
            self.test(train_loader, False)
            print('Test ', end='')
            self.test(test_loader, True)

        print('Best Accuracy: {:.2f}%'.format(100 * self.best_acc))
        return self.best_acc

    def test(self, test_loader, is_test):
        count, correct_num, losses = 0., 0., 0.
        self.model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                inputs, _, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                prediction = torch.argmax(outputs, 1)
                gt = torch.argmax(labels, 1)
                y_pred.extend(prediction.cpu().numpy().tolist())
                y_true.extend(gt.cpu().numpy().tolist())
                correct_num += (prediction == gt).sum().float()
                count += len(labels)
                loss = self.loss_fn(outputs, gt)
                losses += loss.item()

        # print(y_true)
        # print(y_pred)
        score = f1_score(y_true, y_pred, average='macro')
        coef = np.average([pearsonr(outputs[i], labels[i]) for i in range(outputs.shape[0])])
        print(type(coef))
        acc = correct_num /count
        if acc > self.best_acc and is_test:
            self.best_acc = acc
            torch.save(self.model.state_dict(),'rnn_best.pkl')
        print('Accuracy: {:.2f}%'.format(100 * acc))
        print('FScore: {:.2f}%'.format(100 * score))
        print('Coef: {:.2f}%'.format(100 * coef))


        avg_loss = losses / len(test_loader)
        self.lr_scheduler.step(avg_loss)
        return acc,score

