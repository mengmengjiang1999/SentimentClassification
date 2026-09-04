from models.config import MLPConfig,MLPTrainingConfig,Config
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.stats import pearsonr


class textMLP(nn.Module):
    def __init__(self, seq_len, hidden_size = MLPConfig.hidden_size):
        super(textMLP, self).__init__()

        self.input_size = seq_len * MLPConfig.emb_size
        self.hidden_size = hidden_size
        self.label_len = MLPConfig.label_len

        self.Linear1 = nn.Linear(self.input_size, self.hidden_size)
        self.Linear2 = nn.Linear(self.hidden_size, self.label_len)

    def forward(self, input_data):  # (N, seqlen, 300)
        input_data = input_data.view(-1, self.input_size)
        input_data = self.Linear1(input_data)
        input_data = F.relu(input_data)
        output = self.Linear2(input_data)

        return output

class MLP_model(object):
    def __init__(self, vocab_size, embedding=None):
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

        self.model = textMLP(vocab_size).to(self.device)
        self.epoches = MLPTrainingConfig.epoches
        self.learning_rate = MLPTrainingConfig.learning_rate
        self.print_step = MLPTrainingConfig.print_step
        self.lr_decay = MLPTrainingConfig.factor
        self.patience = MLPTrainingConfig.patience
        self.verbose = MLPTrainingConfig.verbose

        if embedding is not None:
            self.model.init_embedding(embedding.to(self.device))

        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate,momentum=0.9)
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

        score = f1_score(y_true, y_pred, average='macro')
        coef = np.average([pearsonr(outputs[i], labels[i]) for i in range(outputs.shape[0])])
        print(type(coef))
        acc = correct_num /count
        if acc > self.best_acc and is_test:
            self.best_acc = acc
            torch.save(self.model.state_dict(),'mlp_best.pkl')
        print('Accuracy: {:.2f}%'.format(100 * acc))
        print('FScore: {:.2f}%'.format(100 * score))
        print('Coef: {:.2f}%'.format(100 * coef))

        avg_loss = losses / len(test_loader)
        self.lr_scheduler.step(avg_loss)
        return acc,score

