import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.config import CNNConfig,CNNTrainingConfig
from models.config import Config
from sklearn.metrics import f1_score
from scipy.stats import pearsonr


class textCNN(nn.Module):
    def __init__(self, vocab_size, num_filters = CNNConfig.num_filters, kernel_size = CNNConfig.kernel_size, dropout_rate = CNNConfig.dropout_rate):
        super(textCNN, self).__init__()

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.emb_size = CNNConfig.emb_size
        self.label_len = CNNConfig.label_len
        self.dropout_rate = dropout_rate

        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=self.num_filters, kernel_size=(K, self.emb_size)) for K in
             self.kernel_size])  # 卷积层，卷积核大小为(K, 300)

        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc = nn.Linear(len(self.kernel_size) * self.num_filters, self.label_len)

    def forward(self, input_data):
        # input x's shape: (N,  W, 300)
        input_data = input_data.unsqueeze(1)  # (N, 1, W, 300)
        input_data= [F.relu(conv(input_data)).squeeze(3) for conv in self.convs]  # (N, C_out, W-K+1), 一共len(self.Ks)个
        input_data = [F.max_pool1d(t, t.size(2)).squeeze(2) for t in input_data]  # (N, C_out)，一共len(self.Ks)个

        input_data = torch.cat(input_data, 1)  # (N, C_out*(len(self.Ks)))
        input_data = self.dropout(input_data)
        output = self.fc(input_data)  # (N, classnum)
        return output

class CNN_model(object):
    def __init__(self, vocab_size, embedding=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = textCNN(vocab_size).to(self.device)
        self.epoches = CNNTrainingConfig.epoches
        self.learning_rate = CNNTrainingConfig.learning_rate
        self.print_step = CNNTrainingConfig.print_step
        self.lr_decay = CNNTrainingConfig.factor
        self.patience = CNNTrainingConfig.patience
        self.verbose = CNNTrainingConfig.verbose

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
            torch.save(self.model.state_dict(),'cnn_best.pkl')
        print('Accuracy: {:.2f}%'.format(100 * acc))
        print('FScore: {:.2f}%'.format(100 * score))
        print('Coef: {:.2f}%'.format(100 * coef))


        avg_loss = losses / len(test_loader)
        self.lr_scheduler.step(avg_loss)
        return acc,score

