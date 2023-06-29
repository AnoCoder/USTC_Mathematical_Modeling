import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from matplotlib import pyplot as plt
from model import DNN
from earlystopping import EarlyStopping
from sklearn import metrics
import torch.nn.functional as Func

# filename_train = './insects/insects-training.txt'
# filename_test = './insects/insects-testing.txt'
filename_train = './insects/insects-2-training.txt'  # 有噪声样本
filename_test = './insects/insects-2-testing.txt'


class Insect_Classifier:
    def __init__(self):
        self.train_data = None
        self.verification_data = None
        self.test_data = None

    def load_data(self, random_state):
        np.random.seed(random_state)
        X = np.loadtxt(filename_train, usecols=(0, 1))
        Y = np.loadtxt(filename_train, dtype='int', usecols=2)
        X_test = np.loadtxt(filename_test, usecols=(0, 1))
        y_test = np.loadtxt(filename_test, dtype='int', usecols=2)
        # 从X，Y中划分训练集与验证集
        data_size = int(len(X))
        indices = np.random.permutation(data_size)
        train_ratio = 0.8  # 默认训练集的比例为0.8
        index_train, index_verification = np.split(indices, [int(data_size*train_ratio)])
        x_train, y_train = X[index_train], Y[index_train]
        x_verification, y_verification = X[index_verification], Y[index_verification]
        self.train_data = (torch.Tensor(x_train).float(), torch.Tensor(y_train).long())
        self.verification_data = (torch.Tensor(x_verification).float(), torch.Tensor(y_verification).long())
        self.test_data = (torch.Tensor(X_test), torch.Tensor(y_test).long())
        return self.train_data, self.verification_data, self.test_data

    def train(self, model, optimizer, criterion):
        x, y = self.train_data
        correct = 0
        y_pred = model(x)
        loss = criterion(y_pred, y)
        pred = y_pred.argmax(dim=1, keepdim=True)
        for i in range(len(pred)):
            if pred[i] == y[i]:
                correct += 1
        acc = correct / len(pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item(), acc

    def validate(self, model, criterion):
        correct = 0
        x_verification, y_verification = self.verification_data
        with torch.no_grad():
            y_pred = model(x_verification)
            loss = criterion(y_pred, y_verification).item()
            pred = y_pred.argmax(dim=1, keepdim=True)
            for i in range(len(pred)):
                if pred[i] == y_verification[i]:
                    correct += 1
        acc = correct / len(pred)
        return loss, acc

    def test(self, model, criterion):
        correct, correct_1, correct_2 = 0, 0, 0
        x_test, y_test = self.test_data
        with torch.no_grad():
            y_pred = model(x_test)
            loss = criterion(y_pred, y_test).item()
            pred = y_pred.argmax(dim=1, keepdim=True)
            for i in range(len(pred)):
                if pred[i] == y_test[i]:
                    correct += 1
                    if i < 60:
                        correct_1 += 1
                    else:
                        correct_2 += 1
        acc = correct / len(pred)
        acc_1 = correct_1 / 60
        acc_2 = correct_2 / 150
        print('Test loss: {0:7.6f}  |total acc: {1:7.6f}   |acc_60: {2:7.6f}    |acc_150: {3:7.6f}'
              .format(loss, acc, acc_1, acc_2))
        self.visualize_result(x_test, y_test, pred)
        paint_ROC(y_test, y_pred)

    def plot_loss_acc(self, num_rounds, loss_tr_hist, loss_val_hist, acc_tr_hist, acc_val_hist):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        x = range(1, num_rounds+1)
        ax1.set_title('Training history')
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss')
        ax2.set_ylabel('acc')
        p1 = ax1.plot(x, loss_tr_hist, label='train_loss')
        p2 = ax1.plot(x, loss_val_hist, label='val_loss')
        p3 = ax2.plot(x, acc_tr_hist, '-.', label='train_acc')
        p4 = ax2.plot(x, acc_val_hist, '-.', label='val_acc')
        lines = p1 + p2 + p3 + p4
        labels = [line.get_label() for line in lines]
        plt.legend(lines, labels)
        plt.show()

    def visualize_result(self, x_test, y_test, y_pred):
        fig = plt.figure(figsize=(12, 8))
        ax1 = fig.add_subplot(121)
        ax1.set_title('Original test set data distribution')
        data_test_0 = x_test[y_test[:] == 0]
        data_test_1 = x_test[y_test[:] == 1]
        data_test_2 = x_test[y_test[:] == 2]
        ax1.set_xlabel('length')
        ax1.set_ylabel('wing length')
        ax1.scatter(data_test_0[:, 0], data_test_0[:, 1], c='r', label='label 0')
        ax1.scatter(data_test_1[:, 0], data_test_1[:, 1], c='b', marker='*', label='label 1')
        ax1.scatter(data_test_2[:, 0], data_test_2[:, 1], c='g', marker='^', label='label 2')
        ax1.legend()

        ax2 = fig.add_subplot(122)
        ax2.set_title('Classification predictions on the test set')
        data_pred_0 = x_test[y_pred[:, 0] == 0]
        data_pred_1 = x_test[y_pred[:, 0] == 1]
        data_pred_2 = x_test[y_pred[:, 0] == 2]
        ax2.set_xlabel('length')
        ax2.set_ylabel('wing length')
        ax2.scatter(data_pred_0[:, 0], data_pred_0[:, 1], c='r', label='label 0')
        ax2.scatter(data_pred_1[:, 0], data_pred_1[:, 1], c='b', marker='*', label='label 1')
        ax2.scatter(data_pred_2[:, 0], data_pred_2[:, 1], c='g', marker='^', label='label 2')
        ax2.legend()
        plt.show()

    def fit(self, neurons, activation, lr=0.001, max_epoch=1000):
        start_epoch = 0
        early_stopping = EarlyStopping(verbose=True)
        num_rounds = 0
        loss_tr_hist, loss_val_hist = [], []
        acc_tr_hist, acc_val_hist = [], []
        model = DNN(neurons, activation)
        print('number of parameters: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
        for epoch in range(start_epoch + 1, max_epoch + 1):
            optim = Adam(model.parameters(), lr)
            criterion = nn.CrossEntropyLoss()
            loss_tr, acc_tr = self.train(model, optim, criterion)
            loss_tr_hist.append(loss_tr)
            acc_tr_hist.append(acc_tr)
            loss_val, acc_val = self.validate(model, criterion)
            loss_val_hist.append(loss_val)
            acc_val_hist.append(acc_val)
            early_stopping(acc_val, model)
            num_rounds += 1
            # 若满足 early stopping 要求
            if early_stopping.early_stop:
                print("Early stopping in epoch", epoch)
                break  # 结束模型训练
        for i in range(num_rounds):
            if (i+1) % 10 == 0:  # 每10轮输出一次
                print('Epoch:{0:d}  |Train Loss: {1:7.6f}  |Val loss: {2:7.6f}   |Train_ACC: {3:7.6f}   '
                      '|Val_ACC:{4:7.6f}'
                      .format(i+1, loss_tr_hist[i], loss_val_hist[i], acc_tr_hist[i], acc_val_hist[i]))
        self.plot_loss_acc(num_rounds, loss_tr_hist, loss_val_hist, acc_tr_hist, acc_val_hist)
        self.test(model, criterion)


def paint_ROC(y_test, y_score):
    y_score = Func.softmax(y_score, dim=1)
    # print(y_score)
    plt.figure()
    colors = ['darkred', 'darkorange', 'cornflowerblue']
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    label = np.zeros((len(y_test), 3),  dtype="uint8")
    for i in range(len(y_test)):
        label[i][int(y_test[i])] = 1
    for i in range(3):
        fpr[i], tpr[i], _ = metrics.roc_curve(label[:,i], y_score[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    fpr["mean"], tpr["mean"], _ = metrics.roc_curve(label.ravel(), y_score.ravel())
    roc_auc["mean"] = metrics.auc(fpr["mean"], tpr["mean"])
    lw = 2
    plt.plot(fpr["mean"], tpr["mean"],
         label='average, ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["mean"]),
         color='k', linewidth=lw)
    for i in range(3):
        auc = roc_auc[i]
        # 输出不同类别的FPR\TPR\AUC
        print('label: {}, fpr: {}, tpr: {}, auc: {}'.format(i, np.mean(fpr[i]), np.mean(tpr[i]), auc))
        plt.plot(fpr[i], tpr[i], color=colors[i], linestyle=':', lw=lw,
                 label='Label = {0}, ROC curve (area = {1:0.2f})'.format(i, auc))

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(linestyle='-.')
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.show()



if __name__ == '__main__':
    classifier = Insect_Classifier()
    classifier.load_data(random_state=7)
    classifier.fit(neurons=[2, 20, 25,40, 3], activation='elu')
