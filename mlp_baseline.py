import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class mlp(nn.Module):
    def __init__(self, input_dim):
        super(mlp, self).__init__()
        self.input_dim = input_dim
        self.linear1 = nn.Linear(self.input_dim, 64)
        self.linear2 = nn.Linear(64, 1)

    def forward(self, x):
        linear1_out = self.linear1(x)
        linear2_out =  self.linear2(linear1_out)
        output = torch.sigmoid(linear2_out)

        return output


def train(train_data, test_data, train_label, test_label):
    train_data = torch.from_numpy(train_data).float().to(device)
    test_data = torch.from_numpy(test_data).float().to(device)
    train_label = torch.from_numpy(train_label).float().to(device)
    test_label = torch.from_numpy(test_label).float().to(device)
    print(train_label)
    model = mlp(train_data.shape[1])
    loss = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    n_epochs = 20000
    epoch_list = []
    acc_list = []
    loss_list = []
    with open('output/mlp_baseline/training_output.txt', 'w') as f:
        f.truncate(0)

    with open('output/mlp_baseline/testing_output.txt', 'w') as f:
        f.truncate(0)
    for epoch in range(n_epochs):
        model.train()
        model.zero_grad()
        pred_probs = model(train_data)
        y_pred = pred_probs > 0.5
        loss_val = loss(pred_probs, train_label)
        loss_val.backward()
        optimizer.step()
        acc = accuracy_score(train_label, y_pred)
        print("Epoch: {}, Loss: {}, Accuracy: {}".format(epoch, loss_val, acc))
        with open('output/mlp_baseline/training_output.txt', 'a') as f:
            f.write("Epoch: {} Loss: {} Accuracy: {}\n".format(epoch, loss_val, acc))
        if epoch % 100 == 0:
            print("-"*80)
            print("Testing Epoch: {}".format(epoch))
            model.eval()
            with torch.no_grad():
                pred_prob_test = model(test_data)
                y_pred_test = pred_prob_test > 0.5
                loss_val_test = loss(pred_prob_test, test_label)
                acc_test = accuracy_score(test_label, y_pred_test)

                print("Epoch: {}, Loss: {}, Accuracy: {}".format(epoch, loss_val_test, acc_test))
            with open('output/mlp_baseline/testing_output.txt', 'a') as f:
                f.write("Epoch: {} Loss: {} Accuracy: {}\n".format(epoch, loss_val_test, acc_test))
            model.train()
            print("-"*80)
        #if epoch

if __name__ == '__main__':
    train_data = np.load('data/train_data_1200_1_mlp.npy')
    test_data = np.load('data/test_data_1200_1_mlp.npy')
    train_label = np.load('data/train_label_1200_1.npy')
    test_label = np.load('data/test_label_1200_1.npy')
    train(train_data, test_data, train_label, test_label)
