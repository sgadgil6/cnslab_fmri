import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
#import matplotlib.pyplot as plt
import csv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class fMRI_LSTM(nn.Module):
    def __init__(self, hidden_dim, input_dim, target_size, batch_size):
        super(fMRI_LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)
        self.hidden_init = self.init_hidden(batch_size=batch_size)
        self.dropout = nn.Dropout(p=0.5)
        self.target_size = target_size

    def init_hidden(self, batch_size):
        return (torch.randn(1, batch_size, self.hidden_dim).to(device), torch.randn(1, batch_size, self.hidden_dim).to(device))

    def forward(self, x):
        #x = x.T
        lstm_out, _ = self.lstm(x) # You don't really need to pass hidden layer state
        # lstm_out, _ = self.lstm(x,self.hidden_init) # or for init hidden state.
        lstm_out = lstm_out.squeeze()[:, -1, :]
        out = self.dropout(lstm_out)
        linear_output = self.linear(out)
        #linear_output = self.dropo(linear_output)
        #mean_pooling_output = F.avg_pool1d(linear_output, kernel_size=1)
        final_output = torch.sigmoid(linear_output)
        return final_output

# def train(training_data, training_labels, training_sample_names, testing_data, testing_labels, testing_sample_names, num_timesteps):
#     model = fMRI_LSTM(256, 22, 1).to(device)
#     loss_function = nn.BCELoss()
#     optimizer = optim.Adagrad(model.parameters(), lr=0.1)
#     n_epochs = 300
#     with open('training_loss_accuracy.csv', mode='w') as wfile:
#         writer = csv.writer(wfile)
#         for epoch in range(n_epochs):
#             epoch_loss = 0.0
#             correct_pred = 0
#             result_dict = {}
#             for i in range(len(training_data)):
#                 model.train()
#                 training_sample = training_data[i][:, np.random.choice(training_data[i].shape[1], num_timesteps, replace=False)]
#                 x = torch.Tensor(training_sample).to(device)
#                 true_label = torch.FloatTensor([training_labels[i]]).to(device)
#                 model.zero_grad()
#                 model.hidden = model.init_hidden()
#                 predicted_prob = model(x)
#                 #print(predicted_prob, true_label)
#                 if predicted_prob > 0.5:
#                     predicted_label = 1
#                 else:
#                     predicted_label = 0
#                 #print("Train Prob {}".format(predicted_prob))
#                 #print(predicted_prob, true_label)
#                 loss = loss_function(predicted_prob, true_label)
#                 epoch_loss += loss.item()
#                 loss.backward()
#                 optimizer.step()
#                 if training_sample_names[i] not in result_dict.keys():
#                     result_dict[training_sample_names[i]] = ([predicted_label], true_label)
#                 else:
#                     result_dict[training_sample_names[i]][0].append(predicted_label)
#             #accuracy = test(model, training_data, training_labels)
#             hit_top_1 = []
#             print(result_dict)
#             for v in result_dict.values():
#                 predicted_val_list = v[0]
#                 actual_label = v[1]
#                 pred_label =  max(set(predicted_val_list), key=predicted_val_list.count)
#                 if pred_label == actual_label:
#                     hit_top_1.append(1)
#                 else:
#                     hit_top_1.append(0)
#
#             accuracy = (sum(hit_top_1) * 1.0) / len(hit_top_1)
#             print("Epoch: {}, loss: {}, Accuracy = {}".format(epoch, epoch_loss/len(training_data), accuracy))
#             writer.writerow(['{}'.format(epoch), '{}'.format(epoch_loss/len(training_data)), '{}'.format(accuracy)])
#
# def test(model, testing_data, testing_labels):
#     model.eval()
#     #print(model.parameters())
#     correct_pred = 0
#     with torch.no_grad():
#         for i in range(len(testing_data)):
#             x = torch.Tensor(testing_data[i]).to(device)
#             true_label = testing_labels[i]
#             predicted_prob = model.forward(x)
#             #print(predicted_prob)
#             if true_label == (predicted_prob > 0.5):
#                 correct_pred += 1
#     model.train()
#     return correct_pred/len(testing_labels)
#
#
# if __name__ == "__main__":
#     training_data = np.load('train_data_22_sex_100.npy')
#     training_labels = np.load('train_label_22_sex_100.pkl')['label']
#     training_sample_names = np.load('train_label_22_sex_100.pkl')['sample_name']
#     testing_data = np.load('test_data_22_sex_100.npy')
#     testing_labels = np.load('test_label_22_sex_100.pkl')['label']
#     testing_samples_names = np.load('test_label_22_sex_100.pkl')['sample_name']
#     print(sorted(training_sample_names))
#     num_timesteps = 25
#     train(training_data, training_labels, training_sample_names, testing_data, testing_labels, testing_samples_names, num_timesteps)
#     #print(testing_labels)

