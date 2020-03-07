import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from net.st_gcn import Model
import random
from scipy import stats

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

###### **model parameters**
W = 128  # window size
TS = 64  # number of voters per test subject

###### **training parameters**
LR = 0.001  # learning rate
batch_size = 64


criterion = nn.BCELoss()  # CrossEntropyLoss()
#optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=0.001)

# state_dict = torch.load('checkpoint.pth')
# net.load_state_dict(state_dict)

train_data = np.load('data/train_data_1200_1.npy')
train_label = np.load('data/train_label_1200_1.npy')
test_data = np.load('data/test_data_1200_1.npy')
test_label = np.load('data/test_label_1200_1.npy')

print(train_data.shape)

###### start training model
training_loss = 0.0

for ts in [16]: #[2, 4, 8, 16, 32, 64, 128, 256]:#[50, 64, 75, 100, 128, 256, 1200]:
    TS = ts
    final_testing_accuracy = 0
    testing_acc_curr_fold = []
    # print('-'*80)
    # print("TS {}".format(TS))
    # print('-'*80)
    print("Window size set to {}\n".format(W))
    for fold in range(1, 6):
        print('-'*80)
        print("TS {}, Fold {}".format(TS, fold))
        print('-'*80)
        best_test_acc_curr_fold = 0
        best_test_epoch_curr_fold = 0
        best_edge_imp_curr_fold = []
        train_data = np.load('data/train_data_1200_'+str(fold)+'.npy')
        train_label = np.load('data/train_label_1200_'+str(fold)+'.npy')
        test_data = np.load('data/test_data_1200_'+str(fold)+'.npy')
        test_label = np.load('data/test_label_1200_'+str(fold)+'.npy')
        
        net = Model(1, 1, None, True)
        net.to(device)
        
        optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=0.001)

        for epoch in range(60001):  # number of mini-batches
            # select a random sub-set of subjects
            idx_batch = np.random.permutation(int(train_data.shape[0]))
            idx_batch = idx_batch[:int(batch_size)]

            # construct a mini-batch by sampling a window W for each subject
            train_data_batch = np.zeros((batch_size, 1, W, 22, 1))
            train_label_batch = train_label[idx_batch]

            for i in range(batch_size):
                r1 = random.randint(0, train_data.shape[2] - W)
                train_data_batch[i] = train_data[idx_batch[i], :, r1:r1 + W, :, :]

            train_data_batch_dev = torch.from_numpy(train_data_batch).float().to(device)
            train_label_batch_dev = torch.from_numpy(train_label_batch).float().to(device)
            #train_data_batch_dev = train_data_batch_dev.squeeze()
            # forward + backward + optimize
            optimizer.zero_grad()
            #net.hidden = net.init_hidden(batch_size)
            outputs = net(train_data_batch_dev)
            loss = criterion(outputs, train_label_batch_dev)
            loss.backward()
            optimizer.step()

            # print training statistics
            training_loss += loss.item()
            if epoch % 1000 == 0:  # print every T mini-batches
                # print(outputs)
                outputs = outputs.data.cpu().numpy() > 0.5
                train_acc = sum(outputs[:, 0] == train_label_batch) / train_label_batch.shape[0]
                print('[%d] training loss: %.3f training batch acc %f' % (epoch + 1, training_loss/1000, train_acc))
                training_loss = 0.0

            # validate on test subjects by voting
            if epoch % 1000 == 0:  # print every K mini-batches
                idx_batch = np.random.permutation(int(test_data.shape[0]))
                idx_batch = idx_batch[:int(batch_size)]

                test_label_batch = test_label[idx_batch]
                prediction = np.zeros((test_data.shape[0],))
                voter = np.zeros((test_data.shape[0],))
                for v in range(TS):
                    idx = np.random.permutation(int(test_data.shape[0]))

                    # testing also performed batch by batch (otherwise it produces error)
                    for k in range(int(test_data.shape[0] / batch_size)):
                        idx_batch = idx[int(batch_size * k):int(batch_size * (k + 1))]

                        # construct random sub-sequences from a batch of test subjects
                        test_data_batch = np.zeros((batch_size, 1, W, 22, 1))
                        for i in range(64):
                            r1 = random.randint(0, test_data.shape[2] - W)
                            test_data_batch[i] = test_data[idx_batch[i], :, r1:r1 + W, :, :]

                        test_data_batch_dev = torch.from_numpy(test_data_batch).float().to(device)
                        outputs = net(test_data_batch_dev)
                        outputs = outputs.data.cpu().numpy()

                        prediction[idx_batch] = prediction[idx_batch] + outputs[:, 0];
                        voter[idx_batch] = voter[idx_batch] + 1;


                # average voting
                prediction = prediction / voter;
                # print(prediction)
                test_acc = sum((prediction > 0.5) == test_label) / test_label.shape[0]
                print(sum((prediction > 0.5) == test_label) / test_label.shape[0])
                with open('output/testing_acc_st_gcn_folds/voters/TS'+str(TS)+'_fold_'+str(fold)+'.txt', 'a') as f:
                    f.write("Epoch {}, Accuracy {}\n".format(epoch, test_acc))
                

                if test_acc > best_test_acc_curr_fold:
                    best_test_acc_curr_fold = test_acc
                    best_test_epoch_curr_fold = epoch
                    tmp_edge_imp = []
                    for edge_importances in net.edge_importance:
                        edge_imp = torch.squeeze(edge_importances.data).cpu().numpy()
                        tmp_edge_imp.append(edge_imp)
                    best_edge_imp_curr_fold = tmp_edge_imp
                torch.save(net.state_dict(), 'checkpoint.pth')
        print("Best accuracy for TS {} and fold {} = {} at epoch = {}".format(TS, fold, best_test_acc_curr_fold, best_test_epoch_curr_fold))
        testing_acc_curr_fold.append(best_test_acc_curr_fold)
        layer_num = 1
        for edge_imp in best_edge_imp_curr_fold:
            filename = "output/edge_importance/edge_imp_layer_" + str(layer_num) + "_TS_" + str(TS) + "_fold_"+str(fold)
            np.save(filename, edge_imp)
            layer_num += 1
    print("TS {} completed. Final testing accuracy = {}".format(TS, np.mean(np.array(testing_acc_curr_fold))))
    with open('output/testing_acc_st_gcn_ts_final.txt', 'a') as f:
        f.write("TS {} completed. Final testing accuracy = {}\n".format(TS, np.mean(np.array(testing_acc_curr_fold))))




