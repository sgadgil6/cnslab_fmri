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
W = 128 # window size
TS = 64 # number of voters per test subject

###### **training parameters**
LR = 0.001 # learning rate
batch_size = 64



#state_dict = torch.load('checkpoint.pth')
#net.load_state_dict(state_dict)

train_data = np.load('data/edge_imp_data.npy')
train_label = np.load('data/edge_imp_label.npy')
# test_data = np.load('data/test_data_1200_1.npy')
# test_label = np.load('data/test_label_1200_1.npy')

print(train_data.shape)
# print(test_data.shape)

for trial in range(10):
    ###### setup model
    net = Model(1,1,None,True)
    net.to(device)

    criterion = nn.BCELoss() #CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=0.001)
    ###### start training model
    training_loss = 0.0
    print("On trial {}".format(trial))
    for epoch in range(60001): # number of mini-batches
        # select a random sub-set of subjects
        
        #net.lstm_layer.hidden = net.lstm_layer.init_hidden(batch_size)
        net.train()
        idx_batch = np.random.permutation(int(train_data.shape[0]))
        idx_batch = idx_batch[:int(batch_size)]

        # construct a mini-batch by sampling a window W for each subject
        train_data_batch = np.zeros((batch_size,1,W,22,1))
        train_label_batch = train_label[idx_batch]
        for i in range(batch_size):
            r1 = random.randint(0, train_data.shape[2]-W)
            train_data_batch[i]  = train_data[idx_batch[i],:,r1:r1+W,:,:]

        train_data_batch_dev = torch.from_numpy(train_data_batch).float().to(device)
        train_label_batch_dev = torch.from_numpy(train_label_batch).float().to(device)

        # forward + backward + optimize
        optimizer.zero_grad()
        outputs = net(train_data_batch_dev)
        # print(outputs)
        loss = criterion(outputs, train_label_batch_dev)
        loss.backward()
        optimizer.step()

        # print training statistics
        training_loss += loss.item()
        if epoch % 1000 == 0:    # print every T mini-batches
            #print(outputs)
            outputs = outputs.data.cpu().numpy() > 0.5
            train_acc = sum(outputs[:,0]==train_label_batch) / train_label_batch.shape[0]
            print('[%d] training loss: %.3f training batch acc %f' %(epoch + 1, training_loss/1000, train_acc))
            training_loss = 0.0
        if epoch == 20000 or epoch == 40000 or epoch == 60000:
            
            for importance in net.edge_importance:
                edge_importances = importance*importance+torch.transpose(importance*importance,0,1)
                edge_imp = torch.squeeze(edge_importances.data).cpu().numpy()
                filename = "output/edge_importance/edge_imp_all_data_epoch_" + str(epoch) + "_trial_" + str(trial)
                np.save(filename, edge_imp)
            #print(torch.squeeze(net.edge_importance[0].data).cpu().numpy().shape)