import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from net.st_gcn_lstm import Model
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

###### setup model & data
net = Model(1,1,None,True, device)
net.to(device)

criterion = nn.BCELoss() #CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=0.001)

#state_dict = torch.load('checkpoint.pth')
#net.load_state_dict(state_dict)

train_data = np.load('data/train_data_1200_1.npy')
train_label = np.load('data/train_label_1200_1.npy')
test_data = np.load('data/test_data_1200_1.npy')
test_label = np.load('data/test_label_1200_1.npy')

print(train_data.shape)
print(test_data.shape)

###### start training model
training_loss = 0.0

for epoch in range(100001): # number of mini-batches
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
        layer_num = 1
        # for edge_importances in net.edge_importance:
        #     edge_imp = torch.squeeze(edge_importances.data).cpu().numpy()
        #     filename = "output/edge_importance/edge_imp_layer_"+str(layer_num) + "_epoch_" + str(epoch)
        #     np.save(filename, edge_imp)
        #     layer_num += 1
        #print(torch.squeeze(net.edge_importance[0].data).cpu().numpy().shape)

    # validate on test subjects by voting
    if epoch % 1000 == 0:    # print every K mini-batches
        # net.eval()
        idx_batch = np.random.permutation(int(test_data.shape[0]))
        idx_batch = idx_batch[:int(batch_size)]       

        test_label_batch = test_label[idx_batch]
        prediction = np.zeros((test_data.shape[0],))
        voter = np.zeros((test_data.shape[0],))
        for v in range(TS):
            idx = np.random.permutation(int(test_data.shape[0]))
            
            # testing also performed batch by batch (otherwise it produces error)
            for k in range(int(test_data.shape[0]/batch_size)):
                idx_batch = idx[int(batch_size*k):int(batch_size*(k+1))]     
                
                # construct random sub-sequences from a batch of test subjects
                test_data_batch = np.zeros((batch_size,1,W,22,1))
                for i in range(64):
                    r1 = random.randint(0, test_data.shape[2]-W)
                    test_data_batch[i]  = test_data[idx_batch[i],:,r1:r1+W,:,:] 
  
                test_data_batch_dev = torch.from_numpy(test_data_batch).float().to(device)
                outputs = net(test_data_batch_dev)
                outputs = outputs.data.cpu().numpy()

                prediction[idx_batch] = prediction[idx_batch] + outputs[:,0];
                voter[idx_batch] = voter[idx_batch] + 1;

        # average voting
        prediction = prediction / voter;
        #print(prediction)
        print(sum((prediction>0.5)==test_label) / test_label.shape[0])  

        torch.save(net.state_dict(), 'checkpoint.pth')   
        # net.train()
        
        
