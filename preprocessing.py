import numpy as np
import keras
from keras.layers import *
from keras.models import *
from keras.optimizers import *
import keras.backend as K

from scipy import stats
from sklearn.model_selection import StratifiedKFold

if __name__ == "__main__":

    demo = np.loadtxt('demo.txt');
    
    L = 1000
    S = 0
    data = np.zeros((demo.shape[0],1,L,22,1));
    label = np.zeros((demo.shape[0],));

    # load all data
    idx = 0
    data_all = None

    for i in range(demo.shape[0]):
        subject_string = format(int(demo[i,0]),'06d')
        #print(subject_string)
        filename_full = '/fs/neurosci01/qingyuz/hcp/hcp_tc_npy_22/'+subject_string+'_cortex.npy'
        full_sequence = np.load(filename_full);
        
        if full_sequence.shape[1] < S+L:
            continue
        
        full_sequence = full_sequence[:,S:S+L];
        z_sequence = stats.zscore(full_sequence, axis=1)
        
        if data_all is None:
            data_all = z_sequence
        else:
            data_all = np.concatenate((data_all, z_sequence), axis=1)

        data[idx,0,:,:,0] = np.transpose(z_sequence)
        label[idx] = demo[i,1]
        idx = idx + 1
 
    # compute adj matrix
    n_regions = 22
    A = np.zeros((n_regions, n_regions))
    for i in range(n_regions):
        for j in range(i, n_regions):
            if i==j:
                A[i][j] = 1
            else:
                A[i][j] = abs(np.corrcoef(data_all[i,:], data_all[j,:])[0][1]) # get value from corrcoef matrix
                A[j][i] = A[i][j]

    np.save('data/adj_matrix.npy', A)

    # split train/test and save data

    data = data[:idx]
    label = label[:idx]
    print(data.shape)
    skf = StratifiedKFold(n_splits=5,shuffle=True)
    fold = 1
    for train_idx, test_idx in skf.split(data, label):
        train_data = data[train_idx]
        train_label = label[train_idx]
        test_data = data[test_idx]
        test_label = label[test_idx] 

        filename = 'data/train_data_'+str(fold)+'.npy'
        np.save(filename,train_data)
        filename = 'data/train_label_'+str(fold)+'.npy'
        np.save(filename,train_label)
        filename = 'data/test_data_'+str(fold)+'.npy'
        np.save(filename,test_data)
        filename = 'data/test_label_'+str(fold)+'.npy'
        np.save(filename,test_label)

        fold = fold + 1
  
