import numpy as np
def covariance_matrx(data):
    n_regions = 22
    A = np.zeros((n_regions, n_regions))
    for i in range(n_regions):
        for j in range(i, n_regions):
            if i == j:
                A[i][j] = 1
            else:
                A[i][j] = abs(np.corrcoef(data[i, :], data[j, :])[0][1])  # get value from corrcoef matrix
                A[j][i] = A[i][j]
    upper_tri_flattened = A[np.triu_indices(22, k=0)]
    #print(upper_tri_flattened)
    return upper_tri_flattened




if __name__ == '__main__':
    train_data = np.load('data/train_data_1200_1.npy').squeeze()
    test_data = np.load('data/test_data_1200_1.npy').squeeze()
    train_label = np.load('data/train_label_1200_1.npy')
    test_label = np.load('data/test_label_1200_1.npy')
    train_data_covar = np.zeros((train_data.shape[0], 253))
    test_data_covar = np.zeros((test_data.shape[0], 253))
    for i in range(train_data.shape[0]):
        train_data_covar[i] = covariance_matrx(train_data[i].T)
        i += 1


    for i in range(test_data.shape[0]):
        test_data_covar[i] = covariance_matrx(test_data[i].T)
        i += 1
    np.save('data/train_data_1200_1_mlp.npy', train_data_covar)
    np.save('data/test_data_1200_1_mlp.npy', test_data_covar)