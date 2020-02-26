import matplotlib
import matplotlib.pyplot as plt

train_epoch_list = []
train_loss_list = []
train_acc_list = []

test_epoch_list = []
test_loss_list = []
test_acc_list = []

for line in open('output/mlp_baseline/training_output.txt'):
    line_split = line.split()
    train_epoch_list.append(int(line_split[1]))
    train_loss_list.append(float(line_split[3]))
    train_acc_list.append(float(line_split[5]))


for line in open('output/mlp_baseline/testing_output.txt'):
    line_split = line.split()
    test_epoch_list.append(int(line_split[1]))
    test_loss_list.append(float(line_split[3]))
    test_acc_list.append(float(line_split[5]))

font = {'family' : 'normal',
        'size'   : 12}

matplotlib.rc('font', **font)

fig, ax = plt.subplots(2,2)
fig.suptitle('MLP-Baseline Output')
ax[0, 0].plot(train_epoch_list, train_loss_list)
ax[0, 0].set_title("Training Loss Curve")

ax[0, 1].plot(test_epoch_list, test_loss_list)
ax[0, 1].set_title("Testing Loss Curve")

ax[1, 0].plot(train_epoch_list, train_acc_list)
ax[1, 0].set_title("Training Acc Curve")

ax[1, 1].plot(test_epoch_list, test_acc_list)
ax[1, 1].set_title("Testing Acc Curve")

plt.show()
