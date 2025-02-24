import numpy as np

train_data = np.load('x_train.npy')
print(train_data.shape)
val_data = np.load('x_valid.npy')
train_label = np.load('y_train.npy')
val_label = np.load('y_valid.npy')

all_data = np.concatenate((train_data, val_data), 0)
all_label = np.concatenate((train_label, val_label), 0)

print(all_data.shape)

np.save('All_train.npy', all_data)
np.save('All_label.npy', all_label)