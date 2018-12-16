from keras.models import Model, Sequential
from keras.layers import Dense, Input
from keras.datasets import fashion_mnist
from keras.regularizers import l1
from keras.optimizers import Adam
from keras import losses 
from keras.utils import to_categorical
import sklearn
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


def plot_loss_accuracy(history):
    historydf = pd.DataFrame(history.history, index=history.epoch)
    loss = historydf.iloc[:,1]
    fig, ax = plt.subplots()
    ax.plot(loss,'r:', label='loss')
    legend = ax.legend(loc=0, shadow=False, fontsize='medium')
    plt.show()
	

	
def loss_count(y, y_pred):
	count = 0
	for i in range (0,len(y)):
		if (y[i] != y_pred[i]):
			count += 1
	return count

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
### Normalise the data and change data type

x_train = np.reshape(x_train, [x_train.shape[0], np.prod(x_train.shape[1:])])
x_test  = np.reshape(x_test, [x_test.shape[0], np.prod(x_test.shape[1:])]) 

x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')
x_train = x_train/255.0
x_test =  x_test/255.0
Node_3 = len(np.unique(y_train))

# Convert to labels
train_Y = to_categorical(y_train, Node_3)
test_Y  = to_categorical(y_test, Node_3) 

# Split the training data into train and validation data
train_X, valid_X, train_label, valid_label = train_test_split(x_train, train_Y, test_size = 0.2)

#define nodes for the neural network
Node_1 = 64 # number of nodes in the first hidden layer
Node_2 = 32 # number of nodes in the second hidden layer
Node_3 = len(np.unique(y_train))
input_0 = train_X.shape[1]

# create the neural_network
model = Sequential()
model.add(Dense(Node_1, input_dim=input_0, activation='tanh'))
model.add(Dense(Node_2, activation='tanh'))
# model.add(Dense(Node_2, activation='relu'))
model.add(Dense(Node_3, activation='softmax'))

model.compile(optimizer='Adam', loss=losses.categorical_crossentropy, metrics=['accuracy'])

loss_count_1 = []

for i in range(1,30):
	history = model.fit(x=train_X, y=train_label, verbose=0, epochs=i, validation_data=(valid_X, valid_label))
	y_pred = model.predict_classes(x_test)
	loss_count_1.append(loss_count(y_pred, y_test)*1.0/len(y_pred))

plot_loss_accuracy(history)

fig, ax = plt.subplots()
ax.plot(loss_count_1, 'r:', label='')
legend = ax.legend(loc=0, shadow=False, fontsize='medium')
plt.show()

