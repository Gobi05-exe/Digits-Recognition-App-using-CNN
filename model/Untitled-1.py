#importing libraries
import tensorflow as tf
import numpy as np
import pandas as pd



#getting dataset
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()


#preprocessing
input_shape=(28,28,1)
x_train=x_train.reshape(len(x_train),28,28,1)
x_train=x_train/255
x_test=x_test.reshape(len(x_test),28,28,1)
x_test=x_test/255


#one hot encoding the labels
y_train=tf.one_hot(y_train.astype(np.int32),depth=10)
y_test=tf.one_hot(y_test.astype(np.int32),depth=10)


#defining values
batch_size=64
epochs=5


#initiating the cnn model
cnn=tf.keras.models.Sequential()


#input layer
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=5,padding='same',activation='relu',input_shape=(28,28,1)))


#first convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=5,padding='same',activation='relu'))
#pooling
cnn.add(tf.keras.layers.MaxPool2D())
#dropout
cnn.add(tf.keras.layers.Dropout(0.25))


#second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,padding='same',activation='relu'))
#pooling
cnn.add(tf.keras.layers.MaxPool2D())
#dropout
cnn.add(tf.keras.layers.Dropout(0.25))


#third convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,padding='same',activation='relu'))
#pooling
cnn.add(tf.keras.layers.MaxPool2D())
#dropout
cnn.add(tf.keras.layers.Dropout(0.25))


#flattening
cnn.add(tf.keras.layers.Flatten())
#full connection
cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))
cnn.add(tf.keras.layers.Dropout(0.5))


#output layer
cnn.add(tf.keras.layers.Dense(units=10,activation='softmax'))


#compiling the cnn
cnn.compile(optimizer=tf.keras.optimizers.RMSprop(epsilon=1e-08),loss='categorical_crossentropy',metrics=['acc'])


#training the model
cnn.fit(x_train,y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1)



#getting accuracy
print(cnn.evaluate(x_test,y_test))


#saving the model
cnn.save('cnn.h5')


