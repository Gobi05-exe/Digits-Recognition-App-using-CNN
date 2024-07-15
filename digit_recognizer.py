#importing libraries
import tkinter as tk
from tkinter import messagebox
from io import BytesIO
from PIL import Image
import numpy as np
from PIL import Image,ImageOps
from tensorflow.keras.saving import load_model
from os import listdir,remove

class Digit_Recognizer:
    def __init__(self,root):
        #defining root window
        self.root=root
        self.root.geometry('400x530')
        self.root.resizable(width='False',height='False')
        self.root.title('Digit Recognizer')
        self.root.configure(background='#ADB1FF')
        self.root.iconbitmap('icon.ico')
        
        #Label above the canvas
        self.label=tk.Label(self.root,text='DRAW A DIGIT',font=('Calibri',25),bg='yellow')
        self.label.grid(row=0,column=0,columnspan=3,sticky='WENS',pady=5)
        
        #frame carrying the canvas
        self.frame=tk.LabelFrame(self.root,height=400,width=400)
        self.frame.grid(row=1,column=0,columnspan=3)
        
        #canvas
        self.canvas=tk.Canvas(self.frame,height=400,width=400)
        self.canvas.pack()
        
        #defining buttons
        
        self.clear_b=tk.Button(text='Clear',font=('Arial',15),command=self.clear)
        self.submit_b=tk.Button(text='Submit',font=('Arial',15),command=self.submit)
        self.exit_b=tk.Button(text='Exit',font=('Arial',15),command=self.root.quit)

        
        #placing buttons on screen
        self.clear_b.grid(row=2,column=0,pady=10)
        self.submit_b.grid(row=2,column=1,pady=10) 
        self.exit_b.grid(row=2,column=2,pady=10)
        
        #binding mouse events
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)
        
        #flag variable to denote state of the canvas
        self.isempty=True
        
        #getting the model
        self.model=Conv_Neural_Network()
        
    #defining drawing action
    def start_drawing(self,event):
        self.drawing=True
        self.last_x, self.last_y = event.x, event.y
    
    def draw(self,event):
        self.isempty=False
        if self.drawing:
            x, y = event.x, event.y
            self.canvas.create_line(self.last_x, self.last_y, x, y, fill='black', width=15,capstyle=tk.ROUND, joinstyle=tk.BEVEL)
            self.last_x, self.last_y = x, y
        
    def stop_drawing(self,event):
        self.drawing=False 
        
        
    #clear button action
    def clear(self):
        self.canvas.delete("all")
        self.isempty=True
        
    #submit button action
    def submit(self):
        if not self.isempty:
            ps_img=self.canvas.postscript() #getting the image from the canvas and converting it to a postscript file
            img=Image.open(BytesIO(ps_img.encode('utf-8'))) #encoding the image to binary data
            img.save('temp/img.jpg') #saving the image as a jpg file
            self.img=Image.open('temp/img.jpg') #opening the image as an Image object
            
            self.display_digit()
        
        else:
            self.warning=messagebox.showerror(title='Error',message='Draw a digit first!')    

    #creates a pop-up and displays the result    
    def display_digit(self):
        self.digit=self.model.predict_digit(self.img)
        self.popup=messagebox.showinfo(title='Prediction',message='You have drawn the digit:\n'+str(self.digit))
   
        
#CNN model to recognize the digit
class Conv_Neural_Network:
    
    def __init__(self):
        if 'cnn.h5' in listdir('model'):
            self.cnn=load_model('model/cnn.h5')
        else:
            self.cnn=self.get_model()

    #trains the model if it is not done already
    def get_model(self):
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
        cnn.save('model/cnn.h5')
        
        return cnn
    
    #preprocesses the image
    def preprocess_inp_img(self):
        self.inp_img=ImageOps.grayscale(self.inp_img)
        self.inp_img=ImageOps.invert(self.inp_img)
        self.inp_img=self.inp_img.resize((28,28))
        self.inp_img=np.expand_dims(self.inp_img,axis=0)
        
    def predict_digit(self,inp_img):
        self.inp_img=inp_img
        self.preprocess_inp_img()
        result=self.cnn.predict(self.inp_img)
        return int(np.argmax(result))
        
if __name__=='__main__':
    root=tk.Tk()
    app=Digit_Recognizer(root)
    root.mainloop()
    remove('temp/img.jpg')