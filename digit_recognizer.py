#importing libraries
import tkinter as tk
from io import BytesIO
from PIL import Image
import numpy as np

class Digit_Recognizer:
    def __init__(self,root):
        #defining root window
        self.root=root
        self.root.geometry('400x500')
        self.root.resizable(width='False',height='False')
        self.root.title('Digit Recognizer')
        self.root.configure(background='#ADB1FF')
        
        #frame carrying the canvas
        self.frame=tk.LabelFrame(self.root,height=400,width=400)
        self.frame.grid(row=0,column=0,pady=20,columnspan=3)
        
        #canvas
        self.canvas=tk.Canvas(self.frame,height=400,width=400)
        self.canvas.pack()
        
        #defining buttons
        
        self.clear_b=tk.Button(text='Clear',font=('Arial',15),command=lambda:self.canvas.delete("all"))
        self.submit_b=tk.Button(text='Submit',font=('Arial',15),command=self.submit)
        
        #placing buttons on screen
        self.clear_b.grid(row=1,column=0)
        self.submit_b.grid(row=1,column=2) 
        
        #binding mouse events
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)
        
        
    #defining drawing action
    def start_drawing(self,event):
        self.drawing=True
        self.last_x, self.last_y = event.x, event.y
    
    def draw(self,event):
        if self.drawing:
            x, y = event.x, event.y
            self.canvas.create_line(self.last_x, self.last_y, x, y, fill='black', width=15,capstyle=tk.ROUND, joinstyle=tk.BEVEL)
            self.last_x, self.last_y = x, y
        
    def stop_drawing(self,event):
        self.drawing=False
        
    #submit button action
    def submit(self):
        ps_img=self.canvas.postscript(colormode='gray') #getting the image from the canvas and converting it to a postscript file
        img=Image.open(BytesIO(ps_img.encode('utf-8'))) #encoding the image to binary data
        img.save('temp/img.jpg') #saving the image as a jpg file
        img=Image.open('temp/img.jpg') #opening the image as an Image object

    
    
        
    #creates a pop-up and displays the result    
    def display_result(self):
        pass
        
        
#CNN model to recognize the digit
class Conv_Neural_Network:
    #preprocesses the image
    def __init__(self,inp_img):
        self.inp_img=inp_img
        self.inp_img=self.inp_img.resize((28,28))
        self.inp_img=np.expand_dims(self.inp_img,axis=0)

    #trains the model if it is not done already
    def get_model(self):
        #make new attribute self.model
        pass
    
    def predict_digit(self):
        #make the prediction and return the digit
        pass
        
if __name__=='__main__':
    root=tk.Tk()
    app=Digit_Recognizer(root)
    root.mainloop()
