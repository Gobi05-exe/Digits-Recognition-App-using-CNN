#importing libraries
import tkinter as tk
from io import BytesIO
from PIL import Image
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
        ps_img=self.canvas.postscript()
        img=Image.open(BytesIO(ps_img.encode('utf-8')))
        img.save('img.jpg')
        
        
#CNN model to recognize the digit
class Conv_Neural_Network:
    def __init__(self,img_ps):
        pass

    def get_digit(self):
        pass
        
if __name__=='__main__':
    root=tk.Tk()
    app=Digit_Recognizer(root)
    root.mainloop()
