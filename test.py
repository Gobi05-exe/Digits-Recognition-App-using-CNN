from PIL import Image,ImageOps
from tensorflow.keras.saving import load_model
import numpy as np
img = Image.open('temp/img.jpg',)
img=ImageOps.grayscale(img)
img=ImageOps.invert(img)
img=img.resize((28,28))
img=np.expand_dims(img,axis=0)
model=load_model('model/cnn.h5')
result=model.predict(img)
print(np.argmax(result))