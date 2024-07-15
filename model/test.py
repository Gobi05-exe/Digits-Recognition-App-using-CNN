from tensorflow.keras.models import load_model
import numpy as np
from skimage import io
from skimage.transform import resize

model=load_model('cnn.h5')

for i in range(10):
    #img = io.imread('{}.jpg'.format(i), as_gray=True)
    img=resize(img,(28,28))
    img=np.expand_dims(img,axis=0)
    result=model.predict(img)
    print("Predicted:", int(np.argmax(result,axis=1)))