from keras.models import load_model
from keras import utils
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.datasets import mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data() 

model = load_model('digit_reader.model')
z_test = []

img = image.load_img('01-4.png', color_mode='grayscale')
x = utils.normalize(img, axis=1)
z_test.append(x)

the_test = z_test

predictions = model.predict([the_test])

#print(the_test[0])
#print(predictions)
print('I predict ' + str(np.argmax(predictions[0])))

#plt.imshow(x_test[0], cmap='gray')
#plt.show()
